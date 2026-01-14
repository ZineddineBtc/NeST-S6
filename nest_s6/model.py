"""
Full PDE-SSM Nested Model (Fast PDE-SSM + Slow DeepOptimizer)
Defaults are tuned for the "Full PDE variant" (use_a_mod enabled).
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic argument / utils
# -------------------------
@dataclass
class ModelArgs:
    n_layers: int
    d_model: int
    d_state: int
    d_conv: int
    expand: int = 2
    attn_window: int = 8
    a_mod_rank: int = 2  # default: enable A modulation (full PDE variant)
    dt_rank: int = None  # rank for B/C factorization

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank is None:
            self.dt_rank = max(4, math.ceil(self.d_model / 16))


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# RMS Norm
# -------------------------
class RMS_Norm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x shape: (..., D)
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


# -------------------------
# Fixed spatial differential ops (Laplacian / gradient)
# -------------------------
class FixedSpatialFilters(nn.Module):
    """
    Provide a small stack of fixed (or optionally learnable) differential filters:
    - laplacian (3x3)
    - grad_x, grad_y (sobel-ish)
    Returns a concatenated tensor of these features.
    """

    def __init__(self, learnable=False):
        super().__init__()
        # 3x3 laplacian kernel
        lap = (
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # simple sobel-like grads
        gx = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gy = gx.transpose(-1, -2)

        kernels = torch.cat([lap, gx, gy], dim=0)  # (3,1,3,3)
        self.register_buffer("kernels", kernels)  # fixed by default

        if learnable:
            # small learnable per-kernel scalar
            self.scale = nn.Parameter(torch.ones(kernels.shape[0]))
        else:
            self.scale = None

    def forward(self, x):
        # x: (B, 1, H, W) or (B, C, H, W)
        # Apply the same spatial kernel to each channel (depthwise), then average over channels.
        # Ensure kernel is same type & device as input
        kernels = self.kernels.to(dtype=x.dtype, device=x.device)
        B, C, H, W = x.shape
        feats = []
        for k in range(kernels.shape[0]):
            k_mat = kernels[k : k + 1, :, :, :]
            # Depthwise conv: weight is (C,1,kh,kw), groups=C => each channel gets the same kernel.
            weight = k_mat.repeat(C, 1, 1, 1)  # (C,1,3,3)
            out = F.conv2d(x, weight, padding=1, groups=C)  # (B, C, H, W)
            # Reduce channel dimension via mean
            out = out.mean(dim=1, keepdim=True)  # (B,1,H,W)
            if self.scale is not None:
                out = out * self.scale[k]
            feats.append(out)
        return torch.cat(feats, dim=1)  # (B, 3, H, W)


# -------------------------
# Windowed local attention
# -------------------------
class WindowedAttention2D(nn.Module):
    def __init__(self, dim, heads=4, window_size=8):
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        ws = self.window_size
        if ws is None or (H <= ws and W <= ws):
            qkv = self.to_qkv(x).chunk(3, dim=1)
            q, k, v = map(lambda t: t.view(B, self.heads, C // self.heads, H * W), qkv)
            attn = torch.einsum("bhcd,bhce->bhde", q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum("bhde,bhce->bhcd", attn, v)
            out = out.reshape(B, C, H, W)
            return self.proj_out(out)

        # Pad so H and W are multiples of ws
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        x_p = F.pad(x, (0, pad_w, 0, pad_h))
        B, C, H_p, W_p = x_p.shape
        n_h = H_p // ws
        n_w = W_p // ws

        # unfold into windows: use reshape trick
        x_windows = x_p.view(B, C, n_h, ws, n_w, ws).permute(0, 2, 4, 1, 3, 5)
        x_windows = x_windows.reshape(B * n_h * n_w, C, ws, ws)  # (B*n_w*n_h, C, ws, ws)

        # per-window attention
        qkv = self.to_qkv(x_windows).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(-1, self.heads, C // self.heads, ws * ws), qkv)
        attn = torch.einsum("bhcd,bhce->bhde", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhde,bhce->bhcd", attn, v)
        out = out.reshape(-1, C, ws, ws)

        # fold windows back
        out = out.view(B, n_h, n_w, C, ws, ws).permute(0, 3, 1, 4, 2, 5)
        out = out.reshape(B, C, n_h * ws, n_w * ws)
        out = out[:, :, :H, :W]
        return self.proj_out(out)


# -------------------------
# Efficient PDE-aware SSM Block (with dynamic low-rank B/C)
# -------------------------
class PDE_SSM_Block(nn.Module):
    """
    PDE-aware SSM with:
    - Dynamic Δ(x)
    - Dynamic low-rank B(x), C(x)
    - Optional low-rank per-pixel A(x)
    - Stable exponential state updates
    """

    def __init__(self, args: ModelArgs, use_a_mod=False):
        super().__init__()
        self.args = args
        D_in = args.d_inner
        D_s = args.d_state
        r = args.dt_rank
        self.use_a_mod = use_a_mod and (args.a_mod_rank > 0)
        a_mod_rank = args.a_mod_rank if self.use_a_mod else 0

        # Input projection generates 3*r channels (delta_r, B_r, C_r)
        self.x_proj = nn.Conv2d(D_in, r * 3, kernel_size=1)

        # Δ lifting (small r -> full channels)
        self.delta_lift = nn.Conv2d(r, D_in, kernel_size=1)

        # simple multiplicative gates
        self.gate_B = nn.Conv2d(D_in, D_in, kernel_size=1)
        self.gate_C = nn.Conv2d(D_in, D_in, kernel_size=1)

        # Base factors for B and C (low-rank factorization)
        self.B_factor = nn.Parameter(torch.randn(D_in, r) * 0.05)
        self.B_out = nn.Parameter(torch.randn(r, D_s) * 0.05)

        self.C_factor = nn.Parameter(torch.randn(D_in, r) * 0.05)
        self.C_out = nn.Parameter(torch.randn(r, D_s) * 0.05)

        # Base A: kept negative via exp()
        self.A_log = nn.Parameter(torch.randn(D_in, D_s) * -3.0)

        # Dynamic low-rank A modulation
        if self.use_a_mod:
            ra = a_mod_rank
            self.A_mod_proj = nn.Conv2d(D_in, ra, kernel_size=1)
            self.A_mod_V = nn.Parameter(torch.randn(ra, D_in, D_s) * 0.05)
            self.A_mod_scale = nn.Parameter(torch.tensor(0.02))
        else:
            self.A_mod_proj = None

        # Small skip for stability
        self.D_skip = nn.Parameter(torch.ones(D_in) * 0.02)

        # Scale for dynamic B/C modulation
        self.dynamic_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, h_prev_flat):
        # x: (B, D_in, H, W)
        B, D_in, H, W = x.shape
        D_s = self.args.d_state
        r = self.args.dt_rank

        # ----- 1. PROJECT FOR PARAMS -----
        params = self.x_proj(x)  # (B,3*r,H,W)
        delta_r, B_r, C_r = torch.split(params, [r, r, r], dim=1)

        # ----- 2. BUILD Δ(x) -----
        delta = torch.sigmoid(self.delta_lift(delta_r))  # (B, D_in, H, W)

        # ----- 3. GATES -----
        gateB = torch.sigmoid(self.gate_B(x))
        gateC = torch.sigmoid(self.gate_C(x))

        # ----- 4. STATIC LOW-RANK BASES -----
        # B_base, C_base are (D_in, D_s)
        B_base = (self.B_factor @ self.B_out).view(D_in, D_s)
        C_base = (self.C_factor @ self.C_out).view(D_in, D_s)

        # reshape to broadcast
        B_base = B_base.view(1, D_in, D_s, 1, 1)
        C_base = C_base.view(1, D_in, D_s, 1, 1)

        # ----- 5. DYNAMIC LOW-RANK MODULATION -----
        # B_r, C_r: (B, r, H, W)
        # Compute dynamic term: (B, D_in, D_s, H, W)
        B_dyn = torch.einsum("b r h w, d r, r s -> b d s h w", B_r, self.B_factor, self.B_out)
        C_dyn = torch.einsum("b r h w, d r, r s -> b d s h w", C_r, self.C_factor, self.C_out)

        # Small scale + gating
        B_eff = (B_base + self.dynamic_scale * torch.tanh(B_dyn)) * gateB.unsqueeze(2)
        C_eff = (C_base + self.dynamic_scale * torch.tanh(C_dyn)) * gateC.unsqueeze(2)

        # ----- 6. BUILD A(x) -----
        A = -torch.exp(self.A_log).view(1, D_in, D_s, 1, 1)

        if self.use_a_mod:
            U = self.A_mod_proj(x)  # (B, ra, H, W)
            V = self.A_mod_V  # (ra, D_in, D_s)
            A_mod = torch.einsum("b r h w, r d s -> b d s h w", U, V)
            A = A * (1 + self.A_mod_scale * torch.tanh(A_mod))

        # ----- 7. STATE UPDATE -----
        h_prev = h_prev_flat.view(B, D_in, D_s, H, W)

        delta_u = delta.unsqueeze(2)
        trans_mul = torch.exp(A * delta_u)

        x_u = x.unsqueeze(2)
        delta_B_u = (delta_u * x_u) * B_eff

        h_new = trans_mul * h_prev + delta_B_u

        # ----- 8. OUTPUT Y -----
        y = torch.sum(h_new * C_eff, dim=2)  # (B, D_in, H, W)
        y = y + x * self.D_skip.view(1, -1, 1, 1)

        # flatten
        h_new_flat = h_new.view(B, D_in * D_s, H, W)
        return y, h_new_flat


# -------------------------
# Spatial Block integrating attention and SSM
# -------------------------
class Spatial_PDE_SSM_Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        Dm = args.d_model
        Di = args.d_inner

        self.in_proj = nn.Conv2d(Dm, Di * 2, kernel_size=1)
        self.depth_conv = nn.Conv2d(
            Di,
            Di,
            kernel_size=args.d_conv,
            padding=(args.d_conv - 1) // 2,
            groups=Di,
        )
        self.attn = WindowedAttention2D(Di, heads=4, window_size=args.attn_window)
        self.attn_norm = RMS_Norm(Di)
        self.norm = RMS_Norm(Dm)

        # SSM block with optional A modulation (enabled by default via ModelArgs)
        use_a_mod = args.a_mod_rank > 0
        self.ssm = PDE_SSM_Block(args, use_a_mod=use_a_mod)

        self.out_proj = nn.Conv2d(Di, Dm, kernel_size=1)
        self.ssm_scale = nn.Parameter(torch.tensor(0.08))

    def forward(self, x, h_prev_flat):
        # x: (B, Dm, H, W)
        residual = x
        # RMS norm expects last dim D in (..., D). So permute
        x_p = x.permute(0, 2, 3, 1)
        x_p = self.norm(x_p)
        x = x_p.permute(0, 3, 1, 2)

        x_and_res = self.in_proj(x)
        x, res = torch.split(x_and_res, [self.args.d_inner, self.args.d_inner], dim=1)

        x = self.depth_conv(x)
        x = F.silu(x)

        # attention
        xp = x.permute(0, 2, 3, 1)
        xp = self.attn_norm(xp)
        x = xp.permute(0, 3, 1, 2)

        x = self.attn(x)

        y, h_new_flat = self.ssm(x, h_prev_flat)
        y = self.ssm_scale * y
        y = y * F.silu(res)

        out = self.out_proj(y) + residual
        return out, h_new_flat


# -------------------------
# Deep Optimizer (The "Slow" Learner)
# -------------------------
class DeepOptimizer2D(nn.Module):
    """
    The 'Outer Loop'.
    Observes the prediction error (Surprise) of the Fast Model
    and generates a memory update.
    """

    def __init__(self, d_model):
        super().__init__()
        # Input: 2 * d_model (Current State + Surprise/Error)
        # We use 1x1 convs to keep it efficient and spatial
        self.net = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.Tanh(),  # Tanh to bound the memory updates
        )

    def forward(self, context, surprise):
        combined = torch.cat([context, surprise], dim=1)
        return self.net(combined)


# -------------------------
# Top-level Nested PDE-SSM Model
# -------------------------
class NeST_S6(nn.Module):
    def __init__(
        self,
        n_layers=2,
        d_model=48,
        d_state=8,
        d_conv=3,
        expand=2,
        attn_window=8,
        a_mod_rank=2,
    ):
        super().__init__()
        self.args = ModelArgs(
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            attn_window=attn_window,
            a_mod_rank=a_mod_rank,
        )

        # 1. The Fast Learner Components
        self.input_proj = nn.Conv2d(2, d_model, kernel_size=3, padding=1)  # frame + diff
        self.layers = nn.ModuleList([Spatial_PDE_SSM_Block(self.args) for _ in range(n_layers)])
        self.norm_f = RMS_Norm(d_model)
        self.output_head = nn.Conv2d(d_model, 1, kernel_size=1)

        # Explicit laplacian/gradient filters to append as channels to the SSM input optionally
        self.fixed_filters = FixedSpatialFilters(learnable=False)

        # 2. NESTED COMPONENTS
        # The Persistent Memory (Slow Loop)
        self.memory_decay = nn.Parameter(torch.tensor(0.95))  # Slow decay

        # The Optimizer (The Bridge)
        self.deep_optimizer = DeepOptimizer2D(d_model)

        # Integration Gate: How much do we listen to long-term memory vs immediate input?
        self.context_gate = nn.Conv2d(d_model * 2, d_model, kernel_size=1)

    def forward(self, x, steps_to_predict=0, return_sequence: bool = False):
        """
        Args:
            x: (B, T, H, W) scalar grid per time step
            steps_to_predict: autoregressive steps
            return_sequence: if True, return (B, T (+steps), H, W). If False (default), return only the last frame (B, H, W).
        Returns:
            predictions: (B, T + steps, H, W) sequence if return_sequence else (B, H, W)
        """
        B, T, H, W = x.shape
        device = x.device

        # Initialize Fast States (Reset every sequence)
        # hidden states flattened for PDE_SSM_Block
        fast_states = [
            torch.zeros(B, self.args.d_inner * self.args.d_state, H, W, device=device)
            for _ in range(self.args.n_layers)
        ]

        # Initialize Slow Memory
        # Shape: (B, D_model, H, W)
        long_term_memory = torch.zeros(B, self.args.d_model, H, W, device=x.device)

        predictions = []

        for t in range(T):
            # --- 1. PREPARE INPUT ---
            curr = x[:, t, :, :].unsqueeze(1)  # (B,1,H,W)
            if t > 0:
                prev = x[:, t - 1, :, :].unsqueeze(1)
                diff = curr - prev
            else:
                diff = torch.zeros_like(curr)

            inp = torch.cat([curr, diff], dim=1)  # (B,2,H,W)
            current_context = self.input_proj(inp)  # (B, d_model, H, W)

            # --- 2. INJECT LONG-TERM MEMORY (Nested Step) ---
            combined_input = torch.cat([current_context, long_term_memory], dim=1)
            gated_context = self.context_gate(combined_input)

            # Add residual for stability
            active_input = current_context + torch.sigmoid(gated_context) * long_term_memory

            # --- 3. FAST LEARNER FORWARD PASS ---
            for i, layer in enumerate(self.layers):
                active_input, h_new = layer(active_input, fast_states[i])
                fast_states[i] = h_new

            # Decode Prediction
            out_feat = active_input.permute(0, 2, 3, 1)  # to NHWC
            out_feat = self.norm_f(out_feat)
            out_feat = out_feat.permute(0, 3, 1, 2)  # back to NCHW
            prediction = self.output_head(out_feat).squeeze(1)  # (B, H, W)

            predictions.append(prediction)

            # --- 4. CALCULATE SURPRISE & UPDATE MEMORY ---
            if t < T - 1:
                real_next_frame = x[:, t + 1, :, :]
                error = (real_next_frame - prediction).unsqueeze(1)  # (B, 1, H, W)

                # Project error to model dimension using input_proj (recycle)
                surprise_signal = self.input_proj(torch.cat([error, error], dim=1))

                # Ask the Optimizer
                mem_update = self.deep_optimizer(active_input.detach(), surprise_signal.detach())

                # Update Slow Memory
                long_term_memory = (self.memory_decay * long_term_memory) + (
                    (1 - self.memory_decay) * mem_update
                )

        # --- 5. AUTOREGRESSIVE PREDICTION ---
        if steps_to_predict > 0:
            current_frame = predictions[-1].unsqueeze(1)
            prev_frame = x[:, T - 1, :, :].unsqueeze(1)

            for _ in range(steps_to_predict):
                diff = current_frame - prev_frame
                inp = torch.cat([current_frame, diff], dim=1)

                current_context = self.input_proj(inp)

                # Memory decays naturally without new updates
                long_term_memory = self.memory_decay * long_term_memory

                combined_input = torch.cat([current_context, long_term_memory], dim=1)
                gated_context = self.context_gate(combined_input)
                active_input = current_context + torch.sigmoid(gated_context) * long_term_memory

                for i, layer in enumerate(self.layers):
                    active_input, h_new = layer(active_input, fast_states[i])
                    fast_states[i] = h_new

                out_feat = active_input.permute(0, 2, 3, 1)
                out_feat = self.norm_f(out_feat)
                out_feat = out_feat.permute(0, 3, 1, 2)
                prediction = self.output_head(out_feat).squeeze(1)

                predictions.append(prediction)

                prev_frame = current_frame
                current_frame = prediction.unsqueeze(1)

        pred_seq = torch.stack(predictions, dim=1)
        return pred_seq if return_sequence else pred_seq[:, -1]


# -------------------------
# Physics-informed / diagnostic losses
# -------------------------
def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def temporal_derivative_loss(pred_seq, target_seq):
    dt_pred = pred_seq[:, 1:, :, :] - pred_seq[:, :-1, :, :]
    dt_tgt = target_seq[:, 1:, :, :] - target_seq[:, :-1, :, :]
    return F.mse_loss(dt_pred, dt_tgt)


def laplacian_loss(pred, target, fixed_filters=None):
    if fixed_filters is None:
        fixed_filters = FixedSpatialFilters()
    pred_lap = fixed_filters(pred.unsqueeze(1))[:, 0, :, :]
    tgt_lap = fixed_filters(target.unsqueeze(1))[:, 0, :, :]
    return F.mse_loss(pred_lap, tgt_lap)


def energy_conservation_loss(pred_seq):
    sums = pred_seq.view(pred_seq.shape[0], pred_seq.shape[1], -1).sum(dim=-1)  # (B, T)
    diffs = sums[:, 1:] - sums[:, :-1]
    return torch.mean(diffs**2)


# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = default_device()
    B = 2
    T = 6
    H = 20
    W = 20
    model = NeST_S6(n_layers=2, d_model=48, d_state=8, d_conv=3, expand=2, attn_window=5, a_mod_rank=2)
    model.to(device)

    x = torch.randn(B, T, H, W, device=device)
    print("--- Nested PDE-SSM (Full PDE variant) Test ---")
    print("Param count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(x, return_sequence=True)
    print("Output shape (sequence):", out.shape)  # (B, T, H, W)

    # Autoregressive
    out_ar = model(x, steps_to_predict=2, return_sequence=True)
    print("Output shape (autoregressive +2):", out_ar.shape)  # (B, T+2, H, W)

    # simple data loss
    target = torch.randn_like(out)
    L = mse_loss(out, target)
    print("Sample MSE:", L.item())


