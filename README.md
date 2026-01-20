# NeST-S6: Nested Convolutional Spatiotemporal (PDE-aware) State-space Model for 5G Network Traffic Forecasting

NeST-S6 (“Nest S 6”) is a **nested-learning spatiotemporal forecasting model** designed for **grid-based cellular traffic prediction**.
It integrates a **fast per-step spatiotemporal predictor** (convolution + windowed 2D attention + **PDE-aware selective state-space updates**) with a **slow persistent memory** updated by a learned **Deep Optimizer**.

<p align="center">
  <img src="nests6-arch.png" width="90%" alt="NeST-S6 Architecture Diagram">
</p>

---

## Key Features

- **Nested Learning Memory**: A slow, persistent spatial memory improves robustness under **global/dynamic drift**.
- **PDE-aware SSM Core**: Spatially-varying selective state-space updates with stable exponential discretization.
- **Local Mixing + Windowed Attention**: Depthwise convolution and windowed attention for efficient 2D context.
- **End-to-end Grid Forecasting**: Predict full spatial kernels/patches at once (instead of scalar-per-pixel).
- **Autoregressive Rollout**: Built-in multi-step forecasting via recursive prediction.

---

## Problem Formulation (Kernel / Patch Forecasting)

Let $M = \{M_1, M_2, \ldots, M_T\}$ be a sequence of traffic grids where each $M_t \in \mathbb{R}^{H \times W}$.
For each spatial position $(i, j)$, we extract a spatiotemporal input kernel:

$$
X^{(i,j)} \in \mathbb{R}^{T \times K \times K}
$$

The goal is to predict the **future spatial kernel** at the next time step:

$$
Y^{(i,j)} = M_{t+1}[i-r : i+r,\; j-r : j+r] \in \mathbb{R}^{K \times K},
\quad r=(K-1)/2
$$

We learn parameters $\theta$ by minimizing a forecasting loss:

$$
\min_{\theta} \; L\big(f_{\theta}(X^{(i,j)}),\, Y^{(i,j)}\big)
$$

---

## Model Architecture

NeST-S6 is a **nested-learning spatiotemporal predictor** with two interacting loops:

- **Fast Learner**: per-step dynamics (local mixing + windowed attention + a spatial selective state-space core)
- **Slow Learner**: a persistent spatial memory updated by a learned “Deep Optimizer”

Below we summarize the *core math* (same notation as the paper section you referenced), including tensor shapes.

### 1) Fast Learner: NeST-S6 Block (Windowed Attention + S-PDE SSM Core)

#### Inputs and stem projection

Given a traffic grid sequence $\mathbf{x}_t \in \mathbb{R}^{B \times 1 \times H \times W}$, we build a **2-channel** per-step input from the current frame and its temporal difference:

$$
\mathbf{u}_t = \text{concat}(\mathbf{x}_t,\; \mathbf{x}_t - \mathbf{x}_{t-1}) \in \mathbb{R}^{B \times 2 \times H \times W}.
$$

A convolutional stem projects $\mathbf{u}_t$ into a latent context:

$$
\mathbf{z}_t \in \mathbb{R}^{B \times D \times H \times W}.
$$

Each **NeST-S6 Block** then applies: (i) depthwise convolution + SiLU for local mixing, (ii) **windowed 2D attention** for local context, and (iii) an **S-PDE SSM core** update (S for *spatial*, PDE-aware selective state-space update).

#### S-PDE SSM Core (PDE-aware selective state-space update)

Let $\mathbf{x}\in\mathbb{R}^{B\times D_{\text{in}}\times H\times W}$ be the per-layer input to the SSM (after attention/mixing).
We maintain a **per-pixel state**:

$$
\mathbf{h}\in\mathbb{R}^{B\times D_{\text{in}}\times D_s\times H\times W}.
$$

**Dynamic low-rank parameter generation.** A $1\times 1$ convolution produces rank-$r$ coefficient maps:

$$
[\Delta_r,\, \mathbf{b}_r,\, \mathbf{c}_r] = \text{Conv}_{1\times 1}(\mathbf{x}),
\qquad
\Delta_r,\mathbf{b}_r,\mathbf{c}_r \in \mathbb{R}^{B\times r\times H\times W}.
$$

The per-channel step size is lifted to $D_{\text{in}}$ and squashed:

$$
\Delta = \sigma(\text{Lift}_{1\times 1}(\Delta_r)) \in \mathbb{R}^{B\times D_{\text{in}}\times H\times W}.
$$

We use learnable low-rank bases $\bar{\mathbf{B}},\bar{\mathbf{C}}\in\mathbb{R}^{D_{\text{in}}\times D_s}$, factorized as:

$$
\bar{\mathbf{B}} = \mathbf{B}_{\text{factor}} \cdot \mathbf{B}_{\text{out}},
\qquad
\bar{\mathbf{C}} = \mathbf{C}_{\text{factor}} \cdot \mathbf{C}_{\text{out}}.
$$

We then expand the rank-$r$ coefficient maps into full **spatially-varying** tensors ($\text{dyn}$ = dynamic, varies across $(h,w)$):

$$
(\mathbf{B}_{\text{dyn}})_{b,d,s,h,w}
= \sum_{k=1}^{r} (\mathbf{b}_r)_{b,k,h,w}\,(\mathbf{B}_{\text{factor}})_{d,k}\,(\mathbf{B}_{\text{out}})_{k,s},
$$

$$
(\mathbf{C}_{\text{dyn}})_{b,d,s,h,w}
= \sum_{k=1}^{r} (\mathbf{c}_r)_{b,k,h,w}\,(\mathbf{C}_{\text{factor}})_{d,k}\,(\mathbf{C}_{\text{out}})_{k,s}.
$$

This yields $\mathbf{B}_{\text{dyn}},\mathbf{C}_{\text{dyn}}\in\mathbb{R}^{B\times D_{\text{in}}\times D_s\times H\times W}$.
With learned gates $g_B=\sigma(\text{Conv}_{1\times 1}(\mathbf{x}))$ and $g_C=\sigma(\text{Conv}_{1\times 1}(\mathbf{x}))$, we define effective parameters (with a learned scale $\alpha$):

$$
\mathbf{B}_{\text{eff}} = \left(\bar{\mathbf{B}} + \alpha \tanh(\mathbf{B}_{\text{dyn}})\right)\odot g_B,
\qquad
\mathbf{C}_{\text{eff}} = \left(\bar{\mathbf{C}} + \alpha \tanh(\mathbf{C}_{\text{dyn}})\right)\odot g_C.
$$

**Stable transition parameterization.** The base transition is parameterized for stability as:

$$
\mathbf{A} = -\exp(\mathbf{A}_{\log}) \in \mathbb{R}^{D_{\text{in}}\times D_s}.
$$

Optionally, apply a low-rank $\mathbf{A}$ modulation (rank $r_A$) from $\mathbf{u}_A=\text{Conv}_{1\times 1}(\mathbf{x})\in\mathbb{R}^{B\times r_A\times H\times W}$:

$$
(\mathbf{A}_{\text{mod}})_{b,d,s,h,w}
= \sum_{k=1}^{r_A} (\mathbf{u}_A)_{b,k,h,w}\,(\mathbf{V}_A)_{k,d,s},
$$

and set (with learned $\beta$):

$$
\mathbf{A}_{\text{eff}} = \mathbf{A}\odot\left(1 + \beta\tanh(\mathbf{A}_{\text{mod}})\right).
$$

**Exponential discretization (per-pixel selective update).** The state update is:

$$
\mathbf{h}_{t+1} = \exp(\mathbf{A}_{\text{eff}}\odot \Delta)\odot \mathbf{h}_t \;+\; (\Delta\odot \mathbf{x})\odot \mathbf{B}_{\text{eff}}.
$$

The output features (summing over the state dimension) include a skip term:

$$
\mathbf{y}_t = \sum_{s=1}^{D_s}\mathbf{h}_{t+1}^{(s)}\odot \mathbf{C}_{\text{eff}}^{(s)} \;+\; \mathbf{D}_{\text{skip}}\odot \mathbf{x}.
$$

### 2) Slow Learner: Nested Memory Optimization (Persistent Spatial Memory + Deep Optimizer)

NeST-S6 maintains a long-term **spatial memory**:

$$
\mathbf{M}_t \in \mathbb{R}^{B \times D \times H \times W},
$$

updated by an outer-loop learned optimizer. At each step, the Fast Learner’s prediction is compared against the incoming frame to generate a **surprise** signal, based on the one-step error:

$$
\mathbf{e}_t = \mathbf{x}_{t+1} - \hat{\mathbf{x}}_{t+1}.
$$

We project this error into the model dimension using a small convolutional stem (a $3\times 3$ projection), yielding:

$$
\mathbf{S}_t = \text{Conv}_{3\times 3}\!\big(\text{concat}(\mathbf{e}_t,\mathbf{e}_t)\big)
\in \mathbb{R}^{B\times D\times H\times W}.
$$

The **Deep Optimizer (2D)** $\phi_{\text{opt}}$ (implemented with $1\times 1$ convolutions) produces a candidate memory write:

$$
\Delta \mathbf{M}_t = \phi_{\text{opt}}(\mathbf{z}_t, \mathbf{S}_t),
$$

and the memory is updated with a learned decay factor $\lambda$:

$$
\mathbf{M}_{t+1} = \lambda \mathbf{M}_t + (1 - \lambda)\Delta \mathbf{M}_t.
$$

Intuitively, the Slow Learner performs a form of **online adaptation** by updating a global memory more aggressively when the Fast Learner’s one-step prediction error is high.

### 3) Memory Injection (Context Gate)

Memory is fused back into the Fast Learner context through a learned **Context Gate**. Given $\mathbf{z}_t$ and $\mathbf{M}_t$:

$$
\mathbf{g}_t = \text{Conv}_{1\times 1}(\text{concat}(\mathbf{z}_t,\mathbf{M}_t)) \in \mathbb{R}^{B\times D\times H\times W},
$$

$$
\tilde{\mathbf{z}}_t = \mathbf{z}_t + \sigma(\mathbf{g}_t)\odot \mathbf{M}_t,
$$

and $\tilde{\mathbf{z}}_t$ is passed through the fast stack to obtain the next prediction $\hat{\mathbf{x}}_{t+1}$ via a convolutional prediction head.

### Notes on the name “NeST-S6”

- **NeST**: Nested Spatiotemporal (nested fast/slow learners).
- **S6**: indicates a *selective state-space* core (SSM) in the “S4/S5/S6-style” family, adapted here to **2D spatial grids** with per-pixel (spatially varying) parameters.

---

## Installation

### 1. Clone and install locally

```bash
git clone git@github.com:ZineddineBtc/NeST-S6.git
cd NeST-S6
pip install -e .
```

### 2. Dependencies

- Python 3.10
- PyTorch==2.6.0

---

## Usage

### Example: Quick test (forward + rollout)

```python
import torch
from nest_s6 import NeST_S6

# Input tensor: (B, T, H, W)
B, T, H, W = 2, 6, 20, 20
x = torch.randn(B, T, H, W)

model = NeST_S6(n_layers=2, d_model=48, d_state=8, d_conv=3, expand=2, attn_window=8, a_mod_rank=2)

with torch.no_grad():
    # last-frame prediction: (B, H, W)
    y_last = model(x)
    print("Last prediction:", y_last.shape)

    # full sequence prediction: (B, T, H, W)
    y_seq = model(x, return_sequence=True)
    print("Sequence prediction:", y_seq.shape)

    # autoregressive rollout (+2): (B, T+2, H, W)
    y_roll = model(x, steps_to_predict=2, return_sequence=True)
    print("Rollout prediction:", y_roll.shape)
```

---

## Training: Nested Memory

What’s *special* about NeST-S6 training is the **nested memory loop**:

- **Fast learner**: predicts the next frame $\hat{\mathbf{x}}_{t+1}$ from recent history.
- **Surprise signal**: compute prediction error $\mathbf{e}_t = \mathbf{x}_{t+1}-\hat{\mathbf{x}}_{t+1}$ and project it to a latent “surprise” $\mathbf{S}_t$.
- **Slow learner (memory write)**: update a persistent memory $\mathbf{M}_t$ using a learned optimizer and decay:
  $\mathbf{M}_{t+1}=\lambda\mathbf{M}_t+(1-\lambda)\Delta\mathbf{M}_t$, where $\Delta\mathbf{M}_t=\phi_{\text{opt}}(\mathbf{z}_t,\mathbf{S}_t)$.

In practice this means memory updates are **error-driven during training** (teacher-forced with access to $\mathbf{x}_{t+1}$), while at inference you often **disable writes / only decay** the memory during autoregressive rollout.

PyTorch-style sketch:

```python
import torch
import torch.nn.functional as F

model.train()
optimizer.zero_grad()

# batch_X: (B, T, H, W), batch_y: (B, H, W)  # next-frame target
pred = model(batch_X)  # internally uses fast dynamics + (optionally) updates slow memory from surprise

# Main one-step forecasting objective (e.g., Huber / SmoothL1)
loss_main = F.smooth_l1_loss(pred, batch_y, beta=1.0)

# Optional physics-inspired spatial regularizer (example: Laplacian consistency)
# loss_lapl = laplacian_loss(pred, batch_y)
# loss = loss_main + w_lapl * loss_lapl
loss = loss_main

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

If you want to reproduce the “nested-learning” behavior precisely, the key requirement is: **the memory update must consume a surprise derived from the one-step prediction error** during training; without that, you’re effectively training only the fast learner.

---

## Citation

If you use NeST-S6 in your research, please cite the (pending-review) paper and/or the repository:

```bibtex

@software{Bettouche2026NeSTS6Repo,
  title  = {NeST-S6 (Reference Implementation)},
  author = {Bettouche, Zineddine and Ali, Khalid and Fischer, Andreas and Kassler, Andreas},
  year   = {2026},
  url    = {https://github.com/ZineddineBtc/NeST-S6},
}
```

---


## Repository Structure

```
NeST-S6/
├── nest_s6/
│   ├── __init__.py          # Exports NeST_S6
│   ├── model.py             # Core NeST-S6 architecture
│   ├── __version__.py
├── nests6-arch.png
├── pyproject.toml
├── README.md
```

