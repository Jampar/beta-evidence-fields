# Beta Evidence Fields 

[![Beta Field Unit Tests](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml/badge.svg)](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml)

A Python library for continuous probabilistic occupancy fields from range-sensor evidence.

Beta Evidence Fields (BEFs) represents occupancy as a continuous Beta distribution constructed directly from range observations. Returned measurements contribute occupied evidence around their endpoints, while observation rays contribute free-space evidence along their length. Gaussian spatial kernels provide a continuous representation, and temporal Gaussian weighting allows evidence to decay as observations age.


## Mathematical Definition

### Notation

Let the environment have spatial dimension $D$, with query position

```math
\mathbf{x}\in\mathbb{R}^{D}.
```

A set of $N$ range observations is evaluated at $M$ query points. The $i$-th observation is

```math
\mathcal{O}_i
=
\left(
\mathbf{o}_i,
\mathbf{d}_i,
\rho_i,
r_i,
t_i,
w_i
\right),
```

where:

- $\mathbf{o}_i\in\mathbb{R}^{D}$ is the sensor origin;
- $\mathbf{d}_i\in\mathbb{R}^{D}$ is a unit direction vector, with $\lVert\mathbf{d}_i\rVert_2=1$;
- $\rho_i\in\mathbb{R}_{\geq 0}$ is the measured range or finite ray length;
- $r_i\in\{0,1\}$ indicates whether the observation produced a return;
- $t_i\in\mathbb{R}$ is the acquisition time; and
- $w_i\in\mathbb{R}_{\geq 0}$ is the observation evidence weight.

The corresponding ray endpoint is

```math
\mathbf{e}_i
=
\mathbf{o}_i+\rho_i\mathbf{d}_i.
```

The field is evaluated at query locations

```math
\mathbf{X}
=
\left\{\mathbf{x}_j\right\}_{j=1}^{M},
\qquad
\mathbf{x}_j\in\mathbb{R}^{D},
```

and query time $t\in\mathbb{R}$.

### Spatial Gaussian kernel

For a positive-definite covariance matrix $\boldsymbol{\Sigma}\in\mathbb{R}^{D\times D}$, define the peak-normalised Gaussian kernel

```math
K
\left(
\mathbf{x},
\boldsymbol{\mu};
\boldsymbol{\Sigma}
\right)
=
\exp\!\left[
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^{\mathsf{T}}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right],
```

where $\mathbf{x}\in\mathbb{R}^{D}$ is the evaluation point and $\boldsymbol{\mu}\in\mathbb{R}^{D}$ is the kernel centre. The kernel is dimensionless and satisfies

```math
K
\left(
\boldsymbol{\mu},
\boldsymbol{\mu};
\boldsymbol{\Sigma}
\right)
=1.
```

### Temporal evidence weight

Let $\sigma_t^2>0$ be the temporal variance. The temporally weighted evidence contributed by observation $i$ at query time $t$ is

```math
W_i(t)
=
w_i
K_t
\left(
t,t_i;\sigma_t^2
\right),
```

with

```math
K_t
\left(
t,t_i;\sigma_t^2
\right)
=
\exp\!\left[
-\frac{(t-t_i)^2}{2\sigma_t^2}
\right].
```

Collecting all observation weights gives

```math
\mathbf{W}(t)
=
\begin{bmatrix}
W_1(t) & \cdots & W_N(t)
\end{bmatrix}^{\mathsf{T}}
\in\mathbb{R}^{N}.
```

### Occupied evidence

Let $\boldsymbol{\Sigma}_s\in\mathbb{R}^{D\times D}$ be the occupied-evidence covariance. The occupied evidence from observation $i$ at position $\mathbf{x}$ is

```math
k_{s,i}
\left(
\mathbf{x};
\boldsymbol{\Sigma}_s
\right)
=
r_i
K
\left(
\mathbf{x},
\mathbf{e}_i;
\boldsymbol{\Sigma}_s
\right).
```

Only observations with $r_i=1$ therefore contribute occupied evidence. Across all observations and query points, define

```math
[\mathbf{K}_s]_{ij}
=
k_{s,i}
\left(
\mathbf{x}_j;
\boldsymbol{\Sigma}_s
\right),
\qquad
\mathbf{K}_s\in\mathbb{R}^{N\times M}.
```

### Finite-ray free-space evidence

Let $\boldsymbol{\Sigma}_f\in\mathbb{R}^{D\times D}$ be the free-space covariance. The normalised finite-ray kernel for observation $i$ is

```math
L_i
\left(
\mathbf{x};
\boldsymbol{\Sigma}_f
\right)
=
\sqrt{
\frac{
\mathbf{d}_i^{\mathsf{T}}
\boldsymbol{\Sigma}_f^{-1}
\mathbf{d}_i
}{2\pi}
}
\int_{0}^{\rho_i}
K
\left(
\mathbf{x},
\mathbf{o}_i+s\mathbf{d}_i;
\boldsymbol{\Sigma}_f
\right)
\,\mathrm{d}s,
```

where $s\in[0,\rho_i]$ parameterises the finite observation ray.

To prevent a returned observation from simultaneously contributing substantial free-space evidence at its occupied endpoint, define the endpoint suppression function

```math
S_i
\left(
\mathbf{x};
\boldsymbol{\Sigma}_s
\right)
=
1-r_i
K
\left(
\mathbf{x},
\mathbf{e}_i;
\boldsymbol{\Sigma}_s
\right).
```

The free-space evidence from observation $i$ is then

```math
k_{f,i}
\left(
\mathbf{x};
\boldsymbol{\Sigma}_f,
\boldsymbol{\Sigma}_s
\right)
=
L_i
\left(
\mathbf{x};
\boldsymbol{\Sigma}_f
\right)
S_i
\left(
\mathbf{x};
\boldsymbol{\Sigma}_s
\right).
```

For a non-returned observation, $r_i=0$, so $S_i(\mathbf{x};\boldsymbol{\Sigma}_s)=1$ and the complete finite ray contributes free-space evidence. For a returned observation, $r_i=1$, so free-space evidence is smoothly suppressed near the returned surface.

At the returned endpoint,

```math
K
\left(
\mathbf{e}_i,
\mathbf{e}_i;
\boldsymbol{\Sigma}_s
\right)
=1,
```

and therefore

```math
k_{f,i}
\left(
\mathbf{e}_i;
\boldsymbol{\Sigma}_f,
\boldsymbol{\Sigma}_s
\right)
=0
\qquad
\text{when }r_i=1.
```

Across all observations and query points, define

```math
[\mathbf{K}_f]_{ij}
=
k_{f,i}
\left(
\mathbf{x}_j;
\boldsymbol{\Sigma}_f,
\boldsymbol{\Sigma}_s
\right),
\qquad
\mathbf{K}_f\in\mathbb{R}^{N\times M}.
```

### Beta evidence field

Let

```math
\boldsymbol{\alpha}_0,
\boldsymbol{\beta}_0
\in
\mathbb{R}_{>0}^{M}
```

be the prior occupied- and free-evidence parameters evaluated at the query points. The posterior Beta parameters are

```math
\begin{aligned}
\boldsymbol{\alpha}
\left(
\mathbf{X},t
\right)
&=
\boldsymbol{\alpha}_0
+
\mathbf{K}_s^{\mathsf{T}}\mathbf{W}(t),
\\[4pt]
\boldsymbol{\beta}
\left(
\mathbf{X},t
\right)
&=
\boldsymbol{\beta}_0
+
\mathbf{K}_f^{\mathsf{T}}\mathbf{W}(t).
\end{aligned}
```

Equivalently, at a single query point $\mathbf{x}$,

```math
\begin{aligned}
\alpha(\mathbf{x},t)
&=
\alpha_0(\mathbf{x})
+
\sum_{i=1}^{N}
W_i(t)
\,k_{s,i}
\left(
\mathbf{x};
\boldsymbol{\Sigma}_s
\right),
\\[4pt]
\beta(\mathbf{x},t)
&=
\beta_0(\mathbf{x})
+
\sum_{i=1}^{N}
W_i(t)
\,k_{f,i}
\left(
\mathbf{x};
\boldsymbol{\Sigma}_f,
\boldsymbol{\Sigma}_s
\right).
\end{aligned}
```

The occupancy probability at $\mathbf{x}$ and time $t$ is represented by

```math
p(\mathbf{x},t)
\sim
\mathrm{Beta}
\left(
\alpha(\mathbf{x},t),
\beta(\mathbf{x},t)
\right).
```

The expected occupancy probability is

```math
\mu(\mathbf{x},t)
=
\mathbb{E}[p(\mathbf{x},t)]
=
\frac{
\alpha(\mathbf{x},t)
}{
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)
}.
```

The occupancy variance is

```math
\mathrm{Var}[p(\mathbf{x},t)]
=
\frac{
\alpha(\mathbf{x},t)\beta(\mathbf{x},t)
}{
\left[
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)
\right]^2
\left[
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)+1
\right]
}.
```

The evidence concentration is

```math
\lambda(\mathbf{x},t)
=
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t).
```

Here, the mean $\mu(\mathbf{x},t)$ describes estimated occupancy, the variance describes uncertainty in that probability, and the concentration $\lambda(\mathbf{x},t)$ describes the total accumulated evidence strength.

### Spatial support

Let $\tau_s^2>0$ be the squared Mahalanobis support threshold. Spatial support indicates whether a query lies within the configured neighbourhood of at least one returned observation:

```math
s
\left(
\mathbf{x};
\boldsymbol{\Sigma}_s,
\tau_s^2
\right)
=
\bigvee_{i=1}^{N}
\left
\{
r_i=1
\;\land\;
(\mathbf{x}-\mathbf{e}_i)^{\mathsf{T}}
\boldsymbol{\Sigma}_s^{-1}
(\mathbf{x}-\mathbf{e}_i)
\leq
\tau_s^2
\right\}.
```

Support is a geometric coverage indicator rather than an additional probabilistic statistic. It identifies locations close to returned surfaces under the occupied-evidence covariance and chosen threshold.


## Features

- Dimension-agnostic continuous Beta occupancy fields
- Gaussian occupied and integrated free-space evidence
- Weighted spatial and temporal evidence
- Vectorised queries for occupancy, variance, concentration, and support

## Installation

Python 3.10 or later is required.

### pip

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/Jampar/beta-evidence-fields.git
```

Alternatively, clone the repository and install it locally:

```bash
git clone https://github.com/Jampar/beta-evidence-fields.git
cd beta-evidence-fields
pip install .
```

For an editable development installation:

```bash
pip install -e .
```

### Pixi

Pixi provides the reproducible development and testing environment used by the project.

Clone the repository and install the environment:

```bash
git clone https://github.com/Jampar/beta-evidence-fields.git
cd beta-evidence-fields
pixi install
```

Run the unit tests with:

```bash
pixi run test
```

## Basic Usage

```python
import numpy as np

from beta_evidence_field import (OccupancyField, OccupancyObservation, OccupancyPrior)


field       = OccupancyField(surface_covariance = 0.1**2*np.eye(2),
                             free_covariance    = 0.2**2*np.eye(2),
                             temporal_variance  = 10.0**2,
                             prior              = OccupancyPrior())


observation = OccupancyObservation(origin    = np.array([0.0, 0.0]),
                                   direction = np.array([1.0, 0.0]),
                                   range     = 2.0,
                                   returned  = True,
                                   time      = 0.0)

field.add([observation])

query = field.query(points = np.array([2.0, 0.0]), time   = 0.0)

print(query.alpha)
print(query.beta)
print(query.mean)
print(query.variance)
print(query.concentration)
print(query.support)
```
The field can then be constructed directly from the configuration:

```python
from beta_evidence_field import OccupancyField

field = OccupancyField.from_yaml("../config/example_2d_config.yaml")
```

Multiple locations can be evaluated simultaneously:

```python
points = np.array([[0.0, 0.0],
                   [1.0, 0.0],
                   [2.0, 0.0]])

query = field.query(points)
```

## Testing

Install the locked Pixi environment and run the unit tests with:

```bash
pixi install
pixi run test
```

The test suite covers Gaussian kernels, occupancy observations, priors, query statistics, and occupancy-field behaviour.
