# Beta Evidence Fields

[![Tests](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml/badge.svg)](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Jampar/beta-evidence-fields/graph/badge.svg)](https://codecov.io/gh/Jampar/beta-evidence-fields)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)

A Python library for continuous probabilistic occupancy fields constructed directly from range-sensor evidence.

Beta Evidence Fields (BEFs) represent occupied and free-space evidence with continuous Gaussian kernels and accumulate that evidence as the parameters of a Beta distribution. The representation is dimension-agnostic, supports vectorised queries, and normalises locally redundant observations in acquisition-state space.

## Features

- Continuous, dimension-agnostic Beta occupancy fields
- Gaussian occupied evidence around returned surfaces
- Analytically integrated free-space evidence along finite rays
- Smooth free-space suppression at returned endpoints
- Sampling-density normalisation in observation-state space
- Finite-support sparse redundancy evaluation
- Vectorised occupancy, variance, concentration, and support queries
- Configurable spatial, state, and temporal support

## Installation

Python 3.10 or later is required.

### pip

Install directly from GitHub:

```bash
pip install git+https://github.com/Jampar/beta-evidence-fields.git
```

Or install a local clone:

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

Pixi provides the reproducible development and testing environment used by the project:

```bash
git clone https://github.com/Jampar/beta-evidence-fields.git
cd beta-evidence-fields
pixi install
pixi run test
```

## Basic Usage

The repository includes an example configuration in `config/example_2d_config.yaml`:

```python
import numpy as np

from beta_evidence_field import OccupancyField, OccupancyObservation


field = OccupancyField.from_yaml("config/example_2d_config.yaml")

observation = OccupancyObservation(
    origin    = np.array([0.0, 0.0]),
    direction = np.array([1.0, 0.0]),
    range     = 2.0,
    returned  = True,
    time      = 0.0,
)

field.add([observation])

query = field.query(np.array([2.0, 0.0]))

print(query.mean)
print(query.variance)
print(query.concentration)
print(query.support)
```

Multiple positions can be queried simultaneously:

```python
points = np.array([[0.0, 0.0],
                   [1.0, 0.0],
                   [2.0, 0.0]])

query = field.query(points)
```

## Mathematical Definition

### Observations

Let the environment have spatial dimension $D$. Observation $i$ is

```math
\mathcal{O}_i = (\mathbf{o}_i,\mathbf{d}_i,r_i,\ell_i,t_i,w_i),
```

where $\mathbf{o}_i\in\mathbb{R}^D$ is the sensor origin, $\mathbf{d}_i\in\mathbb{R}^D$ is a unit direction, $r_i\geq0$ is the measured range, $\ell_i\in\{0,1\}$ indicates a return, $t_i$ is acquisition time, and $w_i\geq0$ is the observation evidence weight. The ray endpoint is

```math
\mathbf{e}_i = \mathbf{o}_i + r_i\mathbf{d}_i.
```

The field is evaluated at $M$ query positions $\mathbf{x}_j\in\mathbb{R}^D$.

### Gaussian kernel

For a positive-definite covariance $\boldsymbol{\Sigma}$, the peak-normalised Gaussian kernel is

```math
K(\mathbf{x},\boldsymbol{\mu};\boldsymbol{\Sigma})
=
\exp\!\left[
-\frac12
(\mathbf{x}-\boldsymbol{\mu})^{\mathsf T}
\boldsymbol{\Sigma}^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right].
```

### Occupied evidence

For occupied-evidence covariance $\boldsymbol{\Sigma}_s$,

```math
k_{s,i}(\mathbf{x})
=
\ell_i K(\mathbf{x},\mathbf{e}_i;\boldsymbol{\Sigma}_s).
```

Only returned observations contribute occupied evidence.

### Finite-ray free-space evidence

For free-space covariance $\boldsymbol{\Sigma}_f$, define

```math
L_i(\mathbf{x})
=
\sqrt{
\frac{
\mathbf{d}_i^{\mathsf T}\boldsymbol{\Sigma}_f^{-1}\mathbf{d}_i
}{2\pi}}
\int_0^{r_i}
K(\mathbf{x},\mathbf{o}_i+s\mathbf{d}_i;\boldsymbol{\Sigma}_f)
\,\mathrm ds.
```

Free-space evidence is smoothly suppressed around a returned endpoint:

```math
k_{f,i}(\mathbf{x})
=
L_i(\mathbf{x})
\left[
1-\ell_i K(\mathbf{x},\mathbf{e}_i;\boldsymbol{\Sigma}_s)
\right].
```

Thus a returned observation contributes zero free-space evidence at its occupied endpoint, while a non-returned observation contributes along its complete finite ray.

### Sampling-density normalisation

Repeated observations should not increase evidence strength purely because the same region was sampled more densely. Each observation is therefore represented by an acquisition state $\mathbf{s}_i$ and compared using a Gaussian state kernel $K_s$.

The local state-space density is

```math
\rho_i^{(s)}
=
\sum_{j\in\mathcal O}
K_s(\mathbf{s}_i,\mathbf{s}_j),
```

with redundancy weight

```math
w_i^{(r)}
=
\frac{1}{\rho_i^{(s)}}.
```

Only observation pairs within the configured state support are evaluated. The effective evidence weight is

```math
\widetilde{w}_i = w_i w_i^{(r)}.
```

### Beta evidence field

Let $\alpha_0(\mathbf{x})$ and $\beta_0(\mathbf{x})$ be the prior Beta parameters. The posterior field is

```math
\alpha(\mathbf{x})
=
\alpha_0(\mathbf{x})
+
\sum_{i=1}^{N}
\widetilde{w}_i k_{s,i}(\mathbf{x}),
```

```math
\beta(\mathbf{x})
=
\beta_0(\mathbf{x})
+
\sum_{i=1}^{N}
\widetilde{w}_i k_{f,i}(\mathbf{x}).
```

Occupancy is represented by

```math
p(\mathbf{x})\sim\mathrm{Beta}(\alpha(\mathbf{x}),\beta(\mathbf{x})).
```

Its mean, variance, and concentration are

```math
\mu(\mathbf{x})
=
\frac{\alpha}{\alpha+\beta},
```

```math
\mathrm{Var}[p(\mathbf{x})]
=
\frac{\alpha\beta}
{(\alpha+\beta)^2(\alpha+\beta+1)},
```

```math
\lambda(\mathbf{x})=\alpha+\beta.
```

The mean describes estimated occupancy, variance describes uncertainty in that probability, and concentration describes accumulated evidence strength.

### Support

A query has spatial support when it lies within the configured Mahalanobis neighbourhood of at least one returned surface:

```math
s(\mathbf{x})
=
\bigvee_{i=1}^N
\left\{
\ell_i=1
\;\land\;
(\mathbf{x}-\mathbf{e}_i)^{\mathsf T}
\boldsymbol{\Sigma}_s^{-1}
(\mathbf{x}-\mathbf{e}_i)
\leq\tau_s^2
\right\}.
```

Support is a geometric coverage indicator, not an additional probabilistic statistic.

## Computational Characteristics

Finite state support permits sparse redundancy evaluation. For $N$ retained observations and $E$ state-space neighbour pairs,

```math
T_{\mathrm{red}}(N)=O(N\log N+E).
```

With bounded local sampling density, $E=O(N)$ and measured redundancy/update behaviour is approximately linear over the characterised range. Increasing sampling density over a fixed extent drives $E$ toward $O(N^2)$, recovering the quadratic worst case.

Evidence evaluation currently forms dense interactions between $N$ retained observations and $M$ query points, giving

```math
T_{\mathrm{query}}(N,M)=O(NM).
```

The reproducible computational experiments are provided in `examples/04_computation.ipynb`.

## Testing

Install the locked Pixi environment and run the unit tests with:

```bash
pixi install
pixi run test
```

The test suite covers Gaussian kernels, sparse and dense distance consistency, occupancy observations, priors, query statistics, redundancy normalisation, and occupancy-field behaviour.

## Development

The project uses Pixi for reproducible development environments and `pytest` for testing. Example notebooks under `examples/` demonstrate the representation, evidence behaviour, sampling-density normalisation, and computational scaling.
