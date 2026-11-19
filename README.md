# Beta Evidence Fields 

[![Beta Field Unit Tests](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml/badge.svg)](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml)

A Python library for continuous probabilistic occupancy fields from range-sensor evidence.

Beta Evidence Fields (BEFs) represents occupancy as a continuous Beta distribution constructed directly from range observations. Returned measurements contribute occupied evidence around their endpoints, while observation rays contribute free-space evidence along their length. Gaussian spatial kernels provide a continuous representation, and temporal Gaussian weighting allows evidence to decay as observations age.

## Mathematical Definition

Define a peak-normalised Gaussian kernel with covariance $\Sigma$:

```math
K_{\Sigma}(\mathbf{x},\boldsymbol{\mu})
=
\exp\left(
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right).
```

Each range observation has endpoint

```math
\mathbf{e}_i = \mathbf{o}_i + \rho_i\mathbf{d}_i.
```

Where:

- $\mathbf{o}_i$: observation origin
- $\mathbf{d}_i$: unit direction
- $\rho_i$: range
- $t_i$: acquisition time
- $w_i$: evidence weight

At query points $\mathbf{x}_1,\ldots,\mathbf{x}_M$ and time $t$, define the temporal weight vector

```math
\mathbf{W}(t)
=
\begin{bmatrix}
w_1 K_{\sigma_t^2}(t,t_1)\\
\vdots\\
w_N K_{\sigma_t^2}(t,t_N)
\end{bmatrix}.
```

The occupied-evidence matrix is

```math
[\mathbf{K}_s]_{ij}
=
r_i K_{\Sigma_s}(\mathbf{x}_j,\mathbf{e}_i),
```

where $r_i\in\{0,1\}$ indicates whether observation $i$ produced a range return.

The free-space evidence matrix is

```math
[\mathbf{K}_f]_{ij}
=
\sqrt{
\frac{
\mathbf{d}_i^\mathsf{T}\Sigma_f^{-1}\mathbf{d}_i
}{
2\pi
}
}
\int_0^{\rho_i}
K_{\Sigma_f}
\left(
\mathbf{x}_j,
\mathbf{o}_i+s\mathbf{d}_i
\right)
\,ds.
```

The Beta field parameters at all query points are then

```math
\begin{aligned}
\boldsymbol{\alpha}(t)
&=
\boldsymbol{\alpha}_0
+
\mathbf{W}(t)^\mathsf{T}\mathbf{K}_s,
\\[4pt]
\boldsymbol{\beta}(t)
&=
\boldsymbol{\beta}_0
+
\mathbf{W}(t)^\mathsf{T}\mathbf{K}_f.
\end{aligned}
```

Here, $\Sigma_s$ controls the spatial extent of occupied evidence, $\Sigma_f$ controls the spatial extent of free-space evidence, and $\sigma_t^2$ controls temporal decay. The matrices $\mathbf{K}_s,\mathbf{K}_f\in\mathbb{R}^{N\times M}$ contain the spatial evidence from each observation at each query point.

At each query point $\mathbf{x}$, the occupancy probability is represented by

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
\mathbb{E}[p(\mathbf{x},t)]
=
\frac{\alpha(\mathbf{x},t)}
{\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)}.
```

The occupancy variance is

```math
\mathrm{Var}[p(\mathbf{x},t)]
=
\frac{
\alpha(\mathbf{x},t)\beta(\mathbf{x},t)
}{
\left(\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)\right)^2
\left(\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)+1\right)
}.
```

The evidence concentration is

```math
\lambda(\mathbf{x},t)
=
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t).
```

Spatial support is defined by proximity to at least one returned observation:

```math
s(\mathbf{x},t)
=
\bigvee_{i:r_i=1}
\left[
(\mathbf{x}-\mathbf{e}_i)^\mathsf{T}
\Sigma_s^{-1}
(\mathbf{x}-\mathbf{e}_i)
\leq
\sigma_s^2
\right].
```

The mean describes estimated occupancy, the variance describes uncertainty in that probability, the concentration describes the amount of accumulated evidence, and the support flag indicates whether the location is supported by returned observations.

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
