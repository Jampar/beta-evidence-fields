# Beta Evidence Fields 

[![Beta Field Unit Tests](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml/badge.svg)](https://github.com/Jampar/beta-evidence-fields/actions/workflows/tests.yml)

A Python library for continuous probabilistic occupancy fields from range-sensor evidence.

Beta Evidence Fields (BEFs) represents occupancy as a continuous Beta distribution constructed directly from range observations. Returned measurements contribute occupied evidence around their endpoints, while observation rays contribute free-space evidence along their length. Gaussian spatial kernels provide a continuous representation, and temporal Gaussian weighting allows evidence to decay as observations age.

## Mathematical Definition

Define an unnormalised Gaussian kernel with covariance $\Sigma$:

```math
k_{\Sigma}(\mathbf{x},\boldsymbol{\mu})
=
\exp\left(
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right).
```

Each range observation consists of an origin $\mathbf{o}_i$, unit direction $\mathbf{d}_i$, range $\rho_i$, acquisition time $t_i$, and evidence weight $w_i$. Its endpoint is

```math
\mathbf{e}_i = \mathbf{o}_i + \rho_i\mathbf{d}_i.
```

At position $\mathbf{x}$ and time $t$, the field is represented by

```math
p(\mathbf{x},t)
\sim
\mathrm{Beta}
\left(
\alpha(\mathbf{x},t),
\beta(\mathbf{x},t)
\right).
```

Occupied evidence is accumulated around returned endpoints:

```math
\alpha(\mathbf{x},t)
=
\alpha_0(\mathbf{x})
+
\sum_{i \in \mathcal{R}}
w_i
k_{\sigma_t^2}(t,t_i)
k_{\Sigma_s}(\mathbf{x},\mathbf{e}_i),
```

where $\mathcal{R}$ is the set of observations that produced a range return.

Free-space evidence is accumulated continuously along every observation ray:

```math
\beta(\mathbf{x},t)
=
\beta_0(\mathbf{x})
+
\sum_i
w_i
k_{\sigma_t^2}(t,t_i)
\sqrt{
\frac{
\mathbf{d}_i^\mathsf{T}\Sigma_f^{-1}\mathbf{d}_i
}{
2\pi
}
}
\int_0^{\rho_i}
k_{\Sigma_f}
\left(
\mathbf{x},
\mathbf{o}_i+s\mathbf{d}_i
\right)
\,ds.
```

Here, $\Sigma_s$ controls the spatial extent of occupied evidence, $\Sigma_f$ controls the spatial extent of free-space evidence, and $\sigma_t^2$ controls temporal decay.

The resulting occupancy estimate is

```math
\mathbb{E}[p(\mathbf{x},t)]
=
\frac{\alpha(\mathbf{x},t)}
{\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)},
```

with evidence concentration

```math
\lambda(\mathbf{x},t)
=
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t).
```

The concentration captures how much evidence supports the estimate, allowing locations with the same expected occupancy to retain different levels of evidential certainty.

## Features

- Dimension-agnostic spatial representation
- Continuous Beta occupancy field
- Gaussian occupied evidence around returned endpoints
- Analytically integrated free-space evidence along finite sensor rays
- Temporal Gaussian evidence decay
- Configurable spatial and temporal support
- Weighted observations
- Scalar and vectorised field queries
- Occupancy mean, variance, concentration, and support

## Installation

Python 3.10 or later is required.

Using Pixi:

```bash
pixi install
```

The package is configured as an editable dependency in the Pixi environment.

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
