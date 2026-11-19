# Beta-Evidence-Fields

A dimension-agnostic Python library for continuous probabilistic occupancy fields from range-sensor evidence.

Beta-Evidence-Fields represents occupancy as a continuous spatial Beta distribution constructed directly from range observations. Returned measurements contribute occupied evidence around their endpoints, while measurement rays contribute free-space evidence along their length. Gaussian spatial kernels provide a continuous representation, with temporal Gaussian weighting allowing evidence to decay as observations age.

## Mathematical Definition

At a spatial location $\mathbf{x}$ and time $t$, occupancy is represented by a Beta distribution:

$$
p(\mathbf{x},t)
\sim
\operatorname{Beta}
\left(
\alpha(\mathbf{x},t),
\beta(\mathbf{x},t)
\right).
$$

The field parameters are constructed from the prior and accumulated range-sensor evidence:

$$
\alpha(\mathbf{x},t)
=
\alpha_0(\mathbf{x})
+
\sum_{i=1}^{N}
r_i w_i
k_t(t,t_i)
k_s(\mathbf{x},\mathbf{e}_i),
$$

$$
\beta(\mathbf{x},t)
=
\beta_0(\mathbf{x})
+
\sum_{i=1}^{N}
w_i
k_t(t,t_i)
\sqrt{
\frac{
\mathbf{d}_i^\mathsf{T}
\Sigma_f^{-1}
\mathbf{d}_i
}{
2\pi
}
}
\int_0^{\rho_i}
k_f
\left(
\mathbf{x},
\mathbf{o}_i+s\mathbf{d}_i
\right)
\,ds.
$$

Each observation is defined by:

- $\mathbf{o}_i$: sensor origin
- $\mathbf{d}_i$: unit ray direction
- $\rho_i$: ray range
- $\mathbf{e}_i=\mathbf{o}_i+\rho_i\mathbf{d}_i$: ray endpoint
- $r_i\in\{0,1\}$: whether the observation produced a return
- $t_i$: observation time
- $w_i$: evidence weight

Only returned observations contribute occupied evidence to $\alpha$. Every retained observation contributes free-space evidence to $\beta$ along its finite ray.

### Spatial Evidence

The unnormalised spatial Gaussian kernel is

$$
k(\mathbf{x},\boldsymbol{\mu})
=
\exp
\left(
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right).
$$

Separate covariance matrices $\Sigma_s$ and $\Sigma_f$ define the occupied-surface and free-space kernels respectively. Free-space evidence is evaluated using the analytic integral of the Gaussian kernel along each finite observation ray.

### Temporal Evidence

Evidence is temporally weighted using

$$
k_t(t,t_i)
=
\exp
\left(
-\frac{(t-t_i)^2}{2\sigma_t^2}
\right),
$$

where $\sigma_t^2$ is the configured temporal variance.

### Occupancy Statistics

The expected occupancy probability is

$$
\mathbb{E}[p(\mathbf{x},t)]
=
\frac{
\alpha(\mathbf{x},t)
}{
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)
}.
$$

The occupancy variance is

$$
\operatorname{Var}[p(\mathbf{x},t)]
=
\frac{
\alpha(\mathbf{x},t)\beta(\mathbf{x},t)
}{
\left(\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)\right)^2
\left(\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)+1\right)
}.
$$

The evidence concentration is

$$
\lambda(\mathbf{x},t)
=
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t).
$$

This separates the estimated occupancy probability from the amount of evidence supporting that estimate.

### Support

A location is spatially supported when it lies within the configured Mahalanobis distance of at least one returned observation endpoint:

$$
(\mathbf{x}-\mathbf{e}_i)^\mathsf{T}
\Sigma_s^{-1}
(\mathbf{x}-\mathbf{e}_i)
\leq
\sigma_{\mathrm{support}}^2.
$$

Observations are retained temporally while

$$
\frac{(t-t_i)^2}{\sigma_t^2}
\leq
\sigma_{\mathrm{temporal}}^2.
$$

Support is distinct from occupancy probability and evidence concentration. It indicates whether a location is spatially supported by returned range evidence.

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

from beta_evidence_field import (
    OccupancyField,
    OccupancyObservation,
    OccupancyPrior
)


field = OccupancyField(
    surface_covariance = 0.1**2*np.eye(2),
    free_covariance    = 0.2**2*np.eye(2),
    temporal_variance  = 10.0**2,
    prior              = OccupancyPrior()
)


observation = OccupancyObservation(
    origin    = np.array([0.0, 0.0]),
    direction = np.array([1.0, 0.0]),
    range     = 2.0,
    returned  = True,
    time      = 0.0
)

field.add([observation])


query = field.query(
    points = np.array([2.0, 0.0]),
    time   = 0.0
)

print(query.alpha)
print(query.beta)
print(query.mean)
print(query.variance)
print(query.concentration)
print(query.support)
```

Multiple points can be evaluated simultaneously:

```python
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0]
])

query = field.query(points)
```

## Testing

Install the locked Pixi environment and run the unit tests with:

```bash
pixi install
pixi run test
```

The test suite covers Gaussian kernels, occupancy observations, priors, query statistics, and occupancy-field behaviour.
