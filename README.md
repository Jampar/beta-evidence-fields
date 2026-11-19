# Beta-Evidence-Fields

A dimension-agnostic Python library for continuous probabilistic occupancy fields from range-sensor evidence.

Beta-Evidence-Fields represents occupancy as a continuous spatial Beta distribution constructed directly from range observations. Returned measurements contribute occupied evidence around their endpoints, while measurement rays contribute free-space evidence along their length. Gaussian spatial kernels provide a continuous representation, with temporal Gaussian weighting allowing evidence to decay as observations age.

## Mathematical Definition

At a spatial location \(\mathbf{x}\) and time \(t\), occupancy probability is represented by a Beta distribution

\[
p(\mathbf{x},t)
\sim
\operatorname{Beta}
\left(
\alpha(\mathbf{x},t),
\beta(\mathbf{x},t)
\right),
\]

where \(\alpha(\mathbf{x},t)\) represents occupied evidence and \(\beta(\mathbf{x},t)\) represents free-space evidence.

For a set of range observations \(i=1,\ldots,N\), the field is

\[
\alpha(\mathbf{x},t)
=
\alpha_0(\mathbf{x})
+
\sum_{i=1}^{N}
r_i w_i
k_t(t,t_i)
k_s(\mathbf{x},\mathbf{e}_i),
\]

\[
\beta(\mathbf{x},t)
=
\beta_0(\mathbf{x})
+
\sum_{i=1}^{N}
w_i
k_t(t,t_i)
\sqrt{\frac{\mathbf{d}_i^\mathsf{T}
\Sigma_f^{-1}\mathbf{d}_i}{2\pi}}
\int_0^{\rho_i}
k_f
\left(
\mathbf{x},
\mathbf{o}_i+s\mathbf{d}_i
\right)
\,ds.
\]

Each observation is defined by

- \(\mathbf{o}_i\): sensor origin,
- \(\mathbf{d}_i\): unit ray direction,
- \(\rho_i\): measured ray range,
- \(\mathbf{e}_i=\mathbf{o}_i+\rho_i\mathbf{d}_i\): ray endpoint,
- \(r_i\in\{0,1\}\): whether the ray produced a return,
- \(t_i\): observation time,
- \(w_i\): evidence weight.

The spatial Gaussian kernel is

\[
k(\mathbf{x},\boldsymbol{\mu})
=
\exp
\left(
-\frac{1}{2}
(\mathbf{x}-\boldsymbol{\mu})^\mathsf{T}
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu})
\right).
\]

Separate covariance matrices \(\Sigma_s\) and \(\Sigma_f\) define the occupied surface and free-space kernels respectively.

Temporal evidence is weighted using the one-dimensional Gaussian kernel

\[
k_t(t,t_i)
=
\exp
\left(
-\frac{(t-t_i)^2}{2\sigma_t^2}
\right),
\]

where \(\sigma_t^2\) is the configured temporal variance.

Only returned observations contribute to \(\alpha\), whereas every retained observation contributes free-space evidence to \(\beta\) along its ray.

### Occupancy Statistics

The expected occupancy probability is

\[
\mathbb{E}[p(\mathbf{x},t)]
=
\frac{\alpha(\mathbf{x},t)}
{\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)}.
\]

The Beta variance is

\[
\operatorname{Var}[p(\mathbf{x},t)]
=
\frac{
\alpha(\mathbf{x},t)\beta(\mathbf{x},t)
}{
\left(\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)\right)^2
\left(\alpha(\mathbf{x},t)+\beta(\mathbf{x},t)+1\right)
}.
\]

The total evidence concentration is

\[
\lambda(\mathbf{x},t)
=
\alpha(\mathbf{x},t)+\beta(\mathbf{x},t).
\]

Consequently, locations with the same mean occupancy may retain different levels of evidential certainty.

### Support

A query is considered spatially supported when it lies within `support_sigma` Mahalanobis standard deviations of at least one returned observation endpoint:

\[
(\mathbf{x}-\mathbf{e}_i)^\mathsf{T}
\Sigma_s^{-1}
(\mathbf{x}-\mathbf{e}_i)
\leq
\sigma_{\mathrm{support}}^2.
\]

Observations are retained temporally while

\[
\frac{(t-t_i)^2}{\sigma_t^2}
\leq
\sigma_{\mathrm{temporal}}^2.
\]

The support flag is therefore distinct from occupancy probability: it indicates whether a location is spatially supported by returned range evidence.

## Features

- Dimension-agnostic spatial representation
- Continuous Beta occupancy field
- Gaussian occupied evidence around range returns
- Analytically integrated free-space evidence along finite sensor rays
- Temporal Gaussian evidence decay
- Configurable spatial and temporal support
- Weighted observations
- Scalar and vectorised field queries
- Occupancy mean, variance, concentration, and support

## Installation

Python 3.10 or later is required.

Using Pixi:

`pixi install`

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

The unit tests run in the project's Pixi environment:

`pixi run test`

The test suite covers Gaussian kernels, occupancy observations, priors, query statistics, and occupancy-field behaviour.