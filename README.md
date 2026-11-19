# Beta-Evidence-Fields

A dimension-agnostic Python library for continuous probabilistic occupancy fields from range-sensor evidence.

Beta-Evidence-Fields represents occupancy as a continuous spatial Beta distribution constructed directly from range observations. Returned measurements contribute occupied evidence around their endpoints, while measurement rays contribute free-space evidence along their length. Gaussian spatial kernels provide a continuous representation, with temporal Gaussian weighting allowing evidence to decay as observations age.

## Mathematical Definition

At a spatial location $\mathbf{x}$ and time $t$, occupancy is represented by a Beta distribution:

$$
p(\mathbf{x},t)
\sim
\mathrm{Beta}
\left(
\alpha(\mathbf{x},t),
\beta(\mathbf{x},t)
\right).
$$
