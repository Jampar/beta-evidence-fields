"""Plotting utilities for Beta Evidence Fields examples."""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from beta_evidence_field import OccupancyObservation
from beta_evidence_field import OccupancyQuery


GRADIENT_STEP  = 5
GRADIENT_SCALE = 8.0
CONTOUR_LEVELS = 31

def configure_plots() -> None:

    plt.rcParams.update({"font.family"      : "serif",
                         "font.serif"       : ["STIXGeneral"],
                         "mathtext.fontset" : "stix",
                         "font.size"        : 11 })               


def plot_evidence_2d(X     : np.ndarray,
                     Y     : np.ndarray,
                     field : OccupancyField,
                     axes  : np.ndarray | None = None) -> np.ndarray:

    if axes is None:
        _, axes = plt.subplots(1, 3,
                               figsize            = (15, 5),
                               constrained_layout = True)

    fig = axes[0].figure

    _validate_grid_2d(X, Y)

    points = np.column_stack((X.ravel(),
                              Y.ravel()))

    query = field.query(points)

    shape = X.shape

    alpha = np.asarray(query.alpha, dtype = float).reshape(shape)
    beta  = np.asarray(query.beta,  dtype = float).reshape(shape)

    alpha_gradient = np.asarray(query.alpha_gradient,
                                dtype = float).reshape(*shape, 2)

    beta_gradient = np.asarray(query.beta_gradient,
                               dtype = float).reshape(*shape, 2)

    alpha_contour = axes[0].contourf(X,
                                     Y,
                                     alpha,
                                     levels = CONTOUR_LEVELS,
                                     cmap   = "Reds",
                                     alpha  = 0.75)

    beta_contour = axes[0].contourf(X,
                                    Y,
                                    beta,
                                    levels = CONTOUR_LEVELS,
                                    cmap   = "Blues",
                                    alpha  = 0.75)

    axes[1].contourf(X,
                     Y,
                     alpha,
                     levels = CONTOUR_LEVELS,
                     cmap   = "Reds",
                     alpha  = 0.75)

    axes[1].quiver(X[::GRADIENT_STEP, ::GRADIENT_STEP],
                   Y[::GRADIENT_STEP, ::GRADIENT_STEP],
                   alpha_gradient[::GRADIENT_STEP, ::GRADIENT_STEP, 0],
                   alpha_gradient[::GRADIENT_STEP, ::GRADIENT_STEP, 1],
                   color       = "tab:red",
                   angles      = "xy",
                   scale_units = "xy",
                   scale       = GRADIENT_SCALE)

    axes[2].contourf(X,
                     Y,
                     beta,
                     levels = CONTOUR_LEVELS,
                     cmap   = "Blues",
                     alpha  = 0.75)

    axes[2].quiver(X[::GRADIENT_STEP, ::GRADIENT_STEP],
                   Y[::GRADIENT_STEP, ::GRADIENT_STEP],
                   beta_gradient[::GRADIENT_STEP, ::GRADIENT_STEP, 0],
                   beta_gradient[::GRADIENT_STEP, ::GRADIENT_STEP, 1],
                   color       = "tab:blue",
                   angles      = "xy",
                   scale_units = "xy",
                   scale       = GRADIENT_SCALE)

    axes[0].set_title("Evidence")
    axes[1].set_title(r"Occupied Evidence Gradient $\nabla\alpha$")
    axes[2].set_title(r"Free Evidence Gradient $\nabla\beta$")

    for axis in axes:
        _format_axis_2d(axis = axis,
                        X    = X,
                        Y    = Y)

    _add_horizontal_colorbar(fig   = fig,
                             axis  = axes[1],
                             image = alpha_contour,
                             label = r"$\alpha(\mathbf{x})$")

    _add_horizontal_colorbar(fig   = fig,
                             axis  = axes[2],
                             image = beta_contour,
                             label = r"$\beta(\mathbf{x})$")


    return axes


def plot_query_2d(X     : np.ndarray,
                  Y     : np.ndarray,
                  field : OccupancyField,
                  axes  : np.ndarray | None = None) -> np.ndarray:

    if axes is None:
        _, axes = plt.subplots(1, 3,
                               figsize            = (15, 5),
                               constrained_layout = True)

    fig = axes[0].figure

    _validate_grid_2d(X, Y)

    points = np.column_stack((X.ravel(),
                              Y.ravel()))

    query = field.query(points)

    shape = X.shape

    mean          = np.asarray(query.mean,          dtype = float).reshape(shape)
    variance      = np.asarray(query.variance,      dtype = float).reshape(shape)
    concentration = np.asarray(query.concentration, dtype = float).reshape(shape)
    gradient      = np.asarray(query.gradient,      dtype = float).reshape(*shape, 2) * 5

    plots = [
        (axes[0],
         mean,
         "RdBu_r",
         TwoSlopeNorm(vmin = 0.0, vcenter = 0.5, vmax = 1.0),
         "Occupancy Mean and Gradient",
         r"$\mu(\mathbf{x})$"),

        (axes[1],
         variance,
         "Purples",
         Normalize(vmin = 0.0,
                   vmax = float(np.max(variance))),
         "Occupancy Variance",
         r"$\mathrm{Var}[p(\mathbf{x})]$"),

        (axes[2],
         concentration,
         "Greens",
         Normalize(vmin = float(np.min(concentration)),
                   vmax = float(np.max(concentration))),
         "Evidence Concentration",
         r"$\kappa(\mathbf{x})$")
    ]

    origins = np.asarray([observation.origin
                          for observation in field.observations])

    returned = [observation
                for observation in field.observations
                if observation.returned]

    for index, (axis, values, cmap, norm, title, label) in enumerate(plots):

        levels = np.linspace(norm.vmin,
                             norm.vmax,
                             CONTOUR_LEVELS)

        image = axis.contourf(X,
                              Y,
                              values,
                              levels = levels,
                              cmap   = cmap,
                              norm   = norm)

        if index == 0:

            axis.quiver(X[::GRADIENT_STEP, ::GRADIENT_STEP],
                        Y[::GRADIENT_STEP, ::GRADIENT_STEP],
                        gradient[::GRADIENT_STEP, ::GRADIENT_STEP, 0],
                        gradient[::GRADIENT_STEP, ::GRADIENT_STEP, 1],
                        color       = "black",
                        angles      = "xy",
                        scale_units = "xy",
                        scale       = GRADIENT_SCALE,
                        alpha       = 0.6)

        if returned:

            endpoints = np.asarray([observation.endpoint
                                    for observation in returned])

            axis.scatter(endpoints[:, 0],
                         endpoints[:, 1],
                         color     = "black",
                         marker    = "o",
                         s         = 10,
                         linewidth = 1.0,
                         alpha     = 0.25,
                         zorder    = 10)

        axis.set_title(title)

        _format_axis_2d(axis = axis,
                        X    = X,
                        Y    = Y)

        _add_horizontal_colorbar(fig   = fig,
                                 axis  = axis,
                                 image = image,
                                 label = label)


def plot_field_comparison_2d(X            : np.ndarray,
                             Y            : np.ndarray,
                             queries      : list[OccupancyQuery],
                             titles       : list[str],
                             value        : str,
                             observations : list[list[OccupancyObservation]] | None = None) -> np.ndarray:

    _validate_grid_2d(X, Y)

    if not queries:
        raise ValueError("queries must not be empty")

    if len(queries) != len(titles):
        raise ValueError("queries and titles must have the same length")

    if observations is not None and len(observations) != len(queries):
        raise ValueError("observations must match the number of queries")

    fields = [np.asarray(getattr(query, value),
                         dtype = float).reshape(X.shape)
              for query in queries]

    cmap, norm, label = _field_style(value  = value,
                                     fields = fields)

    fig, axes = plt.subplots(1, len(fields),
                             figsize            = (5*len(fields), 5),
                             constrained_layout = True,
                             squeeze            = False)

    axes = axes.ravel()

    for index, (axis, field, title) in enumerate(zip(axes,
                                                     fields,
                                                     titles)):

        image = axis.pcolormesh(X,
                                Y,
                                field,
                                cmap    = cmap,
                                norm    = norm,
                                shading = "auto")

        axis.set_title(title)

        _format_axis_2d(axis = axis,
                        X    = X,
                        Y    = Y)

        if observations is not None:
            _plot_observations_2d(axis         = axis,
                                  observations = observations[index])

    fig.colorbar(image,
                 ax          = axes.tolist(),
                 orientation = "horizontal",
                 fraction    = 0.06,
                 pad         = 0.08,
                 label       = label)



def plot_query_progression(counts  : np.ndarray,
                           queries : dict[str, list[OccupancyQuery]]) -> np.ndarray:

    counts = np.asarray(counts)

    styles = {
        "Occupied" : ("tab:red",    "-"),
        "Free"     : ("tab:blue",   "--"),
        "Conflict" : ("tab:purple", "-.")
    }

    fig, axes = plt.subplots(1, 3,
                             figsize            = (12, 4),
                             constrained_layout = True)

    for label, query_list in queries.items():

        if len(counts) != len(query_list):
            raise ValueError("counts and queries must have the same length")

        mean          = np.asarray([float(query.mean)          for query in query_list])
        variance      = np.asarray([float(query.variance)      for query in query_list])
        concentration = np.asarray([float(query.concentration) for query in query_list])

        color, linestyle = styles[label]

        for axis, values in zip(axes,
                                [mean, variance, concentration]):

            axis.plot(counts,
                      values,
                      color     = color,
                      linestyle = linestyle,
                      marker    = "o",
                      linewidth = 2,
                      label     = label,
                      alpha     = 0.5)

    axes[0].axhline(0.5,
                    color     = "0.5",
                    linewidth = 1.0,
                    linestyle = ":",
                    zorder    = 0)

    labels = [
        ("Occupancy Mean",         r"$\mathbb{E}[p]$"),
        ("Occupancy Variance",     r"$\mathrm{Var}[p]$"),
        ("Evidence Concentration", r"$\kappa$")
    ]

    for axis, (title, ylabel) in zip(axes, labels):

        axis.set_title(title)
        axis.set_xlabel("Number of Observations")
        axis.set_ylabel(ylabel)

        axis.grid(color = "0.85",
                  alpha = 0.6)

        axis.legend()

    axes[0].set_ylim(0.0, 1.0)

    return axes


def plot_beta_states(queries : list[OccupancyQuery],
                     labels  : list[str]) -> np.ndarray:

    if len(queries) != len(labels):
        raise ValueError("queries and labels must have the same length")

    mean = np.asarray([float(query.mean)
                       for query in queries])

    standard_deviation = np.sqrt([
        float(query.variance)
        for query in queries
    ])

    concentration = np.asarray([
        float(query.concentration)
        for query in queries
    ])

    positions = np.arange(len(queries))

    fig, axes = plt.subplots(1, 2,
                             figsize            = (10, 4),
                             constrained_layout = True)

    axes[0].errorbar(positions,
                     mean,
                     yerr      = standard_deviation,
                     color     = "black",
                     marker    = "o",
                     capsize   = 5,
                     linestyle = "none")

    axes[0].axhline(0.5,
                    color     = "0.7",
                    linewidth = 1)

    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Occupancy Probability")
    axes[0].set_title("Equal Mean, Different Uncertainty")

    axes[1].bar(positions,
                concentration,
                color = "tab:green")

    axes[1].set_ylabel(r"Evidence Concentration $\kappa$")
    axes[1].set_title("Evidence Strength")

    for axis in axes:

        axis.set_xticks(positions,
                        labels,
                        rotation = 15)

        axis.grid(axis  = "y",
                  color = "0.85",
                  alpha = 0.6)

    return axes


def _validate_grid_2d(X : np.ndarray,
                      Y : np.ndarray) -> None:

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")

    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")


def _field_style(value  : str,
                 fields : list[np.ndarray]) -> tuple[str, Normalize, str]:

    if value == "mean":
        return ("RdBu_r",
                TwoSlopeNorm(vmin = 0.0,
                             vcenter = 0.5,
                             vmax = 1.0),
                "Occupancy Mean")

    if value == "variance":
        return ("Purples",
                Normalize(vmin = 0.0,
                          vmax = max(float(np.max(field))
                                     for field in fields)),
                "Occupancy Variance")

    if value == "concentration":
        return ("Greens",
                Normalize(vmin = min(float(np.min(field))
                                     for field in fields),
                          vmax = max(float(np.max(field))
                                     for field in fields)),
                "Evidence Concentration")

    if value == "support":
        return ("Greys",
                Normalize(vmin = 0.0,
                          vmax = 1.0),
                "Evidence Support")

    raise ValueError(
        "value must be 'mean', 'variance', 'concentration', or 'support'")


def _plot_observation_geometry_2d(axis        : Axes,
                                  observation : OccupancyObservation) -> None:

    origin    = observation.origin[:2]
    endpoint  = observation.endpoint[:2]
    direction = observation.direction[:2]

    axis.plot([origin[0], endpoint[0]],
              [origin[1], endpoint[1]],
              color     = "black",
              linewidth = 1.5)

    axis.scatter(origin[0],
                 origin[1],
                 color  = "black",
                 marker = "o",
                 s      = 50,
                 zorder = 10)

    axis.scatter(endpoint[0],
                 endpoint[1],
                 color     = "black",
                 marker    = "x",
                 s         = 70,
                 linewidth = 2,
                 zorder    = 10)

    axis.annotate(r"$\mathbf{o}$",
                  origin,
                  xytext     = (-5, 10),
                  textcoords = "offset points")

    axis.annotate(r"$\mathbf{p}$",
                  endpoint,
                  xytext     = (5, 10),
                  textcoords = "offset points")

    direction_offset = origin + np.array([0.0, 0.5])

    axis.annotate(r"$\mathbf{d}$",
                  xy         = direction_offset + 0.9*direction,
                  xytext     = direction_offset + 0.3*direction,
                  arrowprops = {"arrowstyle": "->"})

    midpoint = origin + 0.5*observation.range*direction
    normal   = np.array([direction[1], -direction[0]])

    if not np.isclose(np.linalg.norm(normal), 0.0):
        normal = normal/np.linalg.norm(normal)

    label = midpoint + 0.12*normal

    axis.text(label[0],
              label[1],
              r"$r$",
              ha = "center",
              va = "center")


def _add_horizontal_colorbar(fig,
                             axis  : Axes,
                             image,
                             label : str,
                             offset : float = -0.2) -> None:

    colorbar_axis = inset_axes(axis,
                               width          = "100%",
                               height         = "4%",
                               loc            = "lower left",
                               bbox_to_anchor = (0.0, offset, 1.0, 1.0),
                               bbox_transform = axis.transAxes,
                               borderpad      = 0)

    colorbar = fig.colorbar(image,
                            cax         = colorbar_axis,
                            orientation = "horizontal",
                            label       = label)

    colorbar.locator = plt.MaxNLocator(5)
    colorbar.update_ticks()


def _add_vertical_colorbar(fig,
                           axis     : Axes,
                           image,
                           label    : str,
                           location : str = "right") -> None:

    fig.colorbar(image,
                 ax       = axis,
                 label    = label,
                 location = location)


def _format_axis_2d(axis : Axes,
                    X    : np.ndarray,
                    Y    : np.ndarray) -> None:

    axis.set_xlim(float(np.min(X)), float(np.max(X)))
    axis.set_ylim(float(np.min(Y)), float(np.max(Y)))

    axis.set_aspect("equal", adjustable = "box")

    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")

    axis.grid(color     = "0.85",
              linewidth = 0.5,
              alpha     = 0.5)


def _plot_observations_2d(axis         : Axes,
                          observations : list[OccupancyObservation],
                          alpha        : float = 0.35) -> None:

    for observation in observations:

        axis.plot([observation.origin[0], observation.endpoint[0]],
                  [observation.origin[1], observation.endpoint[1]],
                  color     = "black",
                  linewidth = 0.8,
                  alpha     = alpha,
                  zorder    = 5)

        axis.scatter(observation.origin[0],
                     observation.origin[1],
                     color  = "black",
                     marker = "o",
                     s      = 10,
                     alpha  = alpha,
                     zorder = 6)

        if observation.returned:
            axis.scatter(observation.endpoint[0],
                         observation.endpoint[1],
                         color     = "black",
                         marker    = "x",
                         s         = 20,
                         linewidth = 1.0,
                         alpha     = alpha,
                         zorder    = 6)