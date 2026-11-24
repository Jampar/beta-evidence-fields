"""Plotting utilities for Beta Evidence Fields examples."""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm

from beta_evidence_field import OccupancyObservation
from beta_evidence_field import OccupancyQuery



def configure_plots() -> None:

    plt.rcParams.update({
        "font.family"      : "serif",
        "font.serif"       : ["STIXGeneral"],
        "mathtext.fontset" : "stix",
        "font.size"        : 11
    })



def plot_observation_2d(X           : np.ndarray,
                        Y           : np.ndarray,
                        surface     : np.ndarray,
                        free        : np.ndarray,
                        observation : OccupancyObservation) -> Axes:

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")

    if surface.shape != X.shape or free.shape != X.shape:
        raise ValueError("surface and free must match the grid shape")

    fig, axis = plt.subplots(figsize = (5, 5), constrained_layout = True)

    axis.contourf(X, Y, free,    levels = 10, cmap = "Blues", alpha = 0.75)
    axis.contourf(X, Y, surface, levels = 10, cmap = "Reds",  alpha = 0.75)

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

    _format_axis_2d(axis = axis,
                    X    = X,
                    Y    = Y)

    return axis



def plot_field_2d(X            : np.ndarray,
                  Y            : np.ndarray,
                  query        : OccupancyQuery,
                  observations : list[OccupancyObservation] | None = None) -> np.ndarray:

    shape = X.shape

    mean          = np.asarray(query.mean,          dtype = float).reshape(shape)
    variance      = np.asarray(query.variance,      dtype = float).reshape(shape)
    concentration = np.asarray(query.concentration, dtype = float).reshape(shape)
    support       = np.asarray(query.support,       dtype = bool).reshape(shape)

    fig, axes = plt.subplots(2, 2,
                             figsize            = (10, 10),
                             constrained_layout = True)

    mean_image = axes[0, 0].pcolormesh(X,
                                       Y,
                                       mean,
                                       cmap    = "RdBu_r",
                                       norm    = TwoSlopeNorm(vmin = 0.0, vcenter = 0.5, vmax = 1.0),
                                       shading = "auto")

    variance_image = axes[0, 1].pcolormesh(X,
                                           Y,
                                           variance,
                                           cmap    = "Purples",
                                           vmin    = 0.0,
                                           shading = "auto")

    concentration_image = axes[1, 0].pcolormesh(X,
                                                Y,
                                                concentration,
                                                cmap    = "Greens",
                                                vmin    = 0.0,
                                                shading = "auto")

    support_image = axes[1, 1].pcolormesh(X,
                                          Y,
                                          support,
                                          cmap    = "Greys",
                                          vmin    = 0.0,
                                          vmax    = 1.0,
                                          shading = "auto")

    plots = [
        (axes[0, 0], mean_image,          "Occupancy Mean",         None),
        (axes[0, 1], variance_image,      "Occupancy Variance",     None),
        (axes[1, 0], concentration_image, "Evidence Concentration", None),
        (axes[1, 1], support_image,       "Evidence Support",       [0, 1])
    ]

    for axis, image, title, ticks in plots:

        axis.set_title(title)

        _format_axis_2d(axis = axis,
                        X    = X,
                        Y    = Y)

        if observations is not None:
            _plot_observations_2d(axis         = axis,
                                  observations = observations)

        fig.colorbar(image,
                     ax    = axis,
                     ticks = ticks)

    return axes



def plot_field_comparison_2d(X            : np.ndarray,
                             Y            : np.ndarray,
                             queries      : list[OccupancyQuery],
                             titles       : list[str],
                             value        : str,
                             observations : list[list[OccupancyObservation]] | None = None) -> np.ndarray:

    if not queries:
        raise ValueError("queries must not be empty")

    if len(queries) != len(titles):
        raise ValueError("queries and titles must have the same length")

    if observations is not None and len(observations) != len(queries):
        raise ValueError("observations must match the number of queries")

    shape = X.shape

    fields = [
        np.asarray(getattr(query, value), dtype = float).reshape(shape)
        for query in queries
    ]

    if value == "mean":
        cmap  = "RdBu_r"
        norm  = TwoSlopeNorm(vmin = 0.0, vcenter = 0.5, vmax = 1.0)
        label = "Occupancy Mean"

    elif value == "variance":
        cmap  = "Purples"
        norm  = Normalize(vmin = 0.0,
                          vmax = max(float(np.max(field)) for field in fields))
        label = "Occupancy Variance"

    elif value == "concentration":
        cmap  = "Greens"
        norm  = Normalize(vmin = min(float(np.min(field)) for field in fields),
                          vmax = max(float(np.max(field)) for field in fields))
        label = "Evidence Concentration"

    elif value == "support":
        cmap  = "Greys"
        norm  = Normalize(vmin = 0.0, vmax = 1.0)
        label = "Evidence Support"

    else:
        raise ValueError(
            "value must be 'mean', 'variance', 'concentration', or 'support'")

    fig, axes = plt.subplots(1, len(fields),
                             figsize            = (5*len(fields), 5),
                             constrained_layout = True,
                             squeeze            = False)

    axes = axes.ravel()

    for index, (axis, field, title) in enumerate(zip(axes, fields, titles)):

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

    return axes



def plot_query_progression(counts  : np.ndarray,
                           queries : dict[str, list[OccupancyQuery]]) -> np.ndarray:

    counts = np.asarray(counts)

    styles = {
        "Occupied" : ("tab:red",  "-"),
        "Free"     : ("tab:blue", "--"),
        "Conflict" : ("tab:purple", "-.")
    }

    fig, axes = plt.subplots(1, 3,
                             figsize            = (12, 4),
                             constrained_layout = True)


    for label, query_list in queries.items():

        if len(counts) != len(query_list):
            raise ValueError("counts and queries must have the same length")

        mean = np.asarray([
            float(query.mean)
            for query in query_list
        ])

        variance = np.asarray([
            float(query.variance)
            for query in query_list
        ])

        concentration = np.asarray([
            float(query.concentration)
            for query in query_list
        ])

        color, linestyle = styles[label]

        axes[0].plot(counts,
                     mean,
                     color     = color,
                     linestyle = linestyle,
                     marker    = "o",
                     linewidth = 2,
                     label     = label,
                     alpha     = 0.5)

        axes[0].axhline(
            0.5,
            color     = "0.5",
            linewidth = 1.0,
            linestyle = ":",
            zorder    = 0
        )

        axes[1].plot(counts,
                     variance,
                     color     = color,
                     linestyle = linestyle,
                     marker    = "o",
                     linewidth = 2,
                     label     = label,
                     alpha     = 0.5)

        axes[2].plot(counts,
                     concentration,
                     color     = color,
                     linestyle = linestyle,
                     marker    = "o",
                     linewidth = 2,
                     label     = label,
                     alpha     = 0.5)


    labels = [
        ("Occupancy Mean",         r"$\mathbb{E}[p]$"),
        ("Occupancy Variance",     r"$\mathrm{Var}[p]$"),
        ("Evidence Concentration", r"$\lambda$")
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

    mean = np.asarray([float(query.mean) for query in queries])

    standard_deviation = np.sqrt([float(query.variance)
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

    axes[1].set_ylabel(r"Evidence Concentration $\lambda$")
    axes[1].set_title("Evidence Strength")

    for axis in axes:

        axis.set_xticks(positions,
                        labels,
                        rotation = 15)

        axis.grid(axis  = "y",
                  color = "0.85",
                  alpha = 0.6)

    return axes



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
