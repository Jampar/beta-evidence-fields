import matplotlib.pyplot as plt
import numpy as np

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
                        observation : OccupancyObservation) -> None:

    fig, axis = plt.subplots(figsize = (6, 6), constrained_layout = True)

    axis.contourf(X, Y, free,    levels = 10, cmap   = "Blues", alpha  = 0.75)
    axis.contourf(X, Y, surface, levels = 10, cmap   = "Reds",  alpha  = 0.75)

    axis.plot([observation.origin[0], observation.endpoint[0]], 
              [observation.origin[1], observation.endpoint[1]],
              color     = "black",
              linewidth = 1.5)

    axis.scatter(observation.origin[0],
                 observation.origin[1],
                 color  = "black",
                 marker = "o",
                 s      = 50,
                 zorder = 10)

    axis.scatter(observation.endpoint[0],
                 observation.endpoint[1],
                 color     = "black",
                 marker    = "x",
                 s         = 70,
                 linewidth = 2,
                 zorder    = 10)
    
    axis.annotate(r"$\mathbf{o}$",
                  observation.origin[:2],
                  xytext     = (-5, 10),
                  textcoords = "offset points")

    axis.annotate(r"$\mathbf{p}$",
                  observation.endpoint[:2],
                  xytext     = (5, 10),
                  textcoords = "offset points")

    axis.annotate(r"$\mathbf{d}$",
                  xy     = (observation.origin[:2] + [0.0, 0.5]) + 0.9*observation.direction[:2],
                  xytext = (observation.origin[:2] + [0.0, 0.5]) + 0.3*observation.direction[:2],
                  arrowprops = {"arrowstyle": "->"})

    midpoint = (observation.origin[:2] + 0.5*observation.range*observation.direction[:2])
    normal   = np.array([observation.direction[1], -observation.direction[0]])
    label    = midpoint + 0.12*normal

    axis.text(label[0],
              label[1],
              r"$r$",
              ha = "center",
              va = "center")

    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")

    return axis


def plot_field_2d(X     : np.ndarray,
                  Y     : np.ndarray,
                  query : OccupancyQuery) -> np.ndarray:

    shape = X.shape

    mean          = np.asarray(query.mean).reshape(shape)
    variance      = np.asarray(query.variance).reshape(shape)
    concentration = np.asarray(query.concentration).reshape(shape)
    support       = np.asarray(query.support).reshape(shape)


    fig, axes = plt.subplots(2, 2,
                             figsize            = (10, 10),
                             constrained_layout = True)


    fields = [
        (mean,          "Occupancy Mean",         "RdBu_r",  np.linspace(0.0, 1.0, 21)),
        (variance,      "Occupancy Variance",     "Purples", 20),
        (concentration, "Evidence Concentration", "Greens",  20)
    ]


    for axis, (field, title, cmap, levels) in zip(axes.flat[:3], fields):

        image = axis.contourf(X,
                              Y,
                              field,
                              levels = levels,
                              cmap   = cmap)
        
        axis.set_box_aspect(1)
        
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

        fig.colorbar(image,
                     ax = axis)


    axis = axes[1, 1]

    image = axis.pcolormesh(X,
                            Y,
                            support,
                            cmap    = "Greys",
                            vmin    = 0.0,
                            vmax    = 1.0,
                            shading = "auto")
    
    axis.set_title("Evidence Support")

    axis.set_box_aspect(1)

    axis.set_xlim(X.min(), X.max())
    axis.set_ylim(Y.min(), Y.max())
    
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    
    fig.colorbar(image,
                 ax    = axis,
                 ticks = [0, 1])


    return axes