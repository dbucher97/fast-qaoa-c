import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#2E5C9C",
    "#D84141",
    "#808080",
    "#E6A023",
    "#2D8A6B",
    "#20252B",
    "#C4B5A0",
]

def set_my_style(use_latex: bool = True):
    if use_latex:
        font_style = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    else:
        font_style = {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
        }

    my_style = {
        "figure.figsize": (4, 2.5),
        "figure.dpi": 150,
        "axes.grid": True,
        "axes.grid.which": "both",
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "lines.markersize": 3.5,
        "lines.linewidth": 1.0,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "xtick.minor.visible": True,
        "ytick.minor.size": 1.5,
        "ytick.minor.visible": True,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "legend.edgecolor": "#E0E0E0",
        "legend.fancybox": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.dpi": 300,
        **font_style,
    }


    # Apply the style
    # default_rc = mpl.rcParamsDefault
    # mpl.rcParams.update(default_rc)
    mpl.rcParams.update(my_style)

    sns.set_palette(COLORS)

def subplots(*args, **kwargs):
    fig, axs = plt.subplots(*args, **kwargs)
    axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    for ax in axs_flat:
       ax.grid(which='minor', alpha=0.2)
    return fig, axs

def mpl_reset():
    plt.style.use('default')


def shades(num: int | None, col_idx: int = 0, offset: int = 0):
    if num is None:
        return sns.light_palette(COLORS[col_idx], as_cmap=True)
    else:
        return sns.light_palette(COLORS[col_idx], num + offset)[-num:]

def diverging(col_idxs: tuple[int, int] | None, central: str | None = None):
    if col_idxs is None:
        col_idxs = (0, 1)


