"""Light plot theme (dark spec with bg/text swapped for readability)."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_DEEP_MIDNIGHT = "#0B0F19"
_OFF_WHITE = "#E2E8F0"
_MUTED_SLATE = "#1E293B"
_CYAN = "#00B0FF"
_MINT = "#00E676"
_CORAL = "#FF5252"


@dataclass(frozen=True)
class TerminalTheme:
    bg: str = _OFF_WHITE
    text: str = _DEEP_MIDNIGHT
    grid: str = _MUTED_SLATE
    alpha: str = _CYAN
    alpha_alt: str = _MINT
    bearish: str = _CORAL
    font_mono: tuple[str, ...] = ("JetBrains Mono", "Fira Code", "Consolas", "monospace")


THEME = TerminalTheme()


def apply_theme() -> TerminalTheme:
    """Apply Modern Terminal rcParams globally."""
    t = THEME
    mpl.rcParams.update(
        {
            "figure.facecolor": t.bg,
            "axes.facecolor": t.bg,
            "savefig.facecolor": t.bg,
            "savefig.edgecolor": t.bg,
            "text.color": t.text,
            "axes.labelcolor": t.text,
            "xtick.color": t.text,
            "ytick.color": t.text,
            "axes.edgecolor": t.grid,
            "axes.titlecolor": t.text,
            "grid.color": t.grid,
            "grid.alpha": 0.45,
            "font.family": "monospace",
            "font.monospace": list(t.font_mono),
            "legend.facecolor": t.bg,
            "legend.edgecolor": t.grid,
            "legend.labelcolor": t.text,
            "axes.grid": False,
        }
    )
    return t


def style_axes(ax: plt.Axes, *, theme: TerminalTheme | None = None) -> None:
    """Style a single axes with the terminal palette."""
    t = theme or THEME
    ax.set_facecolor(t.bg)
    ax.tick_params(colors=t.text, which="both")
    ax.xaxis.label.set_color(t.text)
    ax.yaxis.label.set_color(t.text)
    ax.title.set_color(t.text)
    for spine in ax.spines.values():
        spine.set_color(t.grid)
    ax.grid(False)


def style_figure(fig: plt.Figure, *, theme: TerminalTheme | None = None) -> None:
    """Style all axes on a figure."""
    t = theme or THEME
    fig.patch.set_facecolor(t.bg)
    for ax in fig.axes:
        style_axes(ax, theme=t)


def diverging_cmap(*, theme: TerminalTheme | None = None) -> LinearSegmentedColormap:
    """Bearish red -> neutral bg -> bullish cyan."""
    t = theme or THEME
    return LinearSegmentedColormap.from_list("terminal_div", [t.bearish, t.bg, t.alpha])
