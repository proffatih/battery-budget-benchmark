"""Shared figure style for the review. Print-first: vector PDF + 300 dpi PNG.
Palette = validated categorical default (slots 1-3 all-pairs safe; slots 1-5 adjacent-safe).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"

# categorical slots, fixed order - never cycled
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]

def setup(base=8.5):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.titlesize": base + 1.0,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)

def save(fig, path_noext):
    fig.savefig(path_noext + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_noext + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", path_noext + ".pdf/.png")

CM = 1 / 2.54


def fit_text(ax, x, y, text, max_w, max_h, fs=6.4, color=None, weight=None,
             ha="center", va="center", min_fs=4.6):
    """Place text inside a (max_w x max_h) data-unit box: try word wraps, then shrink."""
    import textwrap
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transData.inverted()

    def extent(t):
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return abs(x1 - x0), abs(y1 - y0)

    words = text.split()
    cands = []
    for nline in range(1, 5):
        w = max(1, -(-len(words) // nline))
        cand = "\n".join(textwrap.wrap(text, max(1, len(text) // nline + 3),
                                       break_long_words=False))
        if cand not in cands:
            cands.append(cand)
    best = None
    for cand in cands:
        t = ax.text(x, y, cand, fontsize=fs, ha=ha, va=va, color=color,
                    weight=weight, linespacing=1.28)
        w, h = extent(t)
        s = min(max_w / w, max_h / h, 1.0)
        if best is None or s > best[1]:
            if best is not None:
                best[0].remove()
            best = [t, s, cand]
        else:
            t.remove()
    t, s, cand = best
    if s < 1.0:
        t.set_fontsize(max(min_fs, fs * s * 0.98))
    return t
