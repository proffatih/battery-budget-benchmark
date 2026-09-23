"""Figure 1: the estimation chain, as a labelled conceptual strip.

The artwork is a generative-model render supplied as figures/gems/gemd (it carries
no measured data); every label below it is typeset here so that no text is baked
into the image.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(ROOT, "Batteries_MDPI", "figures")
GEMS = os.path.join(FIG, "gems")
# the render is dropped into figures/gems/ by hand; take the newest file there
cand = [os.path.join(GEMS, f) for f in os.listdir(GEMS) if not f.startswith(".")]
SRC = max(cand, key=os.path.getmtime)
print("artwork:", os.path.relpath(SRC, ROOT))

img = mpimg.imread(SRC)
h, w = img.shape[:2]

setup(8.5)
fig = plt.figure(figsize=(7.2, 7.2 * h / w + 0.52))
ax_img = fig.add_axes([0, 0.20, 1, 0.82]); ax_img.axis("off")
ax_img.imshow(img, interpolation="lanczos")

ax = fig.add_axes([0, 0, 1, 0.23]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

PANELS = [
    ("ageing cells",        "capacity fades\nwith use",            C1),
    ("charge window",       "$N$ samples inside\na voltage span",  C2),
    ("estimator",           "int8 bytes,\nMACs per inference",     C3),
    ("SoH estimate",        "cell-wise\nerror and cost",           C4),
    ("deployment",          "a different cell\ndesign breaks it",  C5),
]
# panel centres, matched to the five boxes of the artwork
centres = np.linspace(0.5 / 5, 1 - 0.5 / 5, 5)
for (title, sub, col), cx in zip(PANELS, centres):
    ax.text(cx, 0.95, title, ha="center", va="top", fontsize=8.0, weight="bold", color=col)
    ax.text(cx, 0.58, sub, ha="center", va="top", fontsize=6.9, color=INK2, linespacing=1.32)

save(fig, os.path.join(FIG, "fig00_chain"))
