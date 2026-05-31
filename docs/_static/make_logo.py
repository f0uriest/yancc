"""Generate the yancc logo: trapped-particle "banana" orbits in a poloidal plane.

Running this module writes, into ``images/``:

    logo.png   -- wide logo (orbits + "yancc" text) for docs / README
    icon.png   -- square text-free icon
    icon.ico   -- multi-resolution favicon
"""

import os

import matplotlib.pyplot as plt
import numpy as np

CMAP = plt.get_cmap("turbo")
# Neutral mid-gray for the axis / flux surfaces: keeps usable contrast
# against both white and dark backgrounds (the vivid turbo orbits do too, as
# long as we avoid the darkest ends of the colormap).
NEUTRAL = "#808080"
TEXTCOLOR =  "#AD06F4"
OUTDIR = os.path.join(os.path.dirname(__file__), "images")


def banana(r0, theta_b, width, n=400):
    """Closed guiding-center banana orbit in the poloidal (X, Z) plane.

    Parameters
    ----------
    r0 : float
        Minor radius of the orbit (distance from magnetic axis).
    theta_b : float
        Bounce angle; the orbit oscillates in poloidal angle over [-theta_b, theta_b].
    width : float
        Banana width (radial separation of the two legs at the midplane).
    """
    th = np.linspace(-theta_b, theta_b, n)
    # leg separation: max at the outboard midplane (theta=0), zero at the tips
    f = np.cos(np.pi * th / (2 * theta_b))
    r_out = r0 + 0.5 * width * f
    r_in = r0 - 0.5 * width * f
    # outbound leg then return leg -> closed crescent
    r = np.concatenate([r_out, r_in[::-1]])
    t = np.concatenate([th, th[::-1]])
    return r * np.cos(t), r * np.sin(t)


def draw_orbits(ax):
    """Draw the faint flux surfaces and the nest of banana orbits onto ``ax``."""
    th = np.linspace(0, 2 * np.pi, 300)
    for rr in np.linspace(0.18, 1.0, 7):
        ax.plot(rr * np.cos(th), rr * np.sin(th),
                color=NEUTRAL, lw=0.8, alpha=0.30, zorder=0)

    radii = np.linspace(0.32, 0.92, 6)
    for i, r0 in enumerate(radii):
        x, y = banana(r0, theta_b=1.05, width=0.30)
        c = CMAP(0.12 + 0.76 * i / (len(radii) - 1))
        ax.plot(x, y, color=c, lw=5.5, alpha=0.95, zorder=2 + i,
                solid_capstyle="round")

    ax.plot(0, 0, marker="+", ms=12, mew=2.0, color=NEUTRAL, zorder=1)  # magnetic axis
    ax.set_aspect("equal")
    ax.axis("off")


def make_logo(fname="images/logo.png"):
    """Wide logo: orbits on the left, "yancc" wordmark to the right."""
    fig, ax = plt.subplots(figsize=(8.0, 3.0), dpi=200)
    draw_orbits(ax)
    ax.set_xlim(-1.1, 4.4)
    ax.set_ylim(-1.2, 1.2)
    ax.text(2.75, 0.0, "yancc", ha="center", va="center",
            fontsize=72, fontweight="bold", family="monospace", color=TEXTCOLOR)
    fig.savefig(fname, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", fname)


def make_icon(fname="images/icon.png"):
    """Square, text-free icon."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    draw_orbits(ax)
    ax.set_xlim(-1.05, 1.25)
    ax.set_ylim(-1.15, 1.15)
    fig.savefig(fname, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", fname)


def make_favicon(src="images/icon.png", fname="images/icon.ico"):
    """Convert the square icon into a multi-resolution .ico favicon."""
    from PIL import Image

    img = Image.open(src).convert("RGBA")
    # pad to a square canvas so the .ico isn't distorted
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    canvas.save(fname, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)])
    print("wrote", fname)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    here = os.path.dirname(__file__)
    make_logo(os.path.join(here, "images", "logo.png"))
    make_icon(os.path.join(here, "images", "icon.png"))
    make_favicon(os.path.join(here, "images", "icon.png"),
                 os.path.join(here, "images", "icon.ico"))
