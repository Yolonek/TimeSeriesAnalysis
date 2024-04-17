import mplcyberpunk
from matplotlib.collections import PolyCollection


def enhance_plot(figure, axes, glow=False, alpha_gradient=0, lines=True):
    figure.set_facecolor('black')
    axes.set_facecolor('black')
    if glow:
        if lines:
            mplcyberpunk.make_lines_glow(ax=axes)
        else:
            mplcyberpunk.make_scatter_glow(ax=axes)
    if 1 > alpha_gradient > 0:
        mplcyberpunk.add_gradient_fill(ax=axes, alpha_gradientglow=alpha_gradient)


def set_polygon_color(axes, color):
    for item in axes.collections:
        if isinstance(item, PolyCollection):
            item.set_facecolor(color)
