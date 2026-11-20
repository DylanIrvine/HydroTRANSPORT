# figures

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mimage
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
import numpy as np
from pathlib import Path

# default colours. The app lets the user change them
DEFAULT_COLOURS = {
    'Source': '#CC004C',
    'Series 1': '#0089D0',
    'Series 2': '#F37021',
    'Series 3': '#0DB14B',
    'Series 4': '#6460AA',
    'Series 5': '#FCB711',
}
# plume maps run from pale gold through gold and tangelo to carmine. Below the lowest level is left white
PLUME = LinearSegmentedColormap.from_list('plume', ['#FEF1CC', '#FCB711', '#F37021', '#CC004C', '#5A0A2E'])
LOWEST_SHOWN = 0.005     # lowest level on a linear colour scale, as a fraction of the top
WATERMARK = 'Dylan Irvine, Charles Darwin University'
BADGE_ICON = Path(__file__).resolve().parents[1] / 'logo' / 'HydroTRANSPORT_badge.png'
AQUIFER = '#EFE3C8'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 9,
    'font.weight': 'normal',
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

BADGE_PT = 14       # height of the app icon in the 'Made using' badge, in points


def add_badge(fig, x, y, transform):
    # 'Made using HydroTRANSPORT' with the app icon, right-aligned at (x, y). Skipped if the icon is missing
    try:
        img = mimage.imread(BADGE_ICON)
    except (FileNotFoundError, OSError):
        return
    packed = HPacker(children=[TextArea('Made using HydroTRANSPORT', textprops=dict(fontsize=6, color='0.5')),
                               OffsetImage(img, zoom=BADGE_PT / img.shape[0])],
                     align='center', pad=0, sep=3)
    fig.add_artist(AnnotationBbox(packed, (x, y), xycoords=transform, box_alignment=(1, 0.5),
                                  frameon=False, pad=0, annotation_clip=False))


def add_watermark(fig):
    # credit line bottom left and the 'Made using' badge bottom right, in a strip below the
    # plot so both can be cropped off
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    below = fig.transFigure + mtransforms.ScaledTranslation(0, -(BADGE_PT / 2 + 4) / 72, fig.dpi_scale_trans)
    fig.text(box.x0 / width, box.y0 / height, WATERMARK, transform=below,
             ha='left', va='center', fontsize=6, color='0.5')
    add_badge(fig, box.x1 / width, box.y0 / height, below)


def cm(width, height):
    return (width / 2.54, height / 2.54)


def legend_above(ax, ncols=3):
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncols=ncols, borderaxespad=0.2,
              handlelength=1.5, columnspacing=1.2)


def series_colours(c, n):
    cycle = [c[f'Series {i}'] for i in range(1, 6)]
    return [cycle[i % len(cycle)] for i in range(n)]


def time_label(t, unit, factor):
    return f'{t / factor:g} {unit}'


def _draw_source(ax, key, geometry, c, plane='xy'):
    # the source on a map. plane is 'xy' or 'xz'
    if key in ('point2', 'point3'):
        ax.plot(0, geometry.get('depth', 0.0) if plane == 'xz' else 0, marker='*', color=c['Source'], mec='white', mew=0.5, ms=11, ls='none', zorder=5)
    elif plane == 'xy':
        ax.plot([0, 0], [-geometry['width'] / 2, geometry['width'] / 2], color=c['Source'], lw=3,
                solid_capstyle='butt', zorder=5, clip_on=False)
    else:
        depth = geometry.get('depth', 0.0)
        ax.plot([0, 0], [depth - geometry['height'] / 2, depth + geometry['height'] / 2], color=c['Source'], lw=3,
                solid_capstyle='butt', zorder=5, clip_on=False)


def _draw_points(ax, points, a='x', b='y', names=True):
    for q in points:
        ax.plot(q[a], q[b], '^', mfc='white', mec='0.15', mew=1.0, ms=6, zorder=6)
        if names:
            ax.annotate(q['name'], (q[a], q[b]), textcoords='offset points', xytext=(0, 6), ha='center',
                        fontsize=7, color='0.15', zorder=6,
                        path_effects=[patheffects.withStroke(linewidth=2, foreground='white')])


# %% set-up diagram

def setup(key, geometry, points, ext, c):
    # the source, the flow direction and the observation points, seen from above.
    # The 3D solutions add a side view
    x0, x1, half_y, half_z = ext
    three = key in ('point3', 'patch')
    one = key in ('inlet1', 'inlet3')
    fig, axes = plt.subplots(2 if three else 1, 1, figsize=cm(16, 9.5 if three else (3.6 if one else 6)),
                             sharex=True, squeeze=False)
    top = axes[0, 0]
    pad = 0.04 * (x1 - x0)
    if one:
        top.axhspan(-1, 1, color=AQUIFER, lw=0)
        top.plot([0, 0], [-1, 1], color=c['Source'], lw=3, solid_capstyle='butt', clip_on=False)
        top.annotate('Inlet (source)', (0, 1), textcoords='offset points', xytext=(4, 4), color=c['Source'], fontsize=8)
        for q in points:
            top.plot(q['x'], 0, '^', mfc='white', mec='0.15', ms=6, zorder=6)
            top.annotate(q['name'], (q['x'], 0), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7)
        top.set_ylim(-1.6, 2.6)
        top.set_yticks([])
        top.spines['left'].set_visible(False)
        arrow_y = -1.35
    else:
        top.set_facecolor(AQUIFER)
        _draw_source(top, key, geometry, c)
        _draw_points(top, points)
        top.set_ylim(-half_y, half_y)
        top.set_ylabel('y (m)')
        top.set_title('From above', loc='left', fontsize=8, color='0.3')
        arrow_y = -0.82 * half_y
    top.annotate('', (x0 + pad + 0.16 * (x1 - x0), arrow_y), (x0 + pad, arrow_y),
                 arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.2))
    top.text(x0 + pad + 0.18 * (x1 - x0), arrow_y, 'Groundwater flow', va='center', fontsize=8, color='0.3')
    if three:
        side = axes[1, 0]
        side.set_facecolor(AQUIFER)
        _draw_source(side, key, geometry, c, plane='xz')
        _draw_points(side, points, 'x', 'z')
        side.set_ylim(geometry.get('depth', 0.0) + half_z, 0)
        side.set_ylabel('Depth, z (m)')
        side.set_title('From the side. The water table is at z = 0', loc='left', fontsize=8, color='0.3')
    axes[-1, 0].set_xlim(x0, x1)
    axes[-1, 0].set_xlabel('x, distance along the flow (m)')
    fig.tight_layout()
    return fig


# %% results

def breakthrough(curves, points, c, unit, factor, source_label):
    # source history above, concentration at each observation point below
    fig, (top, ax) = plt.subplots(2, 1, figsize=cm(16, 11), sharex=True, height_ratios=[1, 3])
    time = curves['Time'] / factor
    top.step(time, curves['Source'], where='post', color=c['Source'], lw=1.6)
    top.fill_between(time, curves['Source'], step='post', color=c['Source'], alpha=0.12, lw=0)
    top.set_ylabel(source_label)
    top.set_ylim(0, None)
    for q, colour in zip(points, series_colours(c, len(points))):
        ax.plot(time, curves[q['name']], color=colour, lw=1.6, label=q['name'])
    ax.set_xlabel(f'Time ({unit})')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_xlim(time.iloc[0], time.iloc[-1])
    ax.set_ylim(0, None)
    legend_above(ax, ncols=5)
    fig.align_ylabels()
    fig.tight_layout()
    return fig


def profile(xs, values, times, points, c, unit, factor, centreline, source_x=None):
    # concentration along x at each chosen time. source_x marks a point source, where the lines have a gap
    fig, ax = plt.subplots(figsize=cm(16, 8.5))
    for column, (t, colour) in enumerate(zip(times, series_colours(c, len(times)))):
        ax.plot(xs, values[:, column], color=colour, lw=1.6, label=time_label(t, unit, factor))
    for q in points:
        ax.axvline(q['x'], color='0.6', lw=0.8, ls=':')
        ax.annotate(q['name'], (q['x'], 1), xycoords=('data', 'axes fraction'), textcoords='offset points',
                    xytext=(2, -8), fontsize=7, color='0.4')
    ax.set_xlabel('x, distance along the flow (m)')
    ax.set_ylabel('Concentration on the centreline (mg/L)' if centreline else 'Concentration (mg/L)')
    if source_x is not None:
        ax.axvline(source_x, color=c['Source'], lw=0.8, ls=':')
        ax.annotate('Source', (source_x, 1), xycoords=('data', 'axes fraction'), textcoords='offset points',
                    xytext=(2, -8), fontsize=7, color=c['Source'])
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, None)
    legend_above(ax, ncols=5)
    fig.tight_layout()
    return fig


def nice_top(value, log):
    # round the top of a colour scale up to 1, 2, 2.5 or 5 times a power of ten (a power of ten if log)
    if not np.isfinite(value) or value <= 0:
        return 1.0
    power = 10.0 ** np.floor(np.log10(value))
    if log:
        return power if np.isclose(value, power) else 10 * power
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * power * (1 + 1e-9):
            return step * power


def levels(top, log):
    # contour levels up to top: 10 even steps starting just above zero, or half decades over
    # 4 orders of magnitude. Anything below the lowest level is left white
    if not np.isfinite(top) or top <= 0:
        top = 1.0
    if log:
        return top * 10.0 ** np.arange(-4, 0.01, 0.5)
    return top * np.concatenate([[LOWEST_SHOWN], np.linspace(0.1, 1, 10)])


def _filled(ax, a, b, values, lev, log):
    norm = BoundaryNorm(lev, PLUME.N)
    shown = np.where(np.isfinite(values), values, lev[-1])
    return ax.contourf(a, b, shown, levels=lev, cmap=PLUME, norm=norm, extend='max')


def _bar_ticks(bar, lev, log):
    ticks = lev[::2]
    bar.set_ticks(ticks, labels=[f'{value:.3g}' for value in ticks])
    bar.set_label('Concentration (mg/L)')


def _colourbar(fig, mappable, axes, lev, log):
    bar = fig.colorbar(mappable, ax=axes, pad=0.02, aspect=30, fraction=0.05)
    _bar_ticks(bar, lev, log)
    return bar


def maps(key, geometry, xs, ys, grids, times, points, c, unit, factor, lev, log):
    # plume from above at each chosen time, on one colour scale
    n = len(times)
    fig, axes = plt.subplots(n, 1, figsize=cm(16, 2.2 + 4.3 * n), sharex=True, squeeze=False,
                             layout='constrained')
    for ax, values, t in zip(axes[:, 0], grids, times):
        filled = _filled(ax, xs, ys, values, lev, log)
        _draw_source(ax, key, geometry, c)
        _draw_points(ax, points, names=ax is axes[0, 0])
        ax.set_ylabel('y (m)')
        ax.set_title(time_label(t, unit, factor), loc='left', fontsize=9)
    axes[-1, 0].set_xlabel('x, distance along the flow (m)')
    _colourbar(fig, filled, list(axes[:, 0]), lev, log)
    return fig


def slices(key, geometry, xs, ys, zs, plan, side, across, cut, t, points, c, unit, factor, lev, log):
    # a 3D plume at one time as three slices: from above, from the side, and across the flow.
    # cut holds the position of each slice: z for the plan, y for the side view, x for the cross section
    fig = plt.figure(figsize=cm(16, 12), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[2.6, 1])
    ax_plan = fig.add_subplot(grid[0, 0])
    ax_side = fig.add_subplot(grid[1, 0], sharex=ax_plan)
    ax_cross = fig.add_subplot(grid[1, 1], sharey=ax_side)
    filled = _filled(ax_plan, xs, ys, plan, lev, log)
    _filled(ax_side, xs, zs, side, lev, log)
    _filled(ax_cross, ys, zs, across, lev, log)
    _draw_source(ax_plan, key, geometry, c)
    _draw_source(ax_side, key, geometry, c, plane='xz')
    _draw_points(ax_plan, points)
    _draw_points(ax_side, points, 'x', 'z', names=False)
    line = dict(color='0.25', lw=0.8, ls='--')
    ax_plan.axhline(cut['y'], **line)
    ax_plan.axvline(cut['x'], **line)
    ax_side.axhline(cut['z'], **line)
    ax_side.axvline(cut['x'], **line)
    ax_cross.axhline(cut['z'], **line)
    ax_cross.axvline(cut['y'], **line)
    ax_plan.set_title(f"From above, at z = {cut['z']:g} m", loc='left', fontsize=9)
    ax_side.set_title(f"From the side, at y = {cut['y']:g} m", loc='left', fontsize=9)
    ax_cross.set_title(f"Across the flow, at x = {cut['x']:g} m", loc='left', fontsize=9)
    ax_plan.set_ylabel('y (m)')
    ax_side.set_ylabel('Depth, z (m)')
    ax_side.set_ylim(zs[-1], zs[0])
    ax_side.set_xlabel('x, distance along the flow (m)')
    ax_cross.set_xlabel('y (m)')
    plt.setp(ax_plan.get_xticklabels(), visible=False)
    plt.setp(ax_cross.get_yticklabels(), visible=False)
    bar = fig.colorbar(filled, ax=fig.add_subplot(grid[0, 1], frame_on=False, xticks=[], yticks=[]),
                       location='left', fraction=0.9, aspect=12, pad=0)
    _bar_ticks(bar, lev, log)
    fig.suptitle(time_label(t, unit, factor), x=0.01, ha='left', fontsize=10)
    return fig


def isosurfaces(xs, ys, zs, volume, shells, c, title):
    # an interactive 3D view: one see-through shell for each concentration in shells.
    # volume is indexed [z, y, x]
    import plotly.graph_objects as go
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
    values = np.where(np.isfinite(volume), volume, np.nanmax(volume))
    fig = go.Figure()
    colours = [c['Series 5'], c['Series 2'], c['Source']]
    for value, colour, opacity in zip(shells, colours, (0.18, 0.35, 0.8)):
        fig.add_trace(go.Isosurface(
            x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=values.ravel(), isomin=value, isomax=value,
            surface_count=1, colorscale=[[0, colour], [1, colour]], showscale=False, opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False), name=f'{value:.3g} mg/L', showlegend=True,
            hovertemplate='x %{x:.0f} m<br>y %{y:.1f} m<br>z %{z:.1f} m<extra>' + f'{value:.3g} mg/L</extra>'))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)), margin=dict(l=0, r=0, t=40, b=0), height=520,
        legend=dict(orientation='h', y=1.0, yanchor='bottom', x=0),
        scene=dict(xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='depth z (m)',
                   zaxis_autorange='reversed', aspectmode='manual',
                   aspectratio=dict(x=2.2, y=1, z=0.7), camera=dict(eye=dict(x=-1.6, y=-2.0, z=1.1))))
    return fig
