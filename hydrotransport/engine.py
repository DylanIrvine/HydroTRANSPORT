# source history, output grids and summary tables. Metres, days and g/m³

import numpy as np
import pandas as pd

from . import solutions

MAX_STAGES = 50


def source_history(table):
    # start times and values from the source table: sorted, and checked. Returns (ts, cs, problems)
    problems = []
    rows = table.dropna(how='all')
    if rows.isna().any().any():
        problems.append('Fill in both columns of every row in the source history.')
        rows = rows.dropna()
    ts = rows.iloc[:, 0].to_numpy(float)
    cs = rows.iloc[:, 1].to_numpy(float)
    order = np.argsort(ts, kind='stable')
    ts, cs = ts[order], cs[order]
    if ts.size == 0:
        problems.append('Add at least one row to the source history.')
    if (ts < 0).any():
        problems.append('Source start times cannot be negative.')
    if (cs < 0).any():
        problems.append('Source values cannot be negative.')
    if np.unique(ts).size < ts.size:
        problems.append('Give each row of the source history a different start time.')
    if ts.size > MAX_STAGES:
        problems.append(f'The source history can have at most {MAX_STAGES} rows.')
    if ts.size and not (cs > 0).any():
        problems.append('The source is zero throughout. Give at least one row a value above zero.')
    return ts, cs, problems


def derived(p, dims):
    # values worked out from the inputs, as (name, value, unit) rows
    v, Dx, Dy, Dz = solutions.retarded(p)
    rows = [('Groundwater velocity, v', p['v'], 'm/d'),
            ('Retardation factor, R', p['R'], '-'),
            ('Solute velocity, v / R', v, 'm/d'),
            ('Longitudinal dispersion coefficient, D_L = α_L v + D*', Dx * p['R'], 'm²/d')]
    if dims >= 2:
        rows.append(('Horizontal transverse dispersion coefficient, D_T', Dy * p['R'], 'm²/d'))
    if dims == 3:
        rows.append(('Vertical transverse dispersion coefficient, D_V', Dz * p['R'], 'm²/d'))
    if p['lam'] > 0:
        rows.append(('Decay rate, λ', p['lam'], '1/d'))
        rows.append(('Half-life', np.log(2) / p['lam'], 'd'))
    return pd.DataFrame(rows, columns=['Quantity', 'Value', 'Unit'])


def breakthrough(key, p, geometry, ts, cs, points, times):
    # concentration through time at each observation point. One column per point
    out = pd.DataFrame({'Time': times, 'Source': solutions.source_value(ts, cs, times)})
    for point in points:
        out[point['name']] = solutions.concentration(key, p, geometry, ts, cs, point['x'], point['y'],
                                                     point['z'], times)
    return out


def profiles(key, p, geometry, ts, cs, xs, times, y=0.0, z=0.0):
    # concentration along x at each chosen time. One column per time
    return np.column_stack([solutions.concentration(key, p, geometry, ts, cs, xs, y, z, t) for t in times])


def grid(key, p, geometry, ts, cs, a, b, t, plane='xy', offset=0.0):
    # concentration on a plane at time t. plane is 'xy' (z = offset), 'xz' (y = offset) or
    # 'yz' (x = offset). Returns an array with one row per value of the second axis
    A, B = np.meshgrid(a, b)
    if plane == 'xy':
        return solutions.concentration(key, p, geometry, ts, cs, A, B, offset, t)
    if plane == 'xz':
        return solutions.concentration(key, p, geometry, ts, cs, A, offset, B, t)
    return solutions.concentration(key, p, geometry, ts, cs, offset, A, B, t)


def summary(key, p, ts, cs, points, curves):
    # one row per observation point: distance, travel time and the peak
    v = p['v'] / p['R']
    first = ts[np.argmax(cs > 0)]
    rows = []
    for point in points:
        c = curves[point['name']].to_numpy()
        ok = np.isfinite(c)
        row = {'Point': point['name'], 'x (m)': point['x']}
        if key not in ('inlet1', 'inlet3'):
            row['y (m)'] = point['y']
        if key in ('point3', 'patch'):
            row['z (m)'] = point['z']
        row['Travel time by flow alone, x R / v (d)'] = point['x'] / v if point['x'] > 0 else np.nan
        row['Arrival by flow alone (d)'] = first + point['x'] / v if point['x'] > 0 else np.nan
        row['Distance / dispersivity, x / α_L'] = point['x'] / p['al'] if p['al'] > 0 else np.nan
        row['Peak concentration (mg/L)'] = c[ok].max() if ok.any() else np.nan
        row['Time of peak (d)'] = curves['Time'].to_numpy()[ok][np.argmax(c[ok])] if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def extent(key, p, geometry, points, t_max, spread=2.2):
    # plot limits that hold the plume at t_max and the observation points.
    # spread is the number of standard deviations shown across the flow.
    # Returns x0, x1, half-width in y and half-height in z
    v, Dx, Dy, Dz = solutions.retarded(p)
    reach = v * t_max + 2.5 * np.sqrt(2 * Dx * t_max)
    if p['lam'] > 0:
        # with decay the plume stops growing. Length at which a steady plume falls to 0.1%
        u = np.sqrt(v ** 2 + 4 * p['lam'] * Dx)
        reach = min(reach, np.log(1000) * 2 * Dx / (u - v))
    x1 = max([reach] + [1.15 * q['x'] for q in points])
    travel = min(t_max, x1 / v)
    half_y = geometry.get('width', 0.0) / 2 + spread * np.sqrt(2 * Dy * travel)
    half_z = geometry.get('height', 0.0) / 2 + spread * np.sqrt(2 * Dz * travel)
    half_y = max([half_y] + [1.15 * abs(q['y']) for q in points])
    half_z = max([half_z] + [1.15 * abs(q['z']) for q in points])
    x0 = -0.08 * x1 if key in ('point2', 'point3') else 0.0
    x0 = min([x0] + [1.15 * q['x'] for q in points if q['x'] < 0])
    return x0, x1, half_y, half_z


def last_active(ts, cs, times):
    # the latest of the chosen times at which the source is still on, or the first of them
    on = [t for t in times if solutions.source_value(ts, cs, t * (1 - 1e-12)) > 0]
    return max(on) if on else min(times)
