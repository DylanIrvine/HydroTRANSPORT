# HydroTRANSPORT
# solute transport in groundwater, using analytical solutions with a source that changes in steps.
# Run with: streamlit run app.py

import datetime as dt
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from hydrotransport import VERSION as APP_VERSION
from hydrotransport import engine, plots, run_counter, solutions, workbook

# %% page setup

APP_NAME = 'HydroTRANSPORT'
# app address, used in the citation
APP_URL = 'https://hydrotransport.streamlit.app/'
LOGO = Path('logo/HydroTRANSPORT_logo.png')
FAVICON = Path('logo/HydroTRANSPORT_icon.png')
TIME = {'years': 365.25, 'days': 1.0}
N_TIMES = 400       # points on a breakthrough curve
N_X = 400           # points on a profile
MAP_NX, MAP_NY, MAP_NZ = 140, 61, 41

st.set_page_config(page_title=APP_NAME, page_icon=str(FAVICON) if FAVICON.exists() else None)

# %% header

head_text, head_logo = st.columns([3, 1], vertical_alignment='center')
with head_text:
    st.title(APP_NAME)
    st.subheader('Solute transport in groundwater')
    st.write('Dylan Irvine, Charles Darwin University.')
    st.caption(f'Version {APP_VERSION}')
with head_logo:
    if LOGO.exists():
        st.image(str(LOGO), width='stretch')

st.markdown(
    'A solute released into groundwater is carried along by the flow, spreads out as it goes, and '
    'may be slowed by sorption or lost to decay. HydroTRANSPORT shows how a plume grows, and how '
    'it clears once the source is reduced or switched off, using six analytical solutions in one, '
    'two and three dimensions. Choose a source, describe the aquifer, set a source history, and '
    'press Run. The results can be downloaded as one Excel workbook '
    'with plots.\n\n'
    'These solutions suit teaching and screening. They assume a uniform aquifer with steady flow '
    'in one direction, so they do not replace a site-specific model.')

if 'run_count' not in st.session_state:
    st.session_state['run_count'] = run_counter.get_runs()
if st.session_state['run_count'] is not None:
    st.caption(f"This tool has been run {st.session_state['run_count']:,} times.")

with st.expander('**Privacy and data handling**'):
    st.markdown(
        '**Your data**\n'
        '- Everything you enter is held in memory on the server for your session and discarded '
        'when it ends. Nothing is saved to a database or to permanent storage.\n'
        '- The results exist only in the files you download.\n'
        '- Nothing you enter is sent to any other service.\n\n'
        '**You**\n'
        '- No login, and no record kept by this app of who you are or where you are. The app does '
        'not log IP addresses or locations.\n'
        '- The only thing recorded is a single running total: the number of completed runs. It '
        'has no link to any user, location or input.\n\n'
        '**What this does not cover**\n'
        '- Processing happens on the server that hosts the app, not on your computer. If you are '
        'using the public web version, your inputs travel to that host (which may be outside '
        'Australia). The hosting provider and the Streamlit framework may use their own cookies '
        'and keep their own server logs, under their own privacy policies.\n'
        '- If your inputs are confidential, do not enter them in the public web version.')

with st.expander('**Background: the solutions and their equations**'):
    st.markdown(
        'All six solutions solve the advection-dispersion equation for a solute in steady, uniform '
        'flow in the $x$ direction, with linear sorption and first-order decay:')
    st.latex(r'R\,\frac{\partial C}{\partial t} = D_L \frac{\partial^2 C}{\partial x^2} + '
             r'D_T \frac{\partial^2 C}{\partial y^2} + D_V \frac{\partial^2 C}{\partial z^2} - '
             r'v \frac{\partial C}{\partial x} - \lambda R\, C')
    st.markdown(
        r'$C$ is concentration, $v$ is the groundwater velocity (Darcy flux divided by effective '
        r'porosity), $R$ is the retardation factor and $\lambda$ is the decay rate, applied to '
        r'dissolved and sorbed solute alike. The dispersion coefficients are '
        r'$D_L = \alpha_L v + D^*$, $D_T = \alpha_T v + D^*$ and $D_V = \alpha_V v + D^*$, where '
        r'the $\alpha$ are dispersivities and $D^*$ is the effective molecular diffusion '
        r'coefficient. The 1D solutions keep only the $x$ terms and the 2D solutions drop the $z$ '
        r'term. Below, $v$ and the $D$ values have already been divided by $R$.' '\n\n'
        '**1D, constant concentration at the inlet.** For a source $C_0$ that starts at $t = 0$ '
        r'(Ogata and Banks 1961 when $\lambda = 0$), with $u = \sqrt{v^2 + 4\lambda D_L}$:')
    st.latex(r'\frac{C}{C_0} = \frac{1}{2}\left[\exp\left(\frac{x(v-u)}{2D_L}\right)'
             r'\mathrm{erfc}\left(\frac{x-ut}{2\sqrt{D_L t}}\right) + '
             r'\exp\left(\frac{x(v+u)}{2D_L}\right)\mathrm{erfc}\left(\frac{x+ut}{2\sqrt{D_L t}}\right)\right]')
    st.markdown(
        '**1D, flux at the inlet.** Water of concentration $C_0$ flows in, so the condition at '
        r'$x = 0$ is $vC - D_L\,\partial C/\partial x = vC_0$. Without decay:')
    st.latex(r'\frac{C}{C_0} = \frac{1}{2}\mathrm{erfc}\left(\frac{x-vt}{2\sqrt{D_L t}}\right) + '
             r'\sqrt{\frac{v^2 t}{\pi D_L}}\exp\left(-\frac{(x-vt)^2}{4D_L t}\right) - '
             r'\frac{1}{2}\left(1+\frac{vx}{D_L}+\frac{v^2t}{D_L}\right)\exp\left(\frac{vx}{D_L}\right)'
             r'\mathrm{erfc}\left(\frac{x+vt}{2\sqrt{D_L t}}\right)')
    st.markdown(
        '**The other solutions** are written as an integral over the time since each part of the '
        'source was released. For the 2D point source with a mass rate $\\dot{M}(t)$ in an aquifer '
        'of thickness $b$ and porosity $n$:')
    st.latex(r'C = \frac{1}{4\pi n R\, b\sqrt{D_L D_T}}\int_0^t \frac{\dot{M}(t-\tau)}{\tau}'
             r'\exp\left(-\frac{(x-v\tau)^2}{4D_L\tau}-\frac{y^2}{4D_T\tau}-\lambda\tau\right)d\tau')
    st.markdown('For the 2D line source of width $W$ held at concentration $C_0(t)$:')
    st.latex(r'C = \int_0^t C_0(t-\tau)\,\frac{x}{4\sqrt{\pi D_L \tau^3}}'
             r'\exp\left(-\frac{(x-v\tau)^2}{4D_L\tau}-\lambda\tau\right)'
             r'\left[\mathrm{erfc}\left(\frac{-W/2-y}{2\sqrt{D_T\tau}}\right)-'
             r'\mathrm{erfc}\left(\frac{W/2-y}{2\sqrt{D_T\tau}}\right)\right]d\tau')
    st.markdown(
        'The 3D patch source of width $W$ and height $H$ has the same form, with a second bracket '
        'for $z$, $H$ and $D_V$, and a factor of one half. The 3D point source has a closed form '
        '(Wexler 1992, equation 105).\n\n'
        '**A source that changes.** The equations are linear, so a source that changes in steps '
        'is the sum of sources that each start at one of the changes. The closed forms are added '
        'in this way. For the integral forms the source history sits inside the integral, as '
        'written above, which avoids subtracting two nearly equal numbers long after a change.\n\n'
        '**What switching off means.** For the constant concentration sources (1D inlet, line and '
        'patch), a value of zero holds the source at zero concentration, as if clean water now '
        'enters there. For the flux inlet and the point sources, a value of zero stops solute '
        'entering.\n\n'
        '**Assumptions common to all six**\n'
        '- The aquifer is uniform and the flow is steady, in one direction, at one velocity.\n'
        '- The aquifer extends a long way downstream and, for 2D and 3D, sideways.\n'
        '- In the 3D solutions the water table is flat, at z = 0, and solute cannot cross it. This '
        'is done by adding a mirror image of the source above the water table. There is no '
        'recharge, and the aquifer has no base, so they apply until the plume reaches the base.\n'
        '- The 2D solutions assume the solute is mixed over the full thickness of the aquifer.\n'
        '- Sorption is linear, instant and reversible. Decay is first order.\n'
        '- The source does not change the flow, and the solute does not change the density of '
        'the water.\n'
        '- The aquifer holds none of the solute at the start.\n'
        '- The point sources have no size, so the concentration rises without limit close to them.')

with st.expander('**How to cite and references**'):
    st.write(
        'This app:\n\n'
        f'Irvine, D.J. ({dt.date.today().year}). {APP_NAME} (version {APP_VERSION}) '
        f'[Web application]. Charles Darwin University. {APP_URL}\n\n'
        'The solutions are from:\n\n'
        'Wexler, E.J. (1992). Analytical solutions for one-, two-, and three-dimensional solute '
        'transport in ground-water systems with uniform flow. U.S. Geological Survey Techniques of '
        'Water-Resources Investigations 03-B7, 190 pp. https://doi.org/10.3133/twri03B7\n\n'
        'Other references:\n\n'
        'Ogata, A., and Banks, R.B. (1961). A solution of the differential equation of longitudinal '
        'dispersion in porous media. U.S. Geological Survey Professional Paper 411-A.\n\n'
        'van Genuchten, M.Th., and Alves, W.J. (1982). Analytical solutions of the one-dimensional '
        'convective-dispersive solute transport equation. U.S. Department of Agriculture Technical '
        'Bulletin 1661.')

# %% 1. solution

st.header('1. Solution')
solution = st.selectbox('Analytical solution', list(solutions.SOLUTIONS))
info = solutions.SOLUTIONS[solution]
key, dims = info['key'], info['dims']
mass_source = info['source'] == 'mass'
st.markdown(f"{info['summary']}\n\n{info['when']}")

time_unit = st.radio('Time unit', list(TIME), horizontal=True)
tf = TIME[time_unit]

with st.expander('Plot colours'):
    st.caption('Change these to match your organisation. They apply to the line plots and the source.')
    colours = {}
    picker_columns = st.columns(3)
    for i, (role, default) in enumerate(plots.DEFAULT_COLOURS.items()):
        colours[role] = picker_columns[i % 3].color_picker(role, default)

problems = []
entered = {}

# %% 2. aquifer and transport

st.header('2. Aquifer and transport')
porosity = st.number_input('Effective porosity, n', value=0.3, min_value=0.01, max_value=1.0, format='%g')
entered['Effective porosity n'] = f'{porosity:g}'

v_mode = st.radio('Groundwater velocity', ['Enter velocity', 'Enter hydraulic conductivity and gradient'],
                  horizontal=True)
if v_mode == 'Enter velocity':
    velocity = st.number_input('Groundwater velocity, v (m/d)', value=0.1, min_value=0.0, format='%g')
    entered['Groundwater velocity v (m/d)'] = f'{velocity:g}'
else:
    col_k, col_i = st.columns(2)
    K = col_k.number_input('Hydraulic conductivity, K (m/d)', value=10.0, min_value=0.0, format='%g')
    gradient = col_i.number_input('Hydraulic gradient, i', value=0.003, min_value=0.0, format='%g')
    velocity = K * gradient / porosity
    entered['Hydraulic conductivity K (m/d)'] = f'{K:g}'
    entered['Hydraulic gradient i'] = f'{gradient:g}'
    st.caption(f'v = K x i / n = {velocity:.4g} m/d')
if velocity <= 0:
    problems.append('The groundwater velocity must be greater than zero.')

disp_columns = st.columns(dims)
alpha_l = disp_columns[0].number_input('Longitudinal dispersivity, α_L (m)', value=10.0, min_value=0.0, format='%g')
alpha_t = alpha_v = 0.0
entered['Longitudinal dispersivity α_L (m)'] = f'{alpha_l:g}'
if dims >= 2:
    alpha_t = disp_columns[1].number_input('Horizontal transverse dispersivity, α_T (m)', value=1.0,
                                           min_value=0.0, format='%g')
    entered['Horizontal transverse dispersivity α_T (m)'] = f'{alpha_t:g}'
if dims == 3:
    alpha_v = disp_columns[2].number_input('Vertical transverse dispersivity, α_V (m)', value=0.1,
                                           min_value=0.0, format='%g')
    entered['Vertical transverse dispersivity α_V (m)'] = f'{alpha_v:g}'
diffusion = st.number_input('Effective molecular diffusion coefficient, D* (m²/d)', value=0.0, min_value=0.0,
                            format='%g', help='Often about 0.00001 to 0.0001 m²/d. Leave at zero to ignore it.')
entered['Effective molecular diffusion coefficient D* (m²/d)'] = f'{diffusion:g}'
if velocity > 0 and min([alpha_l, alpha_t, alpha_v][:dims]) * velocity + diffusion <= 0:
    problems.append('Each dispersivity must be greater than zero, unless a diffusion coefficient is entered.')

sorption = st.radio('Sorption', ['None', 'Enter retardation factor', 'Enter distribution coefficient and bulk density'],
                    horizontal=True)
retardation = 1.0
if sorption == 'Enter retardation factor':
    retardation = st.number_input('Retardation factor, R', value=2.0, min_value=1.0, format='%g')
    entered['Retardation factor R'] = f'{retardation:g}'
elif sorption != 'None':
    col_kd, col_rho = st.columns(2)
    kd = col_kd.number_input('Distribution coefficient, K_d (L/kg)', value=0.2, min_value=0.0, format='%g')
    rho = col_rho.number_input('Dry bulk density, ρ_b (kg/L)', value=1.6, min_value=0.0, format='%g')
    retardation = 1 + rho * kd / porosity
    entered['Distribution coefficient K_d (L/kg)'] = f'{kd:g}'
    entered['Dry bulk density ρ_b (kg/L)'] = f'{rho:g}'
    st.caption(f'R = 1 + ρ_b x K_d / n = {retardation:.4g}')

decay = st.radio('Decay', ['None', 'Enter half-life', 'Enter decay rate'], horizontal=True)
decay_rate = 0.0
if decay == 'Enter half-life':
    half_life = st.number_input(f'Half-life ({time_unit})', value=5.0 if time_unit == 'years' else 1800.0,
                                min_value=0.0, format='%g')
    entered[f'Half-life ({time_unit})'] = f'{half_life:g}'
    if half_life <= 0:
        problems.append('The half-life must be greater than zero.')
    else:
        decay_rate = np.log(2) / (half_life * tf)
        st.caption(f'λ = ln(2) / half-life = {decay_rate:.4g} per day')
elif decay == 'Enter decay rate':
    decay_rate = st.number_input('Decay rate, λ (per day)', value=0.0005, min_value=0.0, format='%g')
    entered['Decay rate λ (per day)'] = f'{decay_rate:g}'

p = dict(v=float(velocity), n=float(porosity), al=float(alpha_l), ah=float(alpha_t), av=float(alpha_v),
         Dm=float(diffusion), R=float(retardation), lam=float(decay_rate))

# %% 3. source

st.header('3. Source')
geometry = {}
if key in ('strip', 'patch'):
    geometry['width'] = st.number_input('Source width across the flow, W (m)', value=20.0, min_value=0.0, format='%g')
    entered['Source width W (m)'] = f"{geometry['width']:g}"
if key == 'patch':
    geometry['height'] = st.number_input('Source height, H (m)', value=4.0, min_value=0.0, format='%g')
    entered['Source height H (m)'] = f"{geometry['height']:g}"
if key == 'point2':
    geometry['thickness'] = st.number_input('Aquifer thickness, b (m)', value=10.0, min_value=0.0, format='%g')
    entered['Aquifer thickness b (m)'] = f"{geometry['thickness']:g}"
if any(value <= 0 for value in geometry.values()):
    problems.append('The source size and aquifer thickness must be greater than zero.')
if dims == 3:
    geometry['depth'] = st.number_input(
        'Depth of the ' + ('centre of the ' if key == 'patch' else '') + 'source below the water table (m)',
        value=10.0, min_value=0.0, format='%g',
        help='The water table is at z = 0 and z increases downwards. Solute cannot cross the water table.')
    entered['Source depth below the water table (m)'] = f"{geometry['depth']:g}"
    if key == 'patch' and geometry['depth'] < geometry['height'] / 2:
        problems.append('The source reaches above the water table. Make it deeper or less high.')

source_column = 'Mass rate (g/d)' if mass_source else 'Concentration (mg/L)'
start_column = f'Start ({time_unit})'
if mass_source:
    st.caption('Each row sets the mass of solute released per day from its start time until the next '
               'row. End with a row of zero to switch the source off. The source is at x = 0, y = 0'
               + (', at the depth above.' if dims == 3 else '.'))
else:
    st.caption('Each row sets the source concentration from its start time until the next row. End '
               'with a row of zero to switch the source off. The source is at x = 0'
               + ('.' if dims == 1 else ', centred on y = 0' + (' and the depth above.' if dims == 3 else '.')))
switch_off = 5.0 if time_unit == 'years' else 1800.0
default_source = pd.DataFrame({start_column: [0.0, switch_off], source_column: [100.0, 0.0]})
source_table = st.data_editor(
    default_source, num_rows='dynamic', width='stretch', hide_index=True, key=f'source_{time_unit}_{mass_source}',
    column_config={start_column: st.column_config.NumberColumn(start_column, min_value=0.0, required=True),
                   source_column: st.column_config.NumberColumn(source_column, min_value=0.0, required=True)})
ts, cs, source_problems = engine.source_history(source_table)
ts = ts * tf
problems += source_problems

# %% 4. observation points and times

st.header('4. Observation points and times')
st.caption('Points where the concentration through time is wanted, such as monitoring bores. x is the '
           'distance from the source along the flow'
           + {1: '.', 2: ' and y is the distance across it.',
              3: ', y is the distance across it and z is the depth below the water table.'}[dims])
default_points = pd.DataFrame({'Name': ['Obs 1', 'Obs 2'], 'x': [100.0, 300.0], 'y': [0.0, 0.0], 'z': [10.0, 10.0]})
point_columns = ['Name', 'x', 'y', 'z'][:dims + 1]
point_table = st.data_editor(
    default_points[point_columns], num_rows='dynamic', width='stretch', hide_index=True, key=f'points_{dims}',
    column_config={'Name': st.column_config.TextColumn('Name', required=True),
                   'x': st.column_config.NumberColumn('x (m)', required=True),
                   'y': st.column_config.NumberColumn('y (m)', default=0.0),
                   'z': st.column_config.NumberColumn('z (m)', default=10.0, min_value=0.0)})
points = []
for _, row in point_table.dropna(subset=['Name', 'x']).iterrows():
    points.append(dict(name=str(row['Name']).strip(), x=float(row['x']),
                       y=float(row['y']) if dims >= 2 and not pd.isna(row['y']) else 0.0,
                       z=float(row['z']) if dims == 3 and not pd.isna(row['z']) else 0.0))
if not points:
    problems.append('Add at least one observation point.')
if len({q['name'] for q in points}) < len(points):
    problems.append('Give each observation point a different name.')
if any(q['name'] in ('Time', 'Source') for q in points):
    problems.append('An observation point cannot be named Time or Source.')
if not mass_source and any(q['x'] < 0 for q in points):
    problems.append('This source sits on the upstream edge of the aquifer, so observation points need x of zero or more.')
if dims == 3 and any(q['z'] < 0 for q in points):
    problems.append('Observation points cannot be above the water table (z below zero).')
if mass_source and any(q['x'] == 0 and q['y'] == 0 and q['z'] == geometry.get('depth', 0.0) for q in points):
    problems.append('An observation point cannot sit on a point source, where the concentration is not defined.')

col_t1, col_t2 = st.columns(2)
t_end = col_t1.number_input(f'Length of the calculation ({time_unit})', value=30.0 if time_unit == 'years' else 10000.0,
                            min_value=0.0, format='%g') * tf
snapshot_text = col_t2.text_input(f'Times for the profiles and plume maps ({time_unit})',
                                  value='2, 5, 10, 20' if time_unit == 'years' else '700, 1800, 3600, 7200',
                                  help='Up to five times, separated by commas.')
snapshots = []
try:
    snapshots = sorted({float(part) * tf for part in snapshot_text.replace(';', ',').split(',') if part.strip()})
except ValueError:
    problems.append('Enter the times as numbers separated by commas.')
if not snapshots or len(snapshots) > 5 or min(snapshots, default=1) <= 0:
    problems.append('Enter between one and five times, each greater than zero.')
if t_end <= 0:
    problems.append('The length of the calculation must be greater than zero.')
if ts.size and t_end > 0 and ts[np.argmax(cs > 0)] >= t_end:
    problems.append('The source starts after the end of the calculation.')

# set-up diagram
ready = not problems
if ready:
    figure = plots.setup(key, geometry, points, engine.extent(key, p, geometry, points, max(snapshots)), colours)
    plots.add_watermark(figure)
    st.pyplot(figure)
    plt.close(figure)
    st.caption('The source, the flow direction and the observation points. The axes are not drawn to '
               'the same scale.')

# %% 5. run

st.header('5. Run')
FIGURE_DPI = st.radio('Figure download resolution (dpi)', [300, 600], horizontal=True)
for problem in problems:
    st.info(problem)

SPREAD = 3.0        # standard deviations of sideways spreading shown on the plume maps


def matrix(values, columns, rows, corner):
    # a grid as a table: first column holds the row positions, the header holds the column positions
    table = pd.DataFrame(values, columns=[f'{value:g}' for value in columns])
    table.insert(0, corner, rows)
    return table


if st.button('Run', type='primary', disabled=not ready):
    try:
        with st.spinner('Calculating...'):
            times = np.unique(np.concatenate([np.linspace(0, t_end, N_TIMES), ts[ts < t_end]]))
            curves = engine.breakthrough(key, p, geometry, ts, cs, points, times)
            x0, x1, half_y, half_z = engine.extent(key, p, geometry, points, max(snapshots), SPREAD)
            xs = np.linspace(x0, x1, N_X)
            result = dict(solution=solution, key=key, dims=dims, mass_source=mass_source, p=dict(p),
                          geometry=dict(geometry), ts=ts, cs=cs, points=points, snapshots=snapshots,
                          t_end=t_end, time_unit=time_unit, tf=tf, curves=curves, xs=xs,
                          profiles=engine.profiles(key, p, geometry, ts, cs, xs, snapshots,
                                                   z=geometry.get('depth', 0.0)),
                          summary=engine.summary(key, p, ts, cs, points, curves),
                          derived=engine.derived(p, dims), source_column=source_column)
            if dims == 2:
                result['map_times'] = snapshots[:4]
                result['map_x'] = np.linspace(x0, x1, MAP_NX)
                result['map_y'] = np.linspace(-half_y, half_y, MAP_NY)
                result['grids'] = [engine.grid(key, p, geometry, ts, cs, result['map_x'], result['map_y'], t)
                                   for t in result['map_times']]
            source_unit = source_column[source_column.index('(') + 1:-1]
            settings = [('App', f'{APP_NAME} version {APP_VERSION}'), ('Run date', dt.date.today().isoformat()),
                        ('Solution', solution), ('Time unit', time_unit)]
            settings += list(entered.items())
            settings += [('Value used: groundwater velocity v (m/d)', f"{p['v']:g}"),
                         ('Value used: retardation factor R', f"{p['R']:g}"),
                         ('Value used: decay rate λ (per day)', f"{p['lam']:g}")]
            settings += [(f'Source from {start / tf:g} {time_unit}', f'{value:g} {source_unit}')
                         for start, value in zip(ts, cs)]
            settings += [(f"Observation point {q['name']}",
                          ', '.join(f'{axis} = {q[axis]:g} m' for axis in 'xyz'[:dims])) for q in points]
            settings += [(f'Length of the calculation ({time_unit})', f'{t_end / tf:g}'),
                         (f'Times for the profiles and plume maps ({time_unit})',
                          ', '.join(f'{t / tf:g}' for t in snapshots)),
                         ('Units', f'Distances in metres, times in {time_unit}, concentrations in mg/L')]
            result['settings'] = pd.DataFrame(settings, columns=['Setting', 'Value']).astype(str)
            st.session_state['result'] = result

        new_total = run_counter.increment_runs()
        if new_total is not None:
            st.session_state['run_count'] = new_total
    except Exception as error:
        st.error(f'Run failed: {error}')

# %% 6. results


def freeze(d):
    return tuple(sorted(d.items()))


@st.cache_data(show_spinner=False, max_entries=60)
def calc_grid(key, p, geometry, ts, cs, a, b, t, plane, offset):
    return engine.grid(key, dict(p), dict(geometry), np.array(ts), np.array(cs), np.array(a), np.array(b),
                       t, plane, offset)


@st.cache_data(show_spinner=False, max_entries=10)
def calc_volume(key, p, geometry, ts, cs, xs, ys, zs, t):
    Z, Y, X = np.meshgrid(np.array(zs), np.array(ys), np.array(xs), indexing='ij')
    return solutions.concentration(key, dict(p), dict(geometry), np.array(ts), np.array(cs), X, Y, Z, t)


FIGURES = {}


def show(fig, name):
    # render once, show it, keep the png for the zip
    plots.add_watermark(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    FIGURES[name] = buffer.getvalue()
    st.image(FIGURES[name], width='stretch')
    st.download_button(f'Download PNG ({FIGURE_DPI} dpi)', FIGURES[name], file_name=f'{name}.png',
                       mime='image/png', key=f'download_{name}', on_click='ignore')


if 'result' in st.session_state:
    r = st.session_state['result']
    rkey, rp, rg, rts, rcs, rpoints = r['key'], r['p'], r['geometry'], r['ts'], r['cs'], r['points']
    unit, f, rdims, rmass = r['time_unit'], r['tf'], r['dims'], r['mass_source']
    curves, xs = r['curves'], r['xs']
    prefix = APP_NAME
    args = (rkey, freeze(rp), freeze(rg), tuple(rts), tuple(rcs))

    st.header('6. Results')
    st.caption(f"Solution: {r['solution']}. Change any input and press Run again to update.")
    col1, col2, col3 = st.columns(3)
    col1.metric('Groundwater velocity', f"{rp['v'] * 365.25:,.3g} m/year")
    col2.metric('Solute velocity, v / R', f"{rp['v'] / rp['R'] * 365.25:,.3g} m/year")
    col3.metric('Retardation factor', f"{rp['R']:.3g}")
    workbook_slot = st.container()

    st.subheader('Concentration through time')
    source_label = 'Source\n(g/d)' if rmass else 'Source\n(mg/L)'
    show(plots.breakthrough(curves, rpoints, colours, unit, f, source_label), f'{prefix}_breakthrough')
    st.caption('Top: the source history. Bottom: the concentration at each observation point.')
    summary_shown = r['summary'].copy()
    for column in list(summary_shown.columns):
        if column.endswith('(d)'):
            summary_shown[column] = summary_shown[column] / f
            summary_shown = summary_shown.rename(columns={column: column[:-3] + f'({unit})'})
    st.dataframe(summary_shown, width='stretch', hide_index=True)
    st.caption('Travel time by flow alone is the time for water moving at the solute velocity to reach '
               'the point. Dispersion brings the first solute earlier than this.')

    st.subheader('Concentration along the flow')
    profile_shown = r['profiles']
    if rmass:
        # the line passes through the point source, where the concentration has no limit, so leave a gap there
        profile_shown = np.where((np.abs(xs) < 0.03 * (xs[-1] - xs[0]))[:, None], np.nan, profile_shown)
    show(plots.profile(xs, profile_shown, r['snapshots'], rpoints, colours, unit, f, rdims > 1, 0.0 if rmass else None),
         f'{prefix}_profile')
    if rmass:
        st.caption('Concentration on the line through the source (y = 0' + (', z = source depth' if rdims == 3 else '')
                   + '). The concentration rises without limit at a point source, so the lines stop short of it.')
    elif rdims > 1:
        st.caption('Concentration on the line through the centre of the source (y = 0'
                   + (', z = source depth).' if rdims == 3 else ').'))

    sheets = {}
    if rdims >= 2:
        st.subheader('Plume from above' if rdims == 2 else 'Plume slices')
        log_scale = st.checkbox('Logarithmic colour scale', value=rmass,
                                help='Shows low concentrations at the edge of the plume.')

    if rdims == 2:
        together = np.concatenate([g.ravel() for g in r['grids']])
        top = plots.nice_top(np.nanpercentile(together, 99.5) if rmass else np.nanmax(together), log_scale)
        lev = plots.levels(top, log_scale)
        show(plots.maps(rkey, rg, r['map_x'], r['map_y'], r['grids'], r['map_times'], rpoints, colours, unit, f,
                        lev, log_scale), f'{prefix}_plume')
        st.caption('The plume at each time, on one colour scale. White is below the lowest level on the '
                   'scale. The axes are not drawn to the same scale'
                   + ('. Only the first four times are mapped.' if len(r['snapshots']) > 4 else '.'))
        for t, g in zip(r['map_times'], r['grids']):
            sheets[f'Plume {plots.time_label(t, unit, f)}'] = matrix(g, r['map_x'], r['map_y'], 'y (m) down, x (m) across')

    if rdims == 3:
        snapshots_r = r['snapshots']
        labels = [plots.time_label(t, unit, f) for t in snapshots_r]
        chosen = snapshots_r[0]
        if len(labels) > 1:
            start = labels[snapshots_r.index(engine.last_active(rts, rcs, snapshots_r))]
            chosen = snapshots_r[labels.index(st.select_slider('Time', labels, value=start))]
        x0, x1, half_y, half_z = engine.extent(rkey, rp, rg, rpoints, chosen, SPREAD)
        st.caption('Three slices through the plume. Set where each slice is taken. The dashed lines mark '
                   'the other two slices. These update without pressing Run.')
        cut_columns = st.columns(3)
        cut = dict(z=cut_columns[0].number_input('Slice from above at depth z (m)', value=rg['depth'], min_value=0.0, format='%g'),
                   y=cut_columns[1].number_input('Slice from the side at y (m)', value=0.0, format='%g'),
                   x=cut_columns[2].number_input('Slice across the flow at x (m)', value=rpoints[0]['x'], format='%g'))
        map_x = np.linspace(x0, x1, MAP_NX)
        map_y = np.linspace(-half_y, half_y, MAP_NY)
        map_z = np.linspace(0, rg['depth'] + half_z, MAP_NZ)
        with st.spinner('Calculating the plume slices...'):
            plan = calc_grid(*args, tuple(map_x), tuple(map_y), chosen, 'xy', cut['z'])
            side = calc_grid(*args, tuple(map_x), tuple(map_z), chosen, 'xz', cut['y'])
            across = calc_grid(*args, tuple(map_y), tuple(map_z), chosen, 'yz', cut['x'])
        together = np.concatenate([plan.ravel(), side.ravel()])
        top = plots.nice_top(np.nanpercentile(together, 99.5) if rmass else np.nanmax(together), log_scale)
        lev = plots.levels(top, log_scale)
        show(plots.slices(rkey, rg, map_x, map_y, map_z, plan, side, across, cut, chosen, rpoints, colours,
                          unit, f, lev, log_scale), f'{prefix}_slices')
        st.caption('White is below the lowest level on the scale. The water table is at z = 0 and depth '
                   'increases downwards. The axes are not drawn to the same scale, so the plume looks '
                   'thicker than it is.')
        label = plots.time_label(chosen, unit, f)
        sheets[f'Above {label}'] = matrix(plan, map_x, map_y, 'y (m) down, x (m) across')
        sheets[f'Side {label}'] = matrix(side, map_x, map_z, 'z (m) down, x (m) across')
        sheets[f'Across {label}'] = matrix(across, map_y, map_z, 'z (m) down, y (m) across')

        if st.checkbox('Show an interactive 3D view', value=False,
                       help='Shells of equal concentration that can be rotated. Takes a few seconds.'):
            vol_x = np.linspace(x0, x1, 56)
            vol_y = np.linspace(-half_y, half_y, 29)
            vol_z = np.linspace(0, rg['depth'] + half_z, 23)
            with st.spinner('Calculating the 3D view...'):
                volume = calc_volume(*args, tuple(vol_x), tuple(vol_y), tuple(vol_z), chosen)
            shells = [fraction * top for fraction in (0.01, 0.1, 0.5)]
            if np.nanmax(volume) <= shells[0]:
                st.info('The concentration is too low at this time to draw the shells.')
            else:
                st.plotly_chart(plots.isosurfaces(vol_x, vol_y, vol_z, volume, shells, colours, label))
                st.caption('Each shell joins points of equal concentration: 1%, 10% and 50% of the top of '
                           'the colour scale above. Drag to rotate. The vertical scale is stretched.')

    with st.expander('Values worked out from the inputs'):
        st.dataframe(r['derived'], width='stretch', hide_index=True)

    # workbook: the time series first, the input settings second, then the other tables
    source_column_r = r['source_column']
    curves_out = curves.rename(columns={'Time': f'Time ({unit})',
                                        'Source': f'Source, {source_column_r[0].lower()}{source_column_r[1:]}'})
    curves_out[f'Time ({unit})'] = curves_out[f'Time ({unit})'] / f
    curves_out = curves_out.rename(columns={q['name']: f"{q['name']} (mg/L)" for q in rpoints})
    profile_out = pd.DataFrame(r['profiles'],
                               columns=[f'{plots.time_label(t, unit, f)} (mg/L)' for t in r['snapshots']])
    profile_out.insert(0, 'x (m)', xs)
    all_sheets = {'Time series': curves_out,
                  'Settings': r['settings'],
                  'Observation points': summary_shown,
                  'Concentration along the flow': profile_out,
                  'Derived values': r['derived']}
    all_sheets.update(sheets)
    xlsx = workbook.build(all_sheets)

    with workbook_slot:
        st.download_button('Download the workbook (.xlsx)', xlsx, file_name=f'{prefix}.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           type='primary', on_click='ignore')
        st.caption('The first sheet holds the concentration through time at each observation point. '
                   'The second holds the input settings.')
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f'{prefix}.xlsx', xlsx)
        for name, data in FIGURES.items():
            archive.writestr(f'{name}.png', data)
    st.download_button('Download everything (.zip)', zip_buffer.getvalue(),
                       file_name=f'{prefix}_outputs.zip', mime='application/zip', on_click='ignore')
