# checks for HydroTRANSPORT. No network needed. Run with: pytest tests/

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import erfc, k0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrotransport import engine, solutions  # noqa: E402

P = dict(v=0.5, n=0.3, al=2.0, ah=0.2, av=0.02, Dm=0.0, R=1.0, lam=0.0)
ON = ([0.0], [1.0])


def conc(key, p, geometry, history, x, y=0.0, z=0.0, t=1.0):
    return solutions.concentration(key, p, geometry, history[0], history[1], x, y, z, t)


# %% 1D against numerical Laplace inversion

# kind (1 constant concentration, 3 flux), x, t, v, alpha_L, lambda, R, C/C0.
# Reference values from inverting the Laplace-domain solutions with mpmath at 50 digits,
# by two methods (de Hoog and Talbot) that agreed to 13 figures
LAPLACE = [
    (1, 20, 60, 0.5, 2, 0, 1, 0.8745247384659413),
    (1, 20, 60, 0.5, 2, 0.01, 1, 0.6205708270228696),
    (1, 20, 60, 0.5, 2, 1e-06, 1, 0.8744941724640181),
    (1, 0.5, 1, 0.5, 0.5, 0.001, 1, 0.7135056587630398),
    (1, 10, 300, 0.5, 50, 0.05, 3, 0.5060979915092771),
    (1, 200, 5000, 0.5, 50, 0.001, 3, 0.3804037607551207),
    (1, 1, 300, 0.5, 2, 0.2, 2, 0.6504612675162824),
    (3, 20, 60, 0.5, 2, 0, 1, 0.8251706465981706),
    (3, 20, 60, 0.5, 2, 0.01, 1, 0.5725840540603879),
    (3, 20, 60, 0.5, 2, 1e-06, 1, 0.825139968730707),
    (3, 0.5, 1, 0.5, 0.5, 0.001, 1, 0.4225922721164562),
    (3, 10, 300, 0.5, 50, 0.05, 3, 0.11488845173606407),
    (3, 200, 5000, 0.5, 50, 0.001, 3, 0.30635599654891665),
    (3, 1, 300, 0.5, 2, 0.2, 2, 0.3496827131009987),
]


@pytest.mark.parametrize('kind, x, t, v, al, lam, R, expected', LAPLACE)
def test_1d_against_laplace_inversion(kind, x, t, v, al, lam, R, expected):
    p = dict(P, v=v, al=al, lam=lam, R=R)
    c = conc('inlet1' if kind == 1 else 'inlet3', p, {}, ON, x, t=t)
    assert c == pytest.approx(expected, rel=1e-9)


def test_ogata_banks():
    # the constant concentration inlet without decay is Ogata and Banks (1961)
    x, t, v, D = np.array([1.0, 10.0, 40.0]), 50.0, 0.5, 1.0
    expected = 0.5 * (erfc((x - v * t) / (2 * np.sqrt(D * t)))
                      + np.exp(v * x / D) * erfc((x + v * t) / (2 * np.sqrt(D * t))))
    assert conc('inlet1', P, {}, ON, x, t=t) == pytest.approx(expected, rel=1e-12)


def test_high_peclet_does_not_overflow():
    # x / alpha_L = 2000. exp(v x / D) overflows here unless it is combined with erfc
    p = dict(P, al=0.05)
    for key in ('inlet1', 'inlet3'):
        c = conc(key, p, {}, ON, 100.0, t=[150.0, 200.0, 250.0])
        assert np.all(np.isfinite(c))
        assert c[0] < 1e-10 and c[1] == pytest.approx(0.5, abs=0.01) and c[2] == pytest.approx(1.0, abs=1e-10)
    c = conc('point3', dict(p, ah=0.005, av=0.005), {}, ON, [100.0, 200.0], t=1e5)
    assert np.all(np.isfinite(c)) and np.all(c > 0)


# %% point sources against the integral of instantaneous releases

def point_by_quad(dx, dy, dz, t, p, thickness=None):
    # a continuous point source is a string of instantaneous releases. Each release spreads as
    # a Gaussian cloud, so the sum is an integral that is independent of the closed forms.
    # thickness is given for the 2D case
    R, n, lam = p['R'], p['n'], p['lam']
    v, Dx, Dy, Dz = p['v'] / R, p['al'] * p['v'] / R, p['ah'] * p['v'] / R, p['av'] * p['v'] / R

    def f(tau):
        cloud = np.exp(-(dx - v * tau) ** 2 / (4 * Dx * tau) - dy ** 2 / (4 * Dy * tau) - lam * tau)
        if thickness is not None:
            return cloud / (4 * np.pi * tau * np.sqrt(Dx * Dy) * thickness * n * R)
        return cloud * np.exp(-dz ** 2 / (4 * Dz * tau)) / (8 * (np.pi * tau) ** 1.5 * np.sqrt(Dx * Dy * Dz) * n * R)
    breaks = sorted({t * 1e-8, t * 1e-6, t * 1e-4, t * 1e-2, t * 0.1} | ({dx / v} if 0 < dx / v < t else set()))
    return quad(f, 0, t, limit=800, epsabs=0, epsrel=1e-12, points=breaks)[0]


@pytest.mark.parametrize('lam, R', [(0.0, 1.0), (0.002, 2.0)])
def test_point_sources_against_instantaneous_releases(lam, R):
    p = dict(P, lam=lam, R=R)
    for dx, dy, dz in [(0.5, 0.0, 0.0), (5.0, 1.0, 0.3), (20.0, 0.0, 0.0), (60.0, 4.0, 1.0), (-3.0, 0.5, 0.2)]:
        for t in (1.0, 30.0, 365.0, 3650.0):
            expected = point_by_quad(dx, dy, dz, t, p)
            if expected > 1e-13:
                assert conc('point3', p, {}, ON, dx, dy, dz, t=t) == pytest.approx(expected, rel=1e-8)
            expected = point_by_quad(dx, dy, 0.0, t, p, thickness=4.0)
            if expected > 1e-13:
                assert conc('point2', p, dict(thickness=4.0), ON, dx, dy, t=t) == pytest.approx(expected, rel=1e-8)


def test_point3_steady_state():
    # long after the start the 3D point source has a simple closed form
    v, Dx, Dy, Dz, n, lam = 0.5, 1.0, 0.1, 0.01, 0.3, 0.004
    beta = np.sqrt(v ** 2 + 4 * Dx * lam)
    for dx, dy, dz in [(1.0, 0.0, 0.0), (20.0, 2.0, 0.5), (-4.0, 1.0, 0.2)]:
        gamma = np.sqrt(dx ** 2 + Dx * dy ** 2 / Dy + Dx * dz ** 2 / Dz)
        expected = np.exp((v * dx - gamma * beta) / (2 * Dx)) / (4 * np.pi * n * gamma * np.sqrt(Dy * Dz))
        assert conc('point3', dict(P, lam=lam), {}, ON, dx, dy, dz, t=1e6) == pytest.approx(expected, rel=1e-12)


# %% the time integral

def test_point2_steady_state():
    # long after the start the 2D point source has a closed form with the Bessel function K0
    v, Dx, Dy, n = 0.5, 1.0, 0.1, 0.3
    for x, y in [(0.1, 0.0), (1.0, 0.0), (5.0, 1.0), (20.0, 0.0), (100.0, 5.0), (-5.0, 0.0)]:
        expected = (np.exp(v * x / (2 * Dx)) * k0(v / (2 * Dx) * np.sqrt(x ** 2 + Dx / Dy * y ** 2))
                    / (2 * np.pi * n * np.sqrt(Dx * Dy)))
        assert conc('point2', P, dict(thickness=1.0), ON, x, y, t=1e6) == pytest.approx(expected, rel=1e-10)


def strip_by_quad(x, y, t, width, lam=0.0, v=0.5, Dx=1.0, Dy=0.1, z=0.0, height=None, Dz=0.01):
    # Wexler (1992) equation 91b by adaptive quadrature, or 121b if a height is given
    def f(tau):
        value = (x / (4 * np.sqrt(np.pi * Dx)) * tau ** -1.5 * np.exp(-(x - v * tau) ** 2 / (4 * Dx * tau) - lam * tau)
                 * (erfc((-width / 2 - y) / (2 * np.sqrt(Dy * tau))) - erfc((width / 2 - y) / (2 * np.sqrt(Dy * tau)))))
        if height is not None:
            value *= 0.5 * (erfc((-height / 2 - z) / (2 * np.sqrt(Dz * tau))) - erfc((height / 2 - z) / (2 * np.sqrt(Dz * tau))))
        return value
    breaks = sorted({t * 1e-8, t * 1e-6, t * 1e-4, t * 1e-2, t * 0.1, min(x * x / (6 * Dx), t * 0.99)}
                    | ({x / v} if 0 < x / v < t else set()))
    return quad(f, 0, t, limit=800, epsabs=0, epsrel=1e-12, points=breaks)[0]


@pytest.mark.parametrize('x, y', [(0.01, 0.0), (1.0, 0.0), (5.0, 4.9), (5.0, 5.1), (20.0, 0.0), (100.0, 8.0)])
@pytest.mark.parametrize('lam', [0.0, 0.01])
def test_strip_against_adaptive_quadrature(x, y, lam):
    # from early times to long after steady state, including close to the source
    for t in (0.01, 1.0, 30.0, 365.0, 36500.0):
        expected = strip_by_quad(x, y, t, 10.0, lam)
        if expected > 1e-13:
            assert conc('strip', dict(P, lam=lam), dict(width=10.0), ON, x, y, t=t) == pytest.approx(expected, rel=1e-9)


def test_patch_against_adaptive_quadrature():
    for x, y, z in [(0.01, 0.0, 0.0), (1.0, 0.0, 1.9), (5.0, 4.0, 2.1), (20.0, 0.0, 0.0), (100.0, 8.0, 3.0)]:
        for t in (1.0, 30.0, 365.0, 36500.0):
            expected = strip_by_quad(x, y, t, 10.0, 0.01, z=z, height=4.0)
            if expected > 1e-13:
                c = conc('patch', dict(P, lam=0.01), dict(width=10.0, height=4.0), ON, x, y, z, t=t)
                assert c == pytest.approx(expected, rel=1e-9)


def test_step_changes_match_superposition():
    # the source history inside the integral gives the same answer as adding step responses
    history = ([0.0, 50.0, 120.0], [100.0, 0.0, 40.0])
    for x, y, t in [(5, 0, 49), (5, 0, 51), (5, 0, 60), (5, 0, 130), (5, 0, 1000), (40, 3, 200), (0.05, 0, 50.5),
                    (0.05, 0, 5000)]:
        expected = (100 * strip_by_quad(x, y, t, 10.0) - (100 * strip_by_quad(x, y, t - 50, 10.0) if t > 50 else 0)
                    + (40 * strip_by_quad(x, y, t - 120, 10.0) if t > 120 else 0))
        assert conc('strip', P, dict(width=10.0), history, float(x), float(y), t=float(t)) == pytest.approx(expected, abs=1e-9)


def test_integral_matches_closed_form_in_1d():
    # the 1D constant concentration inlet written as a time integral, against its closed form
    ts, cs = np.array([0.0, 50.0, 120.0, 300.0]), np.array([100.0, 0.0, 40.0, 0.0])
    x, t = np.meshgrid([0.01, 1.0, 20.0, 150.0], [10.0, 50.001, 60.0, 121.0, 299.0, 301.0, 400.0, 2000.0])
    for lam in (0.0, 0.01):
        integral = solutions._first_type_1d(x.ravel(), t.ravel(), ts, cs, 0.5, 1.0, lam)
        closed = solutions.concentration('inlet1', dict(P, lam=lam), {}, ts, cs, x, t=t).ravel()
        assert integral == pytest.approx(closed, abs=1e-10)


def test_flux_inlet_integral_matches_closed_form():
    # with decay the flux inlet uses the integral. A negligible decay rate must match the closed form
    history = ([0.0, 100.0], [1.0, 0.0])
    x, t = np.meshgrid([0.0, 1e-6, 0.5, 10.0, 200.0], [50.0, 150.0, 900.0])
    closed = conc('inlet3', P, {}, history, x, t=t)
    integral = conc('inlet3', dict(P, lam=1e-15), {}, history, x, t=t)
    assert integral == pytest.approx(closed, abs=1e-10)


# %% limits and physical checks

def test_limits_between_solutions():
    x, y, z = np.array([1.0, 20.0, 100.0]), np.array([0.0, 3.0, 8.0]), np.array([0.0, 5.0, -3.0])
    # a very tall patch is a strip
    assert conc('patch', P, dict(width=10.0, height=1e7), ON, x, y, z, t=365.0) == pytest.approx(
        conc('strip', P, dict(width=10.0), ON, x, y, t=365.0), rel=1e-10)
    # a very wide strip is the 1D inlet
    assert conc('strip', P, dict(width=1e7), ON, x, 0.0, t=100.0) == pytest.approx(
        conc('inlet1', P, {}, ON, x, t=100.0), rel=1e-10)


def test_flux_inlet_conserves_mass():
    # with no decay, the solute in the column equals what has flowed in: integral of C dx = v / R x sum of c0 dt
    history = ([0.0, 40.0, 90.0], [100.0, 0.0, 30.0])
    for R in (1.0, 2.5):
        p = dict(P, R=R)
        xs = np.linspace(0, 600, 60001)
        c = conc('inlet3', p, {}, history, xs, t=150.0)
        stored = np.sum((c[1:] + c[:-1]) / 2 * np.diff(xs))
        assert stored == pytest.approx(p['v'] / R * (100 * 40 + 30 * 60), rel=1e-6)


def test_source_switched_off():
    history = ([0.0, 100.0], [100.0, 0.0])
    geometry = dict(width=10.0, height=4.0, thickness=5.0)
    for key in ('inlet1', 'inlet3', 'point2', 'strip', 'point3', 'patch'):
        c = conc(key, P, geometry, history, 20.0, 1.0, 0.5, t=np.array([-5.0, 0.0, 90.0, 5000.0]))
        assert c[0] == 0 and c[1] == 0          # nothing before the source starts
        assert c[2] > 0.01                      # the plume has arrived
        assert 0 <= c[3] < 1e-9 * c[2]          # and has cleared long after the source stops
    # on the source plane the constant concentration sources return the source value
    assert conc('strip', P, geometry, history, 0.0, 2.0, t=[50.0, 150.0]) == pytest.approx([100.0, 0.0])
    assert conc('strip', P, geometry, history, 0.0, 20.0, t=50.0) == 0
    assert conc('inlet1', P, geometry, history, 0.0, t=[50.0, 150.0]) == pytest.approx([100.0, 0.0])


def test_water_table_is_a_barrier():
    history = ([0.0, 200.0], [1.0, 0.3])
    for key, geometry in (('point3', dict(depth=3.0)), ('patch', dict(width=10.0, height=4.0, depth=3.0))):
        # no vertical gradient at the water table, so no solute crosses it: the change in
        # concentration just below it grows with the square of the depth, not in proportion to it
        near = conc(key, P, geometry, history, 30.0, 2.0, np.array([0.0, 1e-3, 1e-2]), t=400.0)
        assert (near[2] - near[0]) / (near[1] - near[0]) == pytest.approx(100, rel=0.01)
        # it is the sum of the source and its mirror image in an aquifer with no top
        free = dict(geometry, depth=None)
        both = conc(key, P, free, history, 30.0, 2.0, 5.0 - 3.0, t=400.0) + conc(key, P, free, history, 30.0, 2.0, 5.0 + 3.0, t=400.0)
        assert conc(key, P, geometry, history, 30.0, 2.0, 5.0, t=400.0) == pytest.approx(both, rel=1e-10)
        # far below the water table the barrier makes no difference, and above it there is no aquifer
        deep = dict(geometry, depth=500.0)
        assert conc(key, P, deep, history, 30.0, 2.0, 501.0, t=400.0) == pytest.approx(
            conc(key, P, free, history, 30.0, 2.0, 1.0, t=400.0), rel=1e-10)
        assert np.isnan(conc(key, P, geometry, history, 30.0, 2.0, -1.0, t=400.0))
    # a point source on the water table sends all its solute downwards: twice the concentration
    assert conc('point3', P, dict(depth=0.0), ON, 30.0, 2.0, 1.0, t=400.0) == pytest.approx(
        2 * conc('point3', P, {}, ON, 30.0, 2.0, 1.0, t=400.0), rel=1e-12)


def test_retardation_delays_arrival():
    # with R = 2 the plume at time 2t matches the unretarded plume at time t (constant concentration sources)
    xs = np.array([5.0, 30.0, 80.0])
    assert conc('inlet1', dict(P, R=2.0), {}, ON, xs, t=200.0) == pytest.approx(conc('inlet1', P, {}, ON, xs, t=100.0), rel=1e-12)
    assert conc('strip', dict(P, R=2.0), dict(width=10.0), ON, xs, 2.0, t=200.0) == pytest.approx(
        conc('strip', P, dict(width=10.0), ON, xs, 2.0, t=100.0), rel=1e-10)


def test_point_source_undefined_at_source():
    assert np.isnan(conc('point3', P, {}, ON, 0.0, 0.0, 0.0, t=100.0))
    assert np.isnan(conc('point2', P, dict(thickness=1.0), ON, 0.0, 0.0, t=100.0))
    assert np.isnan(conc('strip', P, dict(width=10.0), ON, -1.0, 0.0, t=100.0))


# %% engine and app

def test_source_history_checks():
    table = pd.DataFrame({'Start': [5.0, 0.0], 'Value': [0.0, 100.0]})
    ts, cs, problems = engine.source_history(table)
    assert list(ts) == [0.0, 5.0] and list(cs) == [100.0, 0.0] and not problems
    assert engine.source_history(pd.DataFrame({'Start': [0.0, 0.0], 'Value': [1.0, 2.0]}))[2]
    assert engine.source_history(pd.DataFrame({'Start': [0.0], 'Value': [0.0]}))[2]
    assert engine.source_history(pd.DataFrame({'Start': [0.0, None], 'Value': [1.0, 2.0]}))[2]


def test_breakthrough_and_summary():
    points = [dict(name='A', x=20.0, y=0.0, z=0.0), dict(name='B', x=60.0, y=0.0, z=0.0)]
    ts, cs = np.array([0.0, 100.0]), np.array([100.0, 0.0])
    curves = engine.breakthrough('inlet1', P, {}, ts, cs, points, np.linspace(0, 600, 601))
    table = engine.summary('inlet1', P, ts, cs, points, curves)
    assert table['Travel time by flow alone, x R / v (d)'].tolist() == [40.0, 120.0]
    assert table['Peak concentration (mg/L)'].iloc[0] > table['Peak concentration (mg/L)'].iloc[1] > 0
    assert engine.last_active(ts, cs, [50.0, 100.0, 200.0]) == 100.0


@pytest.mark.parametrize('name', list(solutions.SOLUTIONS))
def test_app_runs(name):
    testing = pytest.importorskip('streamlit.testing.v1')
    app = testing.AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180)
    app.run()
    app.selectbox[0].set_value(name).run()
    assert not app.exception and not app.info
    assert not app.get('download_button')      # nothing is calculated until Run is pressed
    [button for button in app.button if button.label == 'Run'][0].click().run()
    assert not app.exception
    assert any(button.label.startswith('Download the workbook') for button in app.get('download_button'))
