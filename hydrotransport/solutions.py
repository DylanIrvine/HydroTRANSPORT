# analytical solutions for solute transport in uniform flow, with a source that changes in steps.
# Equations from Wexler (1992). Metres, days and g/m³ (the same as mg/L).
#
# The closed-form solutions (1D inlets, 3D point source) are written with the scaled
# complementary error function so that exp() cannot overflow at high Peclet numbers.
# The others are a time integral, C = integral of c(t - tau) g(tau) dtau, where g is the
# response to a unit pulse at the source. The integral runs over the whole source history,
# so a step change needs no subtraction of two nearly equal numbers.

import numpy as np
from scipy.special import erfc, erfcx

SOLUTIONS = {
    '1D, constant concentration at the inlet': dict(
        key='inlet1', dims=1, source='concentration',
        summary='Flow along a column or a flow path, with the concentration held at a set value '
                'where water enters (x = 0). This is the Ogata and Banks (1961) case, with '
                'sorption and decay added.',
        when='Use it for column experiments and as the simplest picture of a plume front. The '
             'source pins the concentration at x = 0, so with a low velocity and a high '
             'dispersion it adds mass by dispersion as well as by flow.'),
    '1D, flux at the inlet': dict(
        key='inlet3', dims=1, source='concentration',
        summary='Flow along a column or a flow path, where water of a set concentration flows '
                'in at x = 0. The concentration just inside the inlet builds up over time '
                'instead of being fixed.',
        when='Use it when the inflowing water is known, not the concentration in the aquifer. '
             'It conserves mass at the inlet. It differs from the constant concentration case '
             'only close to the inlet or when dispersion is large compared with flow.'),
    '2D, point source': dict(
        key='point2', dims=2, source='mass',
        summary='A source at one point that releases solute at a set mass rate, mixed over the '
                'full thickness of the aquifer. The plume spreads along and across the flow.',
        when='Use it for a leak or an injection bore that is small compared with the plume, in '
             'an aquifer thin enough for the plume to fill its thickness. The concentration is '
             'not defined at the source itself.'),
    '2D, line source across the flow': dict(
        key='strip', dims=2, source='concentration',
        summary='A line of set width across the flow at x = 0, held at a set concentration, '
                'mixed over the full thickness of the aquifer. The plume spreads along and '
                'across the flow.',
        when='Use it for a source with a known width and concentration, such as a trench, a '
             'landfill edge or a contaminated zone that the flow passes through.'),
    '3D, point source': dict(
        key='point3', dims=3, source='mass',
        summary='A source at one point that releases solute at a set mass rate. The plume '
                'spreads along the flow, sideways and vertically.',
        when='Use it for a small source in a thick aquifer. The aquifer has no top or base in '
             'this solution, so it suits a plume that has not reached either. The concentration '
             'is not defined at the source itself.'),
    '3D, patch source across the flow': dict(
        key='patch', dims=3, source='concentration',
        summary='A rectangle of set width and height across the flow at x = 0, held at a set '
                'concentration. The plume spreads along the flow, sideways and vertically.',
        when='Use it for a source zone of known size in a thick aquifer. The aquifer has no top '
             'or base in this solution, so it suits a plume that has not reached either.'),
}
KEYS = {name: info['key'] for name, info in SOLUTIONS.items()}

# quadrature settings for the time integral
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(12)
LOG_PANELS = 64                    # panels spread evenly in ln(tau)
PEAK_STEPS = np.arange(-8.0, 8.5, 1.0)   # extra panel edges around the advective arrival, in standard deviations
LOWEST = 1e-14                     # lower limit of the integral as a fraction of t
CUT = 3000.0                       # the pulse response is zero where (distance²/D)/tau exceeds this
CHUNK = 3000                       # points handled at once, to limit memory


def retarded(p):
    # velocity and dispersion coefficients divided by the retardation factor
    R = p['R']
    v = p['v'] / R
    Dx = (p['al'] * p['v'] + p['Dm']) / R
    Dy = (p['ah'] * p['v'] + p['Dm']) / R
    Dz = (p['av'] * p['v'] + p['Dm']) / R
    return v, Dx, Dy, Dz


def source_value(ts, cs, t):
    # the source at time t. ts are the start times of each stage, cs the values. Zero before ts[0]
    ts, cs = np.asarray(ts, float), np.asarray(cs, float)
    index = np.searchsorted(ts, np.asarray(t, float), side='right') - 1
    return np.where(index >= 0, cs[np.clip(index, 0, None)], 0.0)


# %% closed forms: response to a source of 1 switched on at t = 0

def _exp_erfc(a, b):
    # exp(a) * erfc(b) without overflow. For b > 0 it is exp(a - b²) * erfcx(b)
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.empty(a.shape)
    pos = b > 0
    out[pos] = np.exp(a[pos] - b[pos] ** 2) * erfcx(b[pos])
    out[~pos] = np.exp(a[~pos]) * erfc(b[~pos])
    return out


def step_inlet1(x, t, v, D, lam):
    # Wexler (1992) equation 60. Constant concentration at x = 0
    u = np.sqrt(v ** 2 + 4 * lam * D)
    s = 2 * np.sqrt(D * t)
    return 0.5 * (np.exp(x * (v - u) / (2 * D)) * erfc((x - u * t) / s)
                  + _exp_erfc(x * (v + u) / (2 * D), (x + u * t) / s))


def step_inlet3(x, t, v, D):
    # Wexler (1992) equation 68. Water of concentration 1 flows in at x = 0, no decay.
    # With decay, equation 67 subtracts two very large terms, so _inlet3 uses the time integral
    s = 2 * np.sqrt(D * t)
    return (0.5 * erfc((x - v * t) / s)
            + np.sqrt(v ** 2 * t / (np.pi * D)) * np.exp(-(x - v * t) ** 2 / (4 * D * t))
            - 0.5 * (1 + v * x / D + v ** 2 * t / D) * _exp_erfc(v * x / D, (x + v * t) / s))


def step_point3(dx, dy, dz, t, v, Dx, Dy, Dz, lam, n, R):
    # Wexler (1992) equation 105, for a mass rate of 1. dx, dy, dz are measured from the source
    beta = np.sqrt(v ** 2 + 4 * Dx * lam)
    gamma = np.sqrt(dx ** 2 + Dx * dy ** 2 / Dy + Dx * dz ** 2 / Dz)
    with np.errstate(divide='ignore', invalid='ignore'):
        front = 1 / (8 * np.pi * n * R * gamma * np.sqrt(Dy * Dz))
        e = (v * dx - gamma * beta) / (2 * Dx)
        s = 2 * np.sqrt(Dx * t)
        value = front * (np.exp(e) * erfc((gamma - beta * t) / s)
                         + np.exp(e - (gamma - beta * t) ** 2 / (4 * Dx * t)) * erfcx((gamma + beta * t) / s))
    return np.where(gamma > 0, value, np.nan)


def _superpose(step, ts, cs, t, *args):
    # sum of step responses, one for each change in the source
    total = np.zeros(np.shape(t))
    previous = 0.0
    for start, value in zip(ts, cs):
        change, previous = value - previous, value
        on = t > start
        if change != 0 and on.any():
            total[on] += change * step(*[a[on] for a in args], t[on] - start)
    return total


# %% time integrals

def _integrate(kernel, t, s2, peak, sigma, ts, cs):
    # integral over tau from 0 to t of c(t - tau) * g(tau), for each point.
    # kernel(tau, rows) returns tau * g(tau), because the integral is taken in ln(tau).
    # s2 is distance²/D, which sets where the pulse response starts to rise.
    # peak and sigma locate the advective arrival, where the response can be sharp.
    # The panels are evenly spaced in ln(tau), with extra edges around the arrival and at
    # each change in the source, and 12-point Gauss-Legendre is used on each panel
    ts, cs = np.asarray(ts, float), np.asarray(cs, float)
    out = np.zeros(t.shape)
    fractions = np.linspace(0, 1, LOG_PANELS + 1)
    for first in range(0, t.size, CHUNK):
        rows = slice(first, first + CHUNK)
        tt = t[rows]
        lo = np.minimum(np.maximum(tt * LOWEST, s2[rows] / CUT), tt)
        edges = [np.exp(np.log(lo)[:, None] + np.log(tt / lo)[:, None] * fractions),
                 peak[rows][:, None] + sigma[rows][:, None] * PEAK_STEPS,
                 tt[:, None] - ts[None, :]]
        edges = np.sort(np.clip(np.hstack(edges), lo[:, None], tt[:, None]), axis=1)
        left, right = np.log(edges[:, :-1]), np.log(edges[:, 1:])
        half = (right - left) / 2
        tau = np.exp((left + right)[:, :, None] / 2 + half[:, :, None] * GL_NODES)
        source = source_value(ts, cs, tt[:, None, None] - tau)
        out[rows] = np.sum(source * kernel(tau, rows) * half[:, :, None] * GL_WEIGHTS, axis=(1, 2))
    return out


def _arrival(dx, v, Dx):
    # time and spread of the advective arrival at a distance dx downstream
    peak = np.where(dx > 0, dx / v, 0.0)
    sigma = np.sqrt(2 * Dx * peak) / v
    return peak, sigma


def _too_close(s2, t):
    # points so close to the source that the integral cannot be started
    return s2 / CUT < t * LOWEST


def _point2(dx, dy, t, ts, cs, v, Dx, Dy, lam, n, R, thickness):
    # Wexler (1992) equation 76, with the mass rate spread over the aquifer thickness
    front = 1 / (4 * np.pi * n * R * np.sqrt(Dx * Dy) * thickness)
    s2 = dx ** 2 / Dx + dy ** 2 / Dy

    def kernel(tau, rows):
        a, b = dx[rows][:, None, None], dy[rows][:, None, None]
        return np.exp(-(a - v * tau) ** 2 / (4 * Dx * tau) - b ** 2 / (4 * Dy * tau) - lam * tau)

    c = front * _integrate(kernel, t, s2, *_arrival(dx, v, Dx), ts, cs)
    return np.where(_too_close(s2, t), np.nan, c)


def _across(lower, upper, D, tau):
    # share of a pulse from a source between lower and upper that reaches this offset
    s = 2 * np.sqrt(D * tau)
    return erfc(lower / s) - erfc(upper / s)


def _strip(x, y, z, t, ts, cs, v, Dx, Dy, Dz, lam, width, height):
    # Wexler (1992) equation 91b (strip, height None) and 121b (patch). Source centred on y = 0, z = 0
    s2 = x ** 2 / Dx
    patch = height is not None

    def kernel(tau, rows):
        a, b = x[rows][:, None, None], y[rows][:, None, None]
        g = (a / (4 * np.sqrt(np.pi * Dx * tau)) * np.exp(-(a - v * tau) ** 2 / (4 * Dx * tau) - lam * tau)
             * _across(-width / 2 - b, width / 2 - b, Dy, tau))
        if patch:
            g = g * 0.5 * _across(-height / 2 - z[rows][:, None, None], height / 2 - z[rows][:, None, None], Dz, tau)
        return g

    c = _integrate(kernel, t, s2, *_arrival(x, v, Dx), ts, cs)
    # on the source plane the concentration is the source itself, inside the source
    inside = np.abs(y) <= width / 2
    if patch:
        inside = inside & (np.abs(z) <= height / 2)
    c = np.where(_too_close(s2, t), np.where(inside, source_value(ts, cs, t), 0.0), c)
    return np.where(x < 0, np.nan, c)


def _first_type_1d(x, t, ts, cs, v, D, lam):
    # the 1D constant concentration inlet as a time integral. Used only to check _integrate
    def kernel(tau, rows):
        a = x[rows][:, None, None]
        return a / (2 * np.sqrt(np.pi * D * tau)) * np.exp(-(a - v * tau) ** 2 / (4 * D * tau) - lam * tau)
    return _integrate(kernel, t, x ** 2 / D, *_arrival(x, v, D), ts, cs)


def _inlet3(x, t, ts, cs, v, D, lam):
    # the 1D flux inlet. Closed form without decay, the time integral with decay
    if lam == 0:
        return _superpose(lambda a, time: step_inlet3(a, time, v, D), ts, cs, t, x)

    def kernel(tau, rows):
        a = x[rows][:, None, None]
        return tau * v * np.exp(-lam * tau) * (
            np.exp(-(a - v * tau) ** 2 / (4 * D * tau)) / np.sqrt(np.pi * D * tau)
            - v / (2 * D) * _exp_erfc(v * a / D, (a + v * tau) / (2 * np.sqrt(D * tau))))
    c = _integrate(kernel, t, x ** 2 / D, *_arrival(x, v, D), ts, cs)
    # at and very near the inlet the response behaves like 1 / sqrt(tau), so the part of the
    # integral below its lower limit is added in closed form
    lo = np.minimum(np.maximum(t * LOWEST, x ** 2 / D / CUT), t)
    a = x / (2 * np.sqrt(D * lo))
    tail = v * (2 * np.sqrt(lo / (np.pi * D)) * np.exp(-a ** 2) - x / D * erfc(a))
    return c + source_value(ts, cs, t) * tail


# %% one entry point

def concentration(key, p, geometry, ts, cs, x, y=0.0, z=0.0, t=1.0):
    # concentration (g/m³) at x, y, z and time t, which are broadcast against each other.
    # p holds v, n, al, ah, av, Dm, R and lam. geometry holds width, height and thickness as needed.
    # ts and cs are the source history: start time of each stage and its value, which is a
    # concentration (g/m³) or, for the point sources, a mass rate (g/d).
    # The source sits at the origin and flow is in the +x direction
    x, y, z, t = np.broadcast_arrays(*[np.asarray(a, float) for a in (x, y, z, t)])
    shape = x.shape
    x, y, z, t = [a.ravel().copy() for a in (x, y, z, t)]
    ts, cs = np.asarray(ts, float), np.asarray(cs, float)
    v, Dx, Dy, Dz = retarded(p)
    lam, n, R = p['lam'], p['n'], p['R']

    out = np.zeros(x.shape)
    live = t > ts[0]
    if live.any():
        xl, yl, zl, tl = x[live], y[live], z[live], t[live]
        if key == 'inlet1':
            c = _superpose(lambda a, time: step_inlet1(a, time, v, Dx, lam), ts, cs, tl, np.maximum(xl, 0))
        elif key == 'inlet3':
            c = _inlet3(np.maximum(xl, 0), tl, ts, cs, v, Dx, lam)
        elif key == 'point3':
            c = _superpose(lambda a, b, d, time: step_point3(a, b, d, time, v, Dx, Dy, Dz, lam, n, R),
                           ts, cs, tl, xl, yl, zl)
        elif key == 'point2':
            c = _point2(xl, yl, tl, ts, cs, v, Dx, Dy, lam, n, R, geometry['thickness'])
        elif key == 'strip':
            c = _strip(xl, yl, zl, tl, ts, cs, v, Dx, Dy, Dz, lam, geometry['width'], None)
        elif key == 'patch':
            c = _strip(xl, yl, zl, tl, ts, cs, v, Dx, Dy, Dz, lam, geometry['width'], geometry['height'])
        else:
            raise ValueError(f'unknown solution {key}')
        out[live] = c
    if key in ('inlet1', 'inlet3'):
        out = np.where(x < 0, np.nan, out)
    # rounding can leave values a hair below zero after a step down
    out = np.where(out < 0, 0.0, out)
    return out.reshape(shape)
