"""
IPOPT-based NLP solver for hydropower scheduling.

This module provides an NLP formulation that fixes integer variables (pump/turbine/idle modes)
from MIQP warm-starts and optimizes continuous variables using true nonlinear constraints
via IPOPT. This serves as a benchmark to evaluate the quality of DFL's Taylor linearization.

Uses cyipopt (Python wrapper for IPOPT) via the scipy-compatible minimize_ipopt interface.
"""

import numpy as np
import torch

try:
    from cyipopt import minimize_ipopt
    CYIPOPT_AVAILABLE = True
except ImportError:
    CYIPOPT_AVAILABLE = False

from .parameters import HydroParameters


# Precomputed bivariate polynomial exponent pairs for degree 1..5
# Matches sklearn PolynomialFeatures ordering from preprocessing.py:
#   for total_degree in range(1, 6):
#       for a in range(total_degree, -1, -1): b = total_degree - a
_POLY_EXPONENTS = []
for _deg in range(1, 6):
    for _a in range(_deg, -1, -1):
        _POLY_EXPONENTS.append((_a, _deg - _a))
# _POLY_EXPONENTS has 20 pairs


def check_ipopt_available():
    """Check if IPOPT solver is available via cyipopt."""
    return CYIPOPT_AVAILABLE


def _to_numpy_flat(val):
    """Convert tensor or array to flat numpy array of floats."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().flatten().astype(float)
    return np.array(val, dtype=float).flatten()


def _extract_poly_coefficients(predict_q_poly):
    """
    Extract polynomial coefficients from the predict_q_poly closure.

    Returns numpy float arrays for turbine and pump polynomials.
    """
    defaults = predict_q_poly.__defaults__
    coefs_tur    = _to_numpy_flat(defaults[0])
    intercept_tur = float(_to_numpy_flat(defaults[1])[0])
    coefs_pump   = _to_numpy_flat(defaults[2])
    intercept_pump = float(_to_numpy_flat(defaults[3])[0])
    return coefs_tur, intercept_tur, coefs_pump, intercept_pump


# ---------------------------------------------------------------------------
# Polynomial helpers (pure numpy, analytical derivatives)
# ---------------------------------------------------------------------------

def _poly_q(p, h, coefs, intercept):
    """Evaluate bivariate polynomial: q = coefs @ features(p,h) + intercept."""
    val = intercept
    for c, (a, b) in zip(coefs, _POLY_EXPONENTS):
        val += c * (p ** a) * (h ** b)
    return val


def _poly_q_dp(p, h, coefs):
    """dq/dp of bivariate polynomial."""
    val = 0.0
    for c, (a, b) in zip(coefs, _POLY_EXPONENTS):
        if a > 0:
            val += c * a * (p ** (a - 1)) * (h ** b)
    return val


def _poly_q_dh(p, h, coefs):
    """dq/dh of bivariate polynomial."""
    val = 0.0
    for c, (a, b) in zip(coefs, _POLY_EXPONENTS):
        if b > 0:
            val += c * (p ** a) * b * (h ** (b - 1))
    return val


def _poly_v(h, coeffs):
    """v_low = polynomial(h), coeffs highest-degree first (numpy polyval)."""
    return np.polyval(coeffs, h)


def _poly_v_dh(h, coeffs):
    """dv/dh using numpy polyder."""
    return np.polyval(np.polyder(coeffs), h)


# ---------------------------------------------------------------------------
# NLP problem builder
# ---------------------------------------------------------------------------

class IPOPTHydroSolver:
    """
    Solves the fixed-mode continuous NLP for hydropower scheduling using IPOPT.

    Variables: x = [p_0..23, q_0..23, h_0..23, v_0..23]  (96 variables)
    Indices:   p[t]=x[t], q[t]=x[24+t], h[t]=x[48+t], v[t]=x[72+t]

    Objective: maximize sum_t(price[t]*p[t] - 0.4*p[t]^2)
               ↔ minimize -sum_t(price[t]*p[t] - 0.4*p[t]^2)

    Constraints:
      - Idle:    p[t]=0, q[t]=0  (via variable bounds)
      - Turbine: power linear bounds (ineq) + nonlinear flow eq
      - Pump:    power linear bounds (ineq) + nonlinear flow eq
      - All:     v[t] = poly_v(h[t])  (nonlinear eq)
      - All:     volume balance  (linear eq)
      - Terminal: v[23] <= target_vol_low  (linear ineq)
    """

    TH = 24  # time horizon (hours)

    def __init__(self, params, preprocess_data,
                 ipopt_max_iter=10000, ipopt_tol=1e-6, ipopt_time_limit=300):
        self.params = params
        self.ipopt_max_iter = ipopt_max_iter
        self.ipopt_tol = ipopt_tol
        self.ipopt_time_limit = ipopt_time_limit

        self.coefs_tur, self.intercept_tur, self.coefs_pump, self.intercept_pump = \
            _extract_poly_coefficients(preprocess_data['predict_q_poly'])

        self.h_v_coeffs = _to_numpy_flat(preprocess_data['h_v_coeffs'])
        self.h_v_deriv  = np.polyder(self.h_v_coeffs)

        self.pos_min_fit = _to_numpy_flat(params.pos_min_fit)  # [slope, intercept]
        self.pos_max_fit = _to_numpy_flat(params.pos_max_fit)
        self.neg_min_fit = _to_numpy_flat(params.neg_min_fit)
        self.neg_max_fit = _to_numpy_flat(params.neg_max_fit)

        self.head_min       = float(params.head_min)
        self.head_max       = float(params.head_max)
        self.v_low_init     = float(params.v_low_init)
        self.target_vol_low = float(params.target_vol_low)
        self.op_cost        = float(params.operational_cost)

    # ------------------------------------------------------------------
    # Objective and gradient
    # ------------------------------------------------------------------

    def _objective(self, x, prices):
        """Minimize negative profit (cyipopt minimizes)."""
        p = x[:self.TH]
        return float(-np.sum(prices * p - self.op_cost * p ** 2))

    def _gradient(self, x, prices):
        p = x[:self.TH]
        grad = np.zeros(4 * self.TH)
        grad[:self.TH] = -prices + 2.0 * self.op_cost * p
        return grad

    # ------------------------------------------------------------------
    # Constraint builders
    # ------------------------------------------------------------------

    def _build_constraints(self, mode_list):
        """
        Build list of constraint dicts for cyipopt.minimize_ipopt.

        Each dict: {'type': 'eq'|'ineq', 'fun': f(x)->scalar, 'jac': f(x)->vector}
        'ineq' means fun(x) >= 0.
        """
        TH = self.TH
        n  = 4 * TH
        constraints = []

        for t, mode in enumerate(mode_list):
            if mode == 'turbine':
                slope_lo, int_lo = self.pos_min_fit[0], self.pos_min_fit[1]
                slope_hi, int_hi = self.pos_max_fit[0], self.pos_max_fit[1]
                coefs, intercept  = self.coefs_tur, self.intercept_tur
            elif mode == 'pump':
                slope_lo, int_lo = self.neg_min_fit[0], self.neg_min_fit[1]
                slope_hi, int_hi = self.neg_max_fit[0], self.neg_max_fit[1]
                coefs, intercept  = self.coefs_pump, self.intercept_pump
            else:
                # idle: handled by variable bounds; no extra constraints needed
                continue

            # --- power lower bound: p[t] - (slope_lo*h[t] + int_lo) >= 0 ---
            def _p_lb_fun(x, _t=t, _sl=slope_lo, _il=int_lo):
                return x[_t] - (_sl * x[TH * 2 + _t] + _il)

            def _p_lb_jac(x, _t=t, _sl=slope_lo):
                jac = np.zeros(n)
                jac[_t] = 1.0
                jac[TH * 2 + _t] = -_sl
                return jac

            constraints.append({'type': 'ineq', 'fun': _p_lb_fun, 'jac': _p_lb_jac})

            # --- power upper bound: (slope_hi*h[t] + int_hi) - p[t] >= 0 ---
            def _p_ub_fun(x, _t=t, _sh=slope_hi, _ih=int_hi):
                return (_sh * x[TH * 2 + _t] + _ih) - x[_t]

            def _p_ub_jac(x, _t=t, _sh=slope_hi):
                jac = np.zeros(n)
                jac[_t] = -1.0
                jac[TH * 2 + _t] = _sh
                return jac

            constraints.append({'type': 'ineq', 'fun': _p_ub_fun, 'jac': _p_ub_jac})

            # --- flow equality: q[t] - poly_q(p[t], h[t]) == 0 ---
            def _flow_eq_fun(x, _t=t, _c=coefs, _ic=intercept):
                p_val = x[_t]
                h_val = x[TH * 2 + _t]
                return x[TH + _t] - _poly_q(p_val, h_val, _c, _ic)

            def _flow_eq_jac(x, _t=t, _c=coefs):
                p_val = x[_t]
                h_val = x[TH * 2 + _t]
                jac = np.zeros(n)
                jac[TH + _t] = 1.0
                jac[_t]         = -_poly_q_dp(p_val, h_val, _c)
                jac[TH * 2 + _t] = -_poly_q_dh(p_val, h_val, _c)
                return jac

            constraints.append({'type': 'eq', 'fun': _flow_eq_fun, 'jac': _flow_eq_jac})

        # --- v_low = poly_v(h[t]) for all t ---
        for t in range(TH):
            def _vh_eq_fun(x, _t=t):
                return x[TH * 3 + _t] - _poly_v(x[TH * 2 + _t], self.h_v_coeffs)

            def _vh_eq_jac(x, _t=t):
                jac = np.zeros(n)
                jac[TH * 3 + _t] = 1.0
                jac[TH * 2 + _t] = -np.polyval(self.h_v_deriv, x[TH * 2 + _t])
                return jac

            constraints.append({'type': 'eq', 'fun': _vh_eq_fun, 'jac': _vh_eq_jac})

        # --- volume balance ---
        # t=0: v[0] - v_init - q[0]*3600 == 0
        v_init = self.v_low_init

        def _vbal_0_fun(x, _vi=v_init):
            return x[TH * 3] - _vi - x[TH] * 3600.0

        def _vbal_0_jac(x):
            jac = np.zeros(n)
            jac[TH * 3] = 1.0
            jac[TH]     = -3600.0
            return jac

        constraints.append({'type': 'eq', 'fun': _vbal_0_fun, 'jac': _vbal_0_jac})

        for t in range(1, TH):
            def _vbal_t_fun(x, _t=t):
                return x[TH * 3 + _t] - x[TH * 3 + _t - 1] - x[TH + _t] * 3600.0

            def _vbal_t_jac(x, _t=t):
                jac = np.zeros(n)
                jac[TH * 3 + _t]     = 1.0
                jac[TH * 3 + _t - 1] = -1.0
                jac[TH + _t]         = -3600.0
                return jac

            constraints.append({'type': 'eq', 'fun': _vbal_t_fun, 'jac': _vbal_t_jac})

        # --- terminal volume: target_vol_low - v[23] >= 0 ---
        tgt = self.target_vol_low

        def _vterm_fun(x, _tgt=tgt):
            return _tgt - x[TH * 3 + TH - 1]

        def _vterm_jac(x):
            jac = np.zeros(n)
            jac[TH * 3 + TH - 1] = -1.0
            return jac

        constraints.append({'type': 'ineq', 'fun': _vterm_fun, 'jac': _vterm_jac})

        return constraints

    # ------------------------------------------------------------------
    # Variable bounds
    # ------------------------------------------------------------------

    def _build_bounds(self, mode_list):
        """Build variable bounds as list of (lb, ub) tuples."""
        TH = self.TH
        INF = 1e30
        bounds = []

        # p[t]
        for t, mode in enumerate(mode_list):
            if mode == 'idle':
                bounds.append((0.0, 0.0))
            elif mode == 'turbine':
                bounds.append((0.0, INF))
            else:  # pump
                bounds.append((-INF, 0.0))

        # q[t]
        for t, mode in enumerate(mode_list):
            if mode == 'idle':
                bounds.append((0.0, 0.0))
            else:
                bounds.append((-INF, INF))

        # h[t]
        for _ in range(TH):
            bounds.append((self.head_min, self.head_max))

        # v[t]
        for _ in range(TH):
            bounds.append((-INF, INF))

        return bounds

    # ------------------------------------------------------------------
    # Warm-start initialization
    # ------------------------------------------------------------------

    def _build_x0(self, mode_list, warm_start_power, warm_start_head, warm_start_flow):
        """Build initial variable vector."""
        TH = self.TH
        x0 = np.zeros(4 * TH)

        if warm_start_power is not None:
            x0[:TH] = np.array(warm_start_power, dtype=float)
        if warm_start_flow is not None:
            x0[TH:2 * TH] = np.array(warm_start_flow, dtype=float)
        if warm_start_head is not None:
            x0[2 * TH:3 * TH] = np.array(warm_start_head, dtype=float)
            # Initialize v_low from h using the polynomial
            for t in range(TH):
                x0[3 * TH + t] = _poly_v(x0[2 * TH + t], self.h_v_coeffs)
        else:
            # Default head = midpoint of bounds
            h_mid = 0.5 * (self.head_min + self.head_max)
            x0[2 * TH:3 * TH] = h_mid
            for t in range(TH):
                x0[3 * TH + t] = _poly_v(h_mid, self.h_v_coeffs)

        # Clamp idle mode
        for t, mode in enumerate(mode_list):
            if mode == 'idle':
                x0[t] = 0.0
                x0[TH + t] = 0.0

        return x0

    # ------------------------------------------------------------------
    # Public solve method
    # ------------------------------------------------------------------

    def solve(self, prices, modes, warm_start_power=None, warm_start_head=None,
              warm_start_flow=None, tee=False):
        """
        Build and solve the NLP for a single day.

        Args:
            prices:            array-like of 24 hourly prices
            modes:             array-like of 24 mode indicators
                               (>0.5 = turbine, <-0.5 = pump, else idle)
                               or strings ('turbine', 'pump', 'idle')
            warm_start_power:  optional array of 24 initial power values
            warm_start_head:   optional array of 24 initial head values
            warm_start_flow:   optional array of 24 initial flow values
            tee:               whether to print IPOPT output

        Returns:
            dict with keys: 'power', 'flow', 'head', 'v_low', 'objective', 'status'
        """
        prices    = np.array(prices, dtype=float).flatten()
        mode_list = self._parse_modes(modes)

        x0          = self._build_x0(mode_list, warm_start_power, warm_start_head, warm_start_flow)
        bounds      = self._build_bounds(mode_list)
        constraints = self._build_constraints(mode_list)

        options = {
            'max_iter':       self.ipopt_max_iter,
            'tol':            self.ipopt_tol,
            'max_cpu_time':   float(self.ipopt_time_limit),
            'print_level':    5 if tee else 0,
        }

        result = minimize_ipopt(
            fun=self._objective,
            x0=x0,
            args=(prices,),
            jac=self._gradient,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        x   = result.x
        TH  = self.TH
        obj = -result.fun  # convert back to profit (we minimized negative)

        return {
            'power':     x[:TH],
            'flow':      x[TH:2 * TH],
            'head':      x[2 * TH:3 * TH],
            'v_low':     x[3 * TH:4 * TH],
            'objective': obj,
            'status':    result.message,
            'success':   result.success,
        }

    @staticmethod
    def _parse_modes(modes):
        """Parse mode indicators into string labels."""
        result = []
        for m in modes:
            if isinstance(m, str):
                m_lo = m.lower()
                if 'tur' in m_lo:
                    result.append('turbine')
                elif 'pump' in m_lo:
                    result.append('pump')
                else:
                    result.append('idle')
            else:
                v = float(m)
                if v > 0.5:
                    result.append('turbine')
                elif v < -0.5:
                    result.append('pump')
                else:
                    result.append('idle')
        return result
