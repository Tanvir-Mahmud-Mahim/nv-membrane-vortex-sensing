"""Core physics library: vortex stray field, vortex noise, NV rates,
fluctuation-dissipation (FDT) loss, and the quasiparticle noise background.

Conventions
-----------
* SI units throughout unless a name says otherwise.
* Spectral densities are ONE-SIDED, S(f) with <x^2> = int_0^inf S df, written
  as functions of angular frequency w for convenience.
* Harmonic vortex (Langevin): eta_v dx/dt = -k_p x + noise, per in-plane axis.
      S_x(w)   = 4 kB T eta_v / (k_p^2 + w^2 eta_v^2)
      Im a(w)  = w eta_v / (k_p^2 + w^2 eta_v^2)     (a = x/F, response)
      FDT      : S_x(w) = (4 kB T / w) Im a(w)
* Power absorbed by one vortex from a force F cos(wt):
      P = (w/2) F^2 Im a(w) = w^2 F^2 S_x(w) / (8 kB T)
* NV rates for a field noise tensor S_ij (one-sided), NV axis n:
      Gamma_(0->+-1) = (gamma^2/4) [Tr S - n.S.n]      (transverse)
      1/T1           = 3 Gamma_(0->+-1)
      Gamma_phi      = (gamma^2/4) n.S(0).n            (motional narrowing)
  The thin normal-film limit reproduces 1/T1 = 3 gamma^2 mu0^2 kB T sigma a
  / (32 pi z^2) of Kolkowitz et al. (checked in tests.py).
"""
import numpy as np
from scipy.special import j0, j1
from scipy.integrate import quad

from params import PHI0, MU0, KB, HBAR, GAMMA_E


# ----------------------------------------------------------------------------
# 1. Stray field of a straight vortex in a London film of thickness d
# ----------------------------------------------------------------------------
def vortex_kernel(k, lam, d):
    """B_z(k) at the top surface divided by Phi0, for a vortex threading a
    London film of thickness d (exact solution of the London equation with
    field matching at both surfaces). Limits: Pearl 1/(1+k Lambda) for
    d << lam, bulk 1/(lam^2 tau (tau+k)) for d >> lam."""
    k = np.asarray(k, float)
    tau = np.sqrt(k**2 + lam**-2)
    x = tau * d / 2
    coth = 1.0 / np.tanh(np.minimum(x, 350.0))
    return 1.0 / (lam**2 * tau * (tau + k * coth))


def _kgrid(h, rho_max):
    """Uniform k grid: k_max = 40/h (e^{-kh} < 1e-17) and >= 60 points per
    J0 oscillation period at the largest radius."""
    kmax = 40.0 / h
    dk = min(0.1 / max(rho_max, h), kmax / 4000)
    return np.linspace(0.0, kmax, int(np.ceil(kmax / dk)) + 1)


def vortex_radial(rho, h, lam, d, form=None, chunk=400):
    """Radial functions at height h above the film surface for one flux
    quantum: returns Bz, Brho, dBz/drho, dBrho/drho (tesla, tesla/m).
    `form(k)` is an optional in-plane form factor (e.g. a hole disk).
    Intended for rho <~ 5 um (maps and line cuts)."""
    rho = np.atleast_1d(np.asarray(rho, float))
    k = _kgrid(h, rho.max())
    F = PHI0 * vortex_kernel(k, lam, d) * np.exp(-k * h)
    if form is not None:
        F = F * form(k)
    w = np.full_like(k, k[1] - k[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    base = (k * F * w) / (2 * np.pi)
    out = np.zeros((4, rho.size))
    for i0 in range(0, rho.size, chunk):
        r = rho[i0:i0 + chunk]
        kr = np.outer(r, k)
        J0 = j0(kr)
        J1 = j1(kr)
        out[0, i0:i0 + chunk] = J0 @ base
        out[1, i0:i0 + chunk] = J1 @ base
        out[2, i0:i0 + chunk] = -(J1 @ (k * base))
        safe = np.where(kr > 1e-9, kr, 1.0)
        J1o = np.where(kr > 1e-9, J1 / safe, 0.5)
        out[3, i0:i0 + chunk] = (J0 - J1o) @ (k * base)
    return out[0], out[1], out[2], out[3]


def disk_form(R):
    """Form factor of flux spread uniformly over a disk of radius R
    (used for a fluxoid trapped in a hole; an approximation)."""
    def f(k):
        x = np.asarray(k) * R
        out = np.ones_like(x)
        m = x > 1e-8
        out[m] = 2 * j1(x[m]) / x[m]
        return out
    return f


def nv_axis(theta_deg, phi_deg=0.0):
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def field_and_gradient_maps(X, Y, h, lam, d, form=None):
    """Field B_i and gradient G_ij = dB_i/dx_j (j over in-plane x, y) on a
    Cartesian grid for one vortex at the origin. Shapes: B (3, ...),
    G (3, 2, ...). A displacement u of the vortex changes the field at a
    fixed sensor by dB_i = -G_ij u_j."""
    R = np.hypot(X, Y)
    rr = np.linspace(0, R.max() * 1.001 + 1e-12, 1500)
    Bz, Br, dBz, dBr = vortex_radial(rr, h, lam, d, form)
    iz = np.interp(R, rr, Bz)
    ir = np.interp(R, rr, Br)
    idz = np.interp(R, rr, dBz)
    idr = np.interp(R, rr, dBr)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(R > 0, X / np.where(R > 0, R, 1), 1.0)
        s = np.where(R > 0, Y / np.where(R > 0, R, 1), 0.0)
        Br_over = np.where(R > 1e-12, ir / np.where(R > 1e-12, R, 1), idr)
    B = np.stack([ir * c, ir * s, iz])
    G = np.empty((3, 2) + X.shape)
    G[0, 0] = idr * c * c + Br_over * s * s
    G[0, 1] = (idr - Br_over) * c * s
    G[1, 0] = G[0, 1]
    G[1, 1] = idr * s * s + Br_over * c * c
    G[2, 0] = idz * c
    G[2, 1] = idz * s
    return B, G


def geometric_factors(G, n):
    """Transverse and parallel gradient factors (T^2/m^2) for an isotropic
    in-plane displacement noise: S_perp_total = S_x * g_perp,
    S_par = S_x * g_par."""
    par = np.einsum("i,ij...->j...", n, G)          # (2, ...)
    tot = np.sum(G**2, axis=(0, 1))
    g_par = np.sum(par**2, axis=0)
    return tot - g_par, g_par


# ----------------------------------------------------------------------------
# 2. Vortex dynamics, FDT and loss
# ----------------------------------------------------------------------------
def Sx_harmonic(w, T, eta_v, kp):
    return 4 * KB * T * eta_v / (kp**2 + (w * eta_v) ** 2)


def Im_alpha_harmonic(w, eta_v, kp):
    return w * eta_v / (kp**2 + (w * eta_v) ** 2)


def Sx_telegraph(w, delta, rate):
    """Two-site hopping, hop length delta, symmetric rate `rate` per
    direction; one-sided spectrum along the hop axis."""
    tc = 1.0 / (2 * rate)
    return delta**2 * tc / (1 + (w * tc) ** 2)


def power_from_noise(w, F, Sx, T):
    """FDT: time-averaged power absorbed from a force F cos(wt)."""
    return w**2 * F**2 * Sx / (8 * KB * T)


def nv_rates_vortex(Sx_nv, Sx_0, g_perp, g_par, gamma=GAMMA_E):
    """Excess NV rates from one vortex: 1/T1 and the motional-narrowing
    dephasing rate."""
    inv_T1 = 3 * gamma**2 / 4 * Sx_nv * g_perp
    gphi = gamma**2 / 4 * Sx_0 * g_par
    return inv_T1, gphi


def invert_two_clock(S_nv, S_0, w_nv, T):
    """Recover (eta_v, kp) from the displacement spectrum at the NV
    frequency and at zero frequency (harmonic model)."""
    A = S_0 / (4 * KB * T)       # eta/kp^2
    B = S_nv / (4 * KB * T)      # eta/(kp^2 + w^2 eta^2)
    eta = (1 / B - 1 / A) / w_nv**2
    kp = np.sqrt(np.clip(eta / A, 0, None))
    return eta, kp


def cpw_current_factor(y_from_center, w):
    """Thin-strip Meissner sheet-current profile normalised to I/w:
    g(y) = (2/pi) / sqrt(1 - (2y/w)^2)."""
    u = 2 * np.asarray(y_from_center) / w
    return (2 / np.pi) / np.sqrt(1 - u**2)


def per_vortex_inverse_Q(w_res, eta_v, kp, g, s, w_strip, Z0):
    """Loss added by one vortex in a half-wave CPW resonator.
    g: sheet-current factor at the vortex (K = g I/w), s = |sin(k x)|
    standing-wave factor. Stored energy W = Z0 I0^2 / (8 f)."""
    pref = 2 * PHI0**2 * g**2 * s**2 / (np.pi * w_strip**2 * Z0)
    return pref * w_res * Im_alpha_harmonic(w_res, eta_v, kp)


# ----------------------------------------------------------------------------
# 3. Quasiparticle (Johnson) noise of the superconducting film
# ----------------------------------------------------------------------------
def bcs_gap(T, Tc):
    D0 = 1.764 * KB * Tc
    if T >= Tc:
        return 0.0
    return D0 * np.tanh(1.74 * np.sqrt(Tc / T - 1))


def _fermi(E, T):
    return 1.0 / (np.exp(np.clip(E / (KB * T), -700, 700)) + 1.0)


def mattis_bardeen(T, Tc, f):
    """sigma1/sigma_n and sigma2/sigma_n (Mattis-Bardeen, hbar w < 2 Delta)."""
    D = bcs_gap(T, Tc)
    hw = HBAR * 2 * np.pi * f
    assert hw < 2 * D
    # sigma1: E = D cosh u removes the edge singularity
    def i1(u):
        E = D * np.cosh(u)
        E2 = E + hw
        num = (E**2 + D**2 + hw * E) * (_fermi(E, T) - _fermi(E2, T))
        return num / np.sqrt(E2**2 - D**2)  # dE = D sinh u du cancels sqrt(E^2-D^2)
    s1 = 2 / hw * quad(i1, 0, 40, limit=400)[0]
    # sigma2: algebraic end-point weights at E = D - hw and E = D
    a, b = D - hw, D
    def i2(E):
        return ((1 - 2 * _fermi(E + hw, T)) * (E**2 + D**2 + hw * E)
                / np.sqrt(E + hw + D) / np.sqrt(D + E))
    s2 = 1 / hw * quad(i2, a, b, weight="alg", wvar=(-0.5, -0.5),
                       limit=400)[0]
    return s1, s2


def slab_reflection(q, w, sigma, d):
    """Quasi-static TE reflection coefficient of a conducting slab of
    thickness d and complex conductivity sigma (exp(-i w t) convention)."""
    tau = np.sqrt(q**2 - 1j * w * MU0 * sigma + 0j)
    tau = np.where(tau.real < 0, -tau, tau)
    e = np.exp(-2 * tau * d)
    return ((q**2 - tau**2) * (1 - e)) / ((q + tau) ** 2 - (q - tau) ** 2 * e)


def sheet_noise_Szz(h, T, w, sigma, d, nq=6000):
    """One-sided B_z noise at height h above the slab (T^2/Hz)."""
    q = np.linspace(1e-3 / h, 40.0 / h, nq)
    r = slab_reflection(q, w, sigma, d)
    integ = q**2 * np.abs(r.imag) * np.exp(-2 * q * h)
    I = np.trapezoid(integ, q)
    return 4 * KB * T / w * MU0 / (4 * np.pi) * I


def nv_T1_from_sheet(Szz, n, gamma=GAMMA_E):
    """1/T1 for sheet noise (S_xx = S_yy = S_zz/2), NV axis n."""
    ct2 = n[2] ** 2
    trans = Szz * (3 - ct2) / 2
    return 3 * gamma**2 / 4 * trans
