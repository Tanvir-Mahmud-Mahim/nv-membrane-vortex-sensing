"""Numerical self-checks of physics.py against closed-form limits.
Run: python tests.py  (all checks must print PASS)."""
import numpy as np
import physics as ph
from params import PHI0, MU0, KB, HBAR, GAMMA_E

ok = True
def check(name, a, b, rtol):
    global ok
    r = abs(a - b) / abs(b)
    flag = r < rtol
    ok &= flag
    print(f"{'PASS' if flag else 'FAIL'}  {name}: {a:.6g} vs {b:.6g} (rel {r:.2e})")

lam, d, h = 50e-9, 150e-9, 50e-9
# 1. flux quantisation through the plane at height h
rho = np.linspace(0, 4e-6, 8001)
Bz, Br, dBz, dBr = ph.vortex_radial(rho, h, lam, d)
flux = np.trapezoid(2*np.pi*rho*Bz, rho)
Rm, hp = rho[-1], h + lam
tail = PHI0*hp/np.sqrt(Rm**2+hp**2)   # monopole flux outside R
check("flux through plane = Phi0 (4 um disk + far tail)", flux+tail, PHI0, 3e-3)
# 2. far field of a thick film: monopole at depth ~lambda below surface
r0 = 3e-6
Bz_far = np.interp(r0, rho, Bz)
hp = h + lam/np.tanh(d/(2*lam))   # effective monopole depth lambda*coth(d/2lambda)
mono = PHI0*hp/(2*np.pi*(r0**2+hp**2)**1.5)
check("far field vs monopole at depth lambda*coth(d/2lambda)", Bz_far, mono, 2e-2)
# 3. Pearl limit kernel
k = np.logspace(4, 9, 50)
dd = 2e-9; Lam = 2*lam**2/dd
check("Pearl limit of kernel (k=1e7)", ph.vortex_kernel(1e7, lam, dd), 1/(1+1e7*Lam), 1e-2)
# 4. derivative consistency
num = np.gradient(Bz, rho)
i = np.argmin(abs(rho-80e-9))
check("dBz/drho analytic vs numeric (rho=80nm)", dBz[i], num[i], 1e-2)
num = np.gradient(Br, rho)
check("dBrho/drho analytic vs numeric (rho=80nm)", dBr[i], num[i], 1e-2)
# 5. normal thin film: Kolkowitz et al. 1/T1 = 3 g^2 mu0^2 kT sigma a/(32 pi z^2)
sig, a, z, T, f = 1e7, 5e-9, 100e-9, 300.0, 2.87e9
w = 2*np.pi*f
Szz = ph.sheet_noise_Szz(z, T, w, sig+0j, a, nq=20000)
rate = ph.nv_T1_from_sheet(Szz, np.array([0,0,1.0]))
# independent Biot-Savart sum over thin layers: int dz'/z'^2 = a/(z(z+a))
ref = 3*GAMMA_E**2*MU0**2*KB*T*sig*a/(32*np.pi*z*(z+a))
check("normal film 1/T1 vs Biot-Savart Johnson-noise formula", rate, ref, 1e-2)
# 6. Mattis-Bardeen: sigma2/sigma_n -> pi Delta/(hbar w) tanh(Delta/2kT) for hw<<Delta
Tc = 4.39
s1, s2 = ph.mattis_bardeen(0.3, Tc, 1e8)
D = ph.bcs_gap(0.3, Tc)
check("MB sigma2 low-T, low-f limit", s2, np.pi*D/(HBAR*2*np.pi*1e8)*np.tanh(D/(2*KB*0.3)), 2e-2)
# 7. MB sigma1 -> 1 near Tc (normal state recovery, T=0.999 Tc)
from scipy.special import k0
Tq, fq = 1.0, 5e9
s1, s2 = ph.mattis_bardeen(Tq, Tc, fq)
D = ph.bcs_gap(Tq, Tc); x = HBAR*2*np.pi*fq/(2*KB*Tq)
approx = 4*D/(HBAR*2*np.pi*fq)*np.exp(-D/(KB*Tq))*np.sinh(x)*k0(x)
check("MB sigma1 vs low-T asymptotic formula (1 K, 5 GHz)", s1, approx, 3e-2)
# 8. FDT identity for the harmonic model
wv = np.logspace(5, 11, 7); eta, kp = 7.5e-15, 1e-6
P1 = ph.power_from_noise(wv, 1e-12, ph.Sx_harmonic(wv, 2.0, eta, kp), 2.0)
P2 = wv/2*1e-24*ph.Im_alpha_harmonic(wv, eta, kp)
check("FDT: w^2F^2S/(8kT) = (w/2)F^2 Im a", P1[3], P2[3], 1e-12)
# 9. two-clock inversion round trip
wn = 2*np.pi*2.87e9
e2, k2 = ph.invert_two_clock(ph.Sx_harmonic(wn,2,eta,kp), ph.Sx_harmonic(0,2,eta,kp), wn, 2)
check("two-clock inversion eta", e2, eta, 1e-9); check("two-clock inversion kp", k2, kp, 1e-9)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
