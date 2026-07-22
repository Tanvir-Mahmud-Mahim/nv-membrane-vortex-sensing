"""Field-cooled vortex configurations via a two-stage protocol.

A clean mesoscopic strip has a strong Bean-Livingston edge barrier, so at
fixed temperature the film stays in a metastable Meissner state at millitesla
fields. Real field-cooled devices nucleate vortices at T ~ Tc where the
barrier vanishes. We emulate this with a two-stage protocol:
  stage 1: nucleation pulse at B_seed >> B_target (order parameter suppressed,
           flux floods in),
  stage 2: relax at B_target with the stage-1 state as the initial condition;
           excess vortices exit and the trapped configuration equilibrates.
Saves the same npz payload as scripts/01_targets_fieldcool.py.
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import tdgl
from tdgl_common import make_device
import params as P

B_SEED = 30.0  # mT

CASES = [
    ["bare_B4", 0.0, None, 4.14],
    ["bare_B8", 0.0, None, 8.27],
    ["anti15_B4", 0.15, 0.5, 4.14],
    ["anti15_B8", 0.15, 0.5, 8.27],
    ["anti20_B8", 0.20, 0.5, 8.27],
    ["anti15_B12", 0.15, 0.5, 12.4],
]


def vortex_positions(sol, dev, holes_xy, hole_diam, grid_n=240):
    mesh = dev.mesh
    psi = sol.tdgl_data.psi
    ph = np.angle(psi)
    x, y = mesh.sites[:, 0], mesh.sites[:, 1]
    gx = np.linspace(x.min(), x.max(), grid_n)
    gy = np.linspace(y.min(), y.max(), grid_n)
    GX, GY = np.meshgrid(gx, gy)
    Ia = LinearNDInterpolator(mesh.sites, np.abs(psi), fill_value=1.0)
    A = Ia(GX, GY)
    Ic = LinearNDInterpolator(mesh.sites, np.exp(1j * ph), fill_value=1.0)
    Ph = np.angle(Ic(GX, GY))
    d1 = np.angle(np.exp(1j * (Ph[:-1, 1:] - Ph[:-1, :-1])))
    d2 = np.angle(np.exp(1j * (Ph[1:, 1:] - Ph[:-1, 1:])))
    d3 = np.angle(np.exp(1j * (Ph[1:, :-1] - Ph[1:, 1:])))
    d4 = np.angle(np.exp(1j * (Ph[:-1, :-1] - Ph[1:, :-1])))
    W = (d1 + d2 + d3 + d4) / (2 * np.pi)
    iy, ix = np.where(np.abs(W) > 0.5)
    pos = np.stack([(gx[ix] + gx[ix + 1]) / 2, (gy[iy] + gy[iy + 1]) / 2], 1)
    keep = []
    for p in pos:
        if any(np.hypot(p[0] - hx, p[1] - hy) < hole_diam / 2 + 0.02
               for hx, hy in holes_xy):
            continue
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < 0.04 for q in keep):
            continue
        keep.append(p)
    return np.array(keep).reshape(-1, 2), A, (gx, gy)


def solve_at(dev, B, t, seed=None):
    opts = tdgl.SolverOptions(solve_time=t, field_units="mT",
                              current_units="uA", save_every=500,
                              progress_interval=0)
    return tdgl.solve(dev, opts, applied_vector_potential=B,
                      seed_solution=seed)


for tag, D, pitch, B in CASES:
    t0 = time.time()
    dev, centers = make_device(tag, hole_diam=D, pitch=pitch, transport=False)
    sol1 = solve_at(dev, B_SEED, 60)
    sol = solve_at(dev, B, 250, seed=sol1)
    occ = [sum(sol.hole_fluxoid(f"hole{i}")).to("Phi_0").magnitude
           for i in range(len(centers))]
    vpos, A, (gx, gy) = vortex_positions(sol, dev, centers, D)
    K = sol.current_density.to("uA/um")
    np.savez(
        f"data/fieldcool_{tag}.npz",
        sites=dev.mesh.sites, psi_abs=np.abs(sol.tdgl_data.psi),
        psi_phase=np.angle(sol.tdgl_data.psi),
        K=np.asarray(K.magnitude), B_mT=B, hole_diam=D,
        pitch=pitch if pitch else 0.0,
        hole_centers=np.array(centers).reshape(-1, 2),
        hole_occupancy=np.array(occ), vortex_pos=vpos,
        grid_x=gx, grid_y=gy, psi_grid=A,
    )
    print(f"{tag}: B={B} mT holes={len(centers)} "
          f"occ={np.round(np.abs(occ),2).tolist() if occ else []} "
          f"interstitial={len(vpos)} ({time.time()-t0:.0f} s)", flush=True)
print("DONE seeded fieldcool")
