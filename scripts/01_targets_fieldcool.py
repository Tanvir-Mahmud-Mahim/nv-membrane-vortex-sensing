"""Field-cooled vortex configurations of the sensing target.

Devices: bare Ta strip and antidot (pinning-hole) lattices.
For each device and field, relax the order parameter at fixed field and save:
  - order parameter |psi| and phase on the mesh
  - sheet current density
  - hole fluxoid occupancy
  - interstitial vortex positions (phase-winding census on a grid)
Outputs: data/fieldcool_<tag>.npz
"""
import sys, time, json, os
sys.path.insert(0, ".")
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tdgl_common import make_device, field_cool
import params as P

CASES = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
    # tag, hole_diam (um), pitch (um), B_mT
    ["bare_B4", 0.0, None, 4.14],
    ["bare_B8", 0.0, None, 8.27],
    ["anti15_B4", 0.15, 0.5, 4.14],
    ["anti15_B8", 0.15, 0.5, 8.27],
    ["anti20_B8", 0.20, 0.5, 8.27],
    ["anti15_B12", 0.15, 0.5, 12.4],
]


def vortex_positions(sol, dev, holes_xy, hole_diam, grid_n=240):
    """Locate interstitial vortices as local minima of |psi| with phase
    winding of 2*pi, excluding hole interiors."""
    mesh = dev.mesh
    x, y = mesh.sites[:, 0], mesh.sites[:, 1]
    psi = sol.tdgl_data.psi
    absp = np.abs(psi)
    ph = np.angle(psi)
    gx = np.linspace(x.min(), x.max(), grid_n)
    gy = np.linspace(y.min(), y.max(), grid_n)
    GX, GY = np.meshgrid(gx, gy)
    Ia = LinearNDInterpolator(mesh.sites, absp, fill_value=1.0)
    A = Ia(GX, GY)
    # winding via interpolated complex field
    Ic = LinearNDInterpolator(mesh.sites, np.exp(1j * ph), fill_value=1.0)
    C = Ic(GX, GY)
    Ph = np.angle(C)
    # candidate cells: plaquette winding
    d1 = np.angle(np.exp(1j * (Ph[:-1, 1:] - Ph[:-1, :-1])))
    d2 = np.angle(np.exp(1j * (Ph[1:, 1:] - Ph[:-1, 1:])))
    d3 = np.angle(np.exp(1j * (Ph[1:, :-1] - Ph[1:, 1:])))
    d4 = np.angle(np.exp(1j * (Ph[:-1, :-1] - Ph[1:, :-1])))
    W = (d1 + d2 + d3 + d4) / (2 * np.pi)
    iy, ix = np.where(np.abs(W) > 0.5)
    pos = np.stack([(gx[ix] + gx[ix + 1]) / 2, (gy[iy] + gy[iy + 1]) / 2], axis=1)
    # cluster within one mesh cell and exclude holes
    keep = []
    for p in pos:
        if any(np.hypot(p[0] - hx, p[1] - hy) < hole_diam / 2 + 0.02
               for hx, hy in holes_xy):
            continue
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < 0.04 for q in keep):
            continue
        keep.append(p)
    return np.array(keep).reshape(-1, 2), A, (gx, gy)


for tag, D, pitch, B in CASES:
    t0 = time.time()
    dev, centers = make_device(tag, hole_diam=D, pitch=pitch, transport=False)
    sol = field_cool(dev, B, cool_time=200)
    occ = []
    for i in range(len(centers)):
        fl = sum(sol.hole_fluxoid(f"hole{i}")).to("Phi_0").magnitude
        occ.append(fl)
    vpos, A, (gx, gy) = vortex_positions(sol, dev, centers, D)
    mesh = dev.mesh
    K = sol.current_density.to("uA/um")  # sheet current on mesh sites
    np.savez(
        f"data/fieldcool_{tag}.npz",
        sites=mesh.sites, psi_abs=np.abs(sol.tdgl_data.psi),
        psi_phase=np.angle(sol.tdgl_data.psi),
        K=np.asarray(K.magnitude), B_mT=B, hole_diam=D,
        pitch=pitch if pitch else 0.0,
        hole_centers=np.array(centers).reshape(-1, 2),
        hole_occupancy=np.array(occ), vortex_pos=vpos,
        grid_x=gx, grid_y=gy, psi_grid=A,
    )
    print(f"{tag}: B={B} mT, holes={len(centers)}, "
          f"occ_sum={np.nansum(occ):.1f}, interstitial={len(vpos)}, "
          f"{time.time()-t0:.0f} s", flush=True)
print("DONE fieldcool")
