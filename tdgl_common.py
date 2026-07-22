"""Common pyTDGL device construction and measurement drivers."""
import os
import numpy as np
import tdgl
from tdgl.geometry import box, circle

import params as P

os.makedirs("data", exist_ok=True)


def make_layer():
    return tdgl.Layer(
        coherence_length=P.XI_0_NM * 1e-3,      # um
        london_lambda=P.LAMBDA_0_NM * 1e-3,     # um
        thickness=P.THICKNESS_NM * 1e-3,        # um
        gamma=P.GAMMA_TDGL,
    )


def hole_centers(pitch, w=None, l=None, margin=0.18):
    """Square antidot lattice centered in the strip, keeping a margin from
    the terminals so current injection stays uniform."""
    w = w or P.FILM_W
    l = l or P.FILM_L
    nx = int(np.floor((l - 2 * margin) / pitch)) + 1
    ny = int(np.floor((w - 2 * margin) / pitch)) + 1
    xs = (np.arange(nx) - (nx - 1) / 2) * pitch
    ys = (np.arange(ny) - (ny - 1) / 2) * pitch
    return [(x, y) for x in xs for y in ys]


def make_device(name, hole_diam=0.0, pitch=None, w=None, l=None,
                transport=True, mesh_edge=None):
    """Transport strip along x with optional square antidot lattice."""
    w = w or P.FILM_W
    l = l or P.FILM_L
    mesh_edge = mesh_edge or P.MESH_EDGE
    layer = make_layer()
    film = tdgl.Polygon("film", points=box(l, w)).resample(401).buffer(0)
    holes = []
    centers = []
    if hole_diam > 0 and pitch:
        centers = hole_centers(pitch, w=w, l=l)
        npts = max(24, int(np.pi * hole_diam / (mesh_edge * 0.8)))
        for i, (x, y) in enumerate(centers):
            holes.append(
                tdgl.Polygon(f"hole{i}",
                             points=circle(hole_diam / 2, center=(x, y),
                                           points=npts)).resample(npts).buffer(0)
            )
    kw = {}
    if transport:
        src = tdgl.Polygon("source", points=box(mesh_edge, w, center=(-l / 2, 0))
                           ).resample(101).buffer(0)
        drn = tdgl.Polygon("drain", points=box(mesh_edge, w, center=(l / 2, 0))
                           ).resample(101).buffer(0)
        kw = dict(terminals=[src, drn],
                  probe_points=[(-0.4 * l, 0), (0.4 * l, 0)])
    dev = tdgl.Device(name, layer=layer, film=film, holes=holes,
                      length_units="um", **kw)
    dev.make_mesh(max_edge_length=mesh_edge, smooth=100)
    return dev, centers


def field_cool(dev, B_mT, cool_time=150, save_every=400):
    """Field-cooled (zero-current) relaxation at field B_mT."""
    opts = tdgl.SolverOptions(
        solve_time=cool_time, field_units="mT", current_units="uA",
        save_every=save_every, progress_interval=0,
    )
    return tdgl.solve(dev, opts, applied_vector_potential=B_mT)


def staircase_iv(dev, B_mT, currents_uA, hold=40.0, skip=10.0,
                 seed=None, save_every=400):
    """Current staircase at fixed field. Returns (I, V) arrays with V the
    time-average voltage in reduced units V0 over the tail of each step."""
    currents_uA = np.asarray(currents_uA, float)
    total = hold * len(currents_uA)

    def term_curr(t):
        i = min(int(t // hold), len(currents_uA) - 1)
        return {"source": currents_uA[i], "drain": -currents_uA[i]}

    opts = tdgl.SolverOptions(
        solve_time=total, field_units="mT", current_units="uA",
        save_every=save_every, progress_interval=0,
    )
    sol = tdgl.solve(dev, opts, applied_vector_potential=B_mT,
                     terminal_currents=term_curr, seed_solution=seed)
    dyn = sol.dynamics
    V = []
    for i in range(len(currents_uA)):
        t0, t1 = i * hold + skip, (i + 1) * hold
        V.append(dyn.mean_voltage(tmin=t0, tmax=t1))
    return currents_uA, np.array(V), sol


XI_UM = P.XI_0_NM * 1e-3  # pyTDGL mesh coordinates are in units of xi


def load_fieldcool(tag):
    """Load a fieldcool npz with all coordinates converted to um.

    pyTDGL stores mesh sites (and hence the derived grids and vortex
    positions) in coherence-length units; hole centers were stored in um.
    Interstitial vortices are re-filtered against hole interiors after the
    unit conversion.
    """
    d = dict(np.load(f"data/fieldcool_{tag}.npz"))
    for k in ["sites", "vortex_pos", "grid_x", "grid_y"]:
        d[k] = d[k] * XI_UM
    keep = []
    for p in d["vortex_pos"]:
        if any(np.hypot(p[0] - hx, p[1] - hy) < d["hole_diam"] / 2 + 0.03
               for hx, hy in d["hole_centers"]):
            continue
        keep.append(p)
    d["vortex_pos"] = np.array(keep).reshape(-1, 2)
    # Robust integer occupancy: phase winding on a ring at 0.35*pitch around
    # each hole center, evaluated on the interpolated complex order parameter.
    from scipy.interpolate import LinearNDInterpolator
    if len(d["hole_centers"]):
        Ic = LinearNDInterpolator(d["sites"], np.exp(1j * d["psi_phase"]),
                                  fill_value=1.0)
        pitch = float(d["pitch"]) or 0.5
        th = np.linspace(0, 2 * np.pi, 241)
        wind = []
        for (hx, hy) in d["hole_centers"]:
            ring = Ic(hx + 0.35 * pitch * np.cos(th),
                      hy + 0.35 * pitch * np.sin(th))
            dph = np.angle(ring[1:] / ring[:-1])
            wind.append(int(np.rint(np.abs(np.sum(dph)) / (2 * np.pi))))
        d["occ_winding"] = np.array(wind)
    else:
        d["occ_winding"] = np.array([], dtype=int)
    d["n_trapped"] = int(np.sum(d["occ_winding"] > 0))
    return d


def count_vortices_in_film(sol, centers, hole_diam, r_probe=0.11):
    """Fluxoid census: number of flux quanta trapped in each hole and an
    estimate of interstitial (free) vortices from the total applied flux."""
    occ = []
    for i in range(len(centers)):
        try:
            fl = sum(sol.hole_fluxoid(f"hole{i}")).to("Phi_0").magnitude
        except Exception:
            fl = np.nan
        occ.append(abs(fl))
    return np.array(occ)
