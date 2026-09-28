"""Common pyTDGL device construction and measurement drivers."""
import os
import numpy as np
import tdgl
from tdgl.geometry import box, circle

import params as P



def make_layer():
    return tdgl.Layer(
        coherence_length=P.XI_0_NM * 1e-3,      # um
        london_lambda=P.LAMBDA_0_NM * 1e-3,     # um
        thickness=P.THICKNESS_NM * 1e-3,        # um
        conductivity=P.SIGMA_N * 1e-6,          # S/um
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


def vortices_on_mesh(sites, triangles, psi):
    """Vortex cores as mesh triangles with phase winding +-1 (holes carry no
    triangles, so hole fluxoids are excluded automatically). Returns
    centroids (same length units as `sites`) and the winding sign."""
    ph = np.angle(psi)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    w = (np.angle(np.exp(1j * (ph[b] - ph[a])))
         + np.angle(np.exp(1j * (ph[c] - ph[b])))
         + np.angle(np.exp(1j * (ph[a] - ph[c])))) / (2 * np.pi)
    m = np.abs(w) > 0.5
    cen = sites[triangles[m]].mean(axis=1)
    return cen, np.rint(w[m]).astype(int)
