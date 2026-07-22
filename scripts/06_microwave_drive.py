"""Direct TDGL microwave-drive validation of the pinning contrast.

Drives the strip with a sinusoidal transport current at the NV transition
frequency f = 2.87 GHz (period 180 tau0 with tau0 = 1.93 ps) at fixed
B = B_phi, for the bare film and the antidot film. The cycle-averaged product
<I V> measures the vortex microwave dissipation directly, the quantity that
the Gittleman-Rosenblum analysis of script 03 models analytically.
"""
import sys, time, json
sys.path.insert(0, ".")
import numpy as np
import tdgl
from tdgl_common import make_device
import params as P

F_GHZ = 2.87
PERIOD_TAU = 1e3 / F_GHZ / (P.TAU_0_S * 1e12)  # ~180 tau
N_CYCLES = 4
I0 = 150.0  # uA, below the depinning staircase onset

out = {"period_tau": PERIOD_TAU, "f_GHz": F_GHZ, "I0_uA": I0}
for tag, D, pitch in [("bare", 0.0, None), ("anti15", 0.15, 0.5)]:
    t0 = time.time()
    dev, centers = make_device(tag + "_mw", hole_diam=D, pitch=pitch,
                               transport=True)
    # field-cooled seed
    o1 = tdgl.SolverOptions(solve_time=60, field_units="mT",
                            current_units="uA", save_every=500,
                            progress_interval=0)
    seed = tdgl.solve(dev, o1, applied_vector_potential=30.0)
    o2 = tdgl.SolverOptions(solve_time=150, field_units="mT",
                            current_units="uA", save_every=500,
                            progress_interval=0)
    seed = tdgl.solve(dev, o2, applied_vector_potential=P.B_PHI_MT,
                      seed_solution=seed)

    def drive(t):
        I = I0 * np.sin(2 * np.pi * t / PERIOD_TAU)
        return {"source": I, "drain": -I}

    total = N_CYCLES * PERIOD_TAU
    o3 = tdgl.SolverOptions(solve_time=total, field_units="mT",
                            current_units="uA", save_every=20,
                            progress_interval=0)
    sol = tdgl.solve(dev, o3, applied_vector_potential=P.B_PHI_MT,
                     terminal_currents=drive, seed_solution=seed)
    dyn = sol.dynamics
    t = dyn.time
    V = dyn.voltage()
    I_t = I0 * np.sin(2 * np.pi * t / PERIOD_TAU)
    # discard first cycle (transient), average I*V over remaining cycles
    m = t > PERIOD_TAU
    P_avg = float(np.mean(I_t[m] * V[m]))
    P_abs = float(np.mean(np.abs(I_t[m] * V[m])))
    out[tag] = {"P_avg_red": P_avg, "P_absavg_red": P_abs,
                "V_rms": float(np.std(V[m])), "runtime_s": time.time() - t0}
    np.savez(f"data/mw_{tag}.npz", t=t, V=V, I=I_t)
    print(tag, out[tag], flush=True)

with open("data/microwave_drive.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE mw")
