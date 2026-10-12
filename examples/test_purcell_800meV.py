"""
Focused 800-meV Purcell-factor test for the inverse triangular PhC.

Purpose
-------
Do NOT sweep the whole spectrum.  Compute only E = 800 meV and check
whether the integrated total Purcell factor is high and whether it is
converging with q-space sampling.

Paper geometry:
    a = 445 nm
    r = 0.405 a
    h = 230 nm
    inverse PhC = air holes in Si on SiO2
    B-point, x-oriented dipole
    z = h/2

Expected paper benchmark:
    the 800-meV B-point x-dipole peak is > 10.

Khepri conventions used here:
    backend = "dense"
    pw = (7,7)
    passive substrate loss = -0.001j
    Purcell reference = homogeneous silicon, n = 3.48

The script tests several q-grid densities at ONLY 800 meV.
"""

# ============================================================
# 0. LIMIT BLAS THREADS BEFORE NUMPY IMPORT
# ============================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ============================================================
# 1. IMPORTS
# ============================================================

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import dipole_power_q, homogeneous_reference_power_xy

# ============================================================
# 2. SETTINGS
# ============================================================

PW = (9, 9)

# q-grid convergence test.
# If runtime is too large, start with [81, 101].
NQ_LIST = [201]

PIX = (512, 512)
WORKERS = max(1, min(8, (os.cpu_count() or 2)//2))

ENERGY_MEV = 800.0

# ============================================================
# 3. PAPER GEOMETRY / MATERIALS
# ============================================================

A_NM = 445.0
H_NM = 230.0
R = 0.405

N_SI = 3.48
N_SIO2 = 1.45

EPS_AIR = 1.0
EPS_SI = N_SI**2
EPS_SUB = N_SIO2**2 - 0.001j

REFERENCE_INDEX = 1.0
CURRENT = (1.0, 0.0, 0.0)  # x dipole

HC_EV_NM = 1239.841984

H = H_NM / A_NM
WAVELENGTH = (HC_EV_NM / (ENERGY_MEV/1000.0)) / A_NM

# Rotated physical triangular lattice used in the current reproduction.
a1 = np.array([np.sqrt(3.0)/2.0, +0.5])
a2 = np.array([np.sqrt(3.0)/2.0, -0.5])
lattice = np.array([a1, a2])

# Current B-point assumption.
B = (a1 + a2)/2.0
DIPOLE_POSITION = (float(B[0]), float(B[1]), H/2.0)

REAL_CELL_AREA = abs(a1[0]*a2[1] - a1[1]*a2[0])

# ============================================================
# 4. RECIPROCAL LATTICE
# ============================================================

def reciprocal_vectors(u, v):
    det = u[0]*v[1] - u[1]*v[0]

    b1 = (2*np.pi/det) * np.array([v[1], -v[0]])
    b2 = (2*np.pi/det) * np.array([-u[1], u[0]])

    # Safety check
    dual = np.array([
        [np.dot(u, b1), np.dot(u, b2)],
        [np.dot(v, b1), np.dot(v, b2)],
    ])

    if not np.allclose(dual, 2*np.pi*np.eye(2), atol=1e-11):
        raise RuntimeError("Reciprocal-lattice construction failed.")

    return b1, b2

b1, b2 = reciprocal_vectors(a1, a2)

# ============================================================
# 5. BUILD CRYSTAL
# ============================================================

def build_crystal():
    c = Crystal(
        pw=PW,
        lattice=lattice,
        lattice_pitch=1.0,
        epsi=EPS_AIR,
        epse=EPS_SUB,
    )

    d = Drawing(
        PIX,
        EPS_SI,
        c.lattice,
    )

    d.disc(
        (0.0, 0.0),
        R,
        EPS_AIR,
    )

    pattern = d.canvas()

    # Split exactly at the dipole plane.
    c.add_layer_pixmap("L", pattern, H/2.0)
    c.add_layer_pixmap("R", pattern, H/2.0)

    c.set_device(
        ["L", "R"],
        fields_mask=[True, False],
    )

    return c

# ============================================================
# 6. ONE q-POINT
# ============================================================

def solve_q(crystal, qx, qy, details=False):
    return dipole_power_q(
        crystal=crystal,
        source_after="L",
        wavelength=WAVELENGTH,
        kp=(float(qx), float(qy)),
        position=DIPOLE_POSITION,
        current=CURRENT,
        backend="dense",
        return_details=details,
    )

# ============================================================
# 7. q GRID
# ============================================================

def make_q_grid(N):
    """
    Uniform midpoint grid over one reciprocal primitive cell:

        q = u*b1 + v*b2,
        -1/2 <= u,v < 1/2.

    For odd N, the grid is inversion symmetric and includes q=0.
    """

    if N % 2 == 0:
        raise ValueError("NQ must be odd for the q<->-q speedup.")

    u = (np.arange(N) + 0.5)/N - 0.5
    v = (np.arange(N) + 0.5)/N - 0.5

    U, V = np.meshgrid(u, v, indexing="xy")

    QX = U*b1[0] + V*b2[0]
    QY = U*b1[1] + V*b2[1]

    return QX, QY

# ============================================================
# 8. MULTIPROCESSING
# ============================================================

_worker_crystal = None

def init_worker():
    global _worker_crystal
    _worker_crystal = build_crystal()

def worker(task):
    qx, qy, multiplicity = task

    p = solve_q(
        _worker_crystal,
        qx,
        qy,
    )["P_total"]

    return float(p), int(multiplicity)

# ============================================================
# 9. 800-meV PURCELL FACTOR FOR ONE NQ
# ============================================================

def purcell_for_NQ(executor, N):
    QX, QY = make_q_grid(N)

    tasks = []

    # Use P(q)=P(-q).  For an odd midpoint grid, inversion is exact.
    for iy in range(N):
        for ix in range(N):

            jy = N - 1 - iy
            jx = N - 1 - ix

            if iy > jy or (iy == jy and ix > jx):
                continue

            multiplicity = 1 if (iy == jy and ix == jx) else 2

            tasks.append(
                (
                    QX[iy, ix],
                    QY[iy, ix],
                    multiplicity,
                )
            )

    chunksize = max(
        1,
        len(tasks)//(WORKERS*20),
    )

    weighted_sum = 0.0
    pmax = -np.inf
    pmax_q = None

    # To retain q location of sampled maximum, map task/output in order.
    for task, (p, multiplicity) in zip(
        tasks,
        executor.map(
            worker,
            tasks,
            chunksize=chunksize,
        ),
    ):
        qx, qy, _ = task

        weighted_sum += multiplicity*p

        if p > pmax:
            pmax = p
            pmax_q = (qx, qy)

    # Reciprocal primitive-cell midpoint quadrature:
    #
    #   integral P(q) d²q/(2pi)²
    #     = mean(P)/A_real
    #
    P_structure = (
        weighted_sum
        /
        (N*N*REAL_CELL_AREA)
    )

    P_reference = homogeneous_reference_power_xy(
        wavelength=WAVELENGTH,
        refractive_index=REFERENCE_INDEX,
        current=CURRENT,
    )

    Fp = P_structure / P_reference

    return {
        "NQ": N,
        "solves": len(tasks),
        "P_structure": float(P_structure),
        "P_reference": float(P_reference),
        "F_purcell": float(Fp),
        "P_q_max": float(pmax),
        "q_at_max": tuple(float(x) for x in pmax_q),
    }

# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("="*74)
    print("FOCUSED PURCELL TEST AT 800 meV")
    print("="*74)

    print(f"PW                  = {PW}")
    print(f"NQ values           = {NQ_LIST}")
    print(f"workers             = {WORKERS}")
    print(f"lambda/a            = {WAVELENGTH:.9f}")
    print(f"B/a                 = {B}")
    print(f"z/a                 = {H/2:.9f}")
    print(f"substrate epsilon   = {EPS_SUB}")
    print(f"reference n         = {REFERENCE_INDEX}")

    # --------------------------------------------------------
    # Source sanity check at Gamma
    # --------------------------------------------------------

    c = build_crystal()

    g = solve_q(
        c,
        0.0,
        0.0,
        details=True,
    )

    print(
        f"\nGamma source check: "
        f"P={g['P_total']:.6e}, "
        f"jump={g['jump_relative_error']:.3e}"
    )

    if g["jump_relative_error"] > 1e-8:
        raise RuntimeError("Dipole source-jump check failed.")

    # --------------------------------------------------------
    # Explicit reciprocity check
    # --------------------------------------------------------

    qtest = 0.6*b1 + 0.13*b2

    p1 = solve_q(c, qtest[0], qtest[1])["P_total"]
    p2 = solve_q(c, -qtest[0], -qtest[1])["P_total"]

    rel = abs(p1-p2)/max(abs(p1), abs(p2), 1e-14)

    print(
        f"Reciprocity check: rel.diff={rel:.3e}"
    )

    if rel > 2e-5:
        raise RuntimeError(
            "P(q)=P(-q) check failed. "
            "Do not use the inversion speedup."
        )

    # --------------------------------------------------------
    # NQ convergence at ONLY 800 meV
    # --------------------------------------------------------

    results = []

    total_start = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
    ) as executor:

        for N in NQ_LIST:

            t0 = time.perf_counter()

            result = purcell_for_NQ(
                executor,
                N,
            )

            dt = time.perf_counter() - t0
            results.append(result)

            print("\n" + "-"*74)

            print(
                f"NQ={N:3d}  "
                f"solves={result['solves']:6d}  "
                f"time={dt:7.1f} s"
            )

            print(
                f"P_structure = "
                f"{result['P_structure']:.12e}"
            )

            print(
                f"P_reference = "
                f"{result['P_reference']:.12e}"
            )

            print(
                f"Fp(800 meV) = "
                f"{result['F_purcell']:.8f}"
            )

            print(
                f"largest sampled P(q) = "
                f"{result['P_q_max']:.6e}"
            )

            print(
                f"q at sampled max = "
                f"{result['q_at_max']}"
            )

    # --------------------------------------------------------
    # Convergence summary
    # --------------------------------------------------------

    print("\n" + "="*74)
    print("CONVERGENCE SUMMARY")
    print("="*74)

    for result in results:
        print(
            f"NQ={result['NQ']:3d}  "
            f"Fp={result['F_purcell']:.8f}"
        )

    for a, b in zip(results[:-1], results[1:]):

        rel_change = (
            abs(
                b["F_purcell"]
                -
                a["F_purcell"]
            )
            /
            max(
                abs(b["F_purcell"]),
                1e-14,
            )
        )

        print(
            f"NQ {a['NQ']} -> {b['NQ']}: "
            f"relative Fp change = "
            f"{100*rel_change:.3f}%"
        )

    best = results[-1]["F_purcell"]

    print("\n" + "="*74)

    if best > 10:
        print(
            f"RESULT: HIGH PURCELL FACTOR. "
            f"Finest-grid Fp(800 meV) = {best:.6f} > 10."
        )
    elif best > 5:
        print(
            f"RESULT: appreciable enhancement, but below the paper's >10 "
            f"benchmark. Finest-grid Fp = {best:.6f}."
        )
    else:
        print(
            f"RESULT: NOT YET reproducing the paper's high 800-meV peak. "
            f"Finest-grid Fp = {best:.6f}."
        )

    print(
        f"\nTotal runtime = "
        f"{(time.perf_counter()-total_start)/60:.2f} min"
    )

    # Save numerical output only.
    np.savez(
        "purcell_800meV_convergence.npz",
        NQ=np.array([r["NQ"] for r in results]),
        Fp=np.array([r["F_purcell"] for r in results]),
        P_structure=np.array([r["P_structure"] for r in results]),
        P_reference=np.array([r["P_reference"] for r in results]),
        P_q_max=np.array([r["P_q_max"] for r in results]),
    )

    print(
        "\nSaved numerical convergence data -> "
        "purcell_800meV_convergence.npz"
    )


if __name__ == "__main__":
    main()
