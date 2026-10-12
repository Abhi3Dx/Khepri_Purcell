"""
Fig. 4(c) focused reproduction around 800 meV
================================================

This script computes the BZ-integrated TOTAL Purcell factor for the
inverse triangular photonic-crystal slab, B-point, x-oriented dipole,
while resolving the very narrow band-edge resonances near the M points.

Main idea
---------
A simple uniform midpoint grid can miss an extremely narrow resonance at
the BZ edge.  Here we therefore use:

    1) a coarse uniform midpoint grid for the smooth BZ background;
    2) three local nonuniform q-grids centered exactly on the three
       inequivalent M points of the reciprocal torus;
    3) strong point clustering around M so that the M point itself and
       its immediate neighborhood are explicitly sampled.

The local M patches exactly replace selected cells of the coarse grid,
so they are not double-counted.

Current intended setup
----------------------
Inverse PhC:
    air holes in silicon on silica
    a = 445 nm
    r = 0.405 a
    h = 230 nm
    dipole at z = h/2
    B-point, x-oriented dipole

Numerics:
    backend = "dense"
    PW = (9, 9)
    substrate loss = -0.001j in the corrected Khepri convention
    reference n = 1.0 for vacuum-referenced Purcell factor

The default energy window is 790--810 meV in 2-meV steps.
After locating the peak, reduce ENERGY_STEP_MEV to 0.5 or 0.2 meV.

IMPORTANT
---------
The exact real optical constants used by the paper are not explicitly
given in the main text.  Here we retain the current working values:
    n_Si = 3.48
    n_SiO2 = 1.45

Therefore this is a numerical reproduction test, not a claim of exact
material matching.
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

import csv
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import dipole_power_q, homogeneous_reference_power_xy

# ============================================================
# 2. USER SETTINGS
# ============================================================

PW = (9, 9)
PIX = (384, 384)

WORKERS = max(1, min(8, (os.cpu_count() or 2)//2))

# ------------------------------------------------------------
# Energy scan around the Fig. 4(c) peak
# ------------------------------------------------------------

ENERGY_MIN_MEV = 790.0
ENERGY_MAX_MEV = 810.0
ENERGY_STEP_MEV = 2.0

ENERGIES_MEV = np.arange(
    ENERGY_MIN_MEV,
    ENERGY_MAX_MEV + 0.5*ENERGY_STEP_MEV,
    ENERGY_STEP_MEV,
)

# ------------------------------------------------------------
# q-space integration
# ------------------------------------------------------------

# Smooth-background grid.
# Keep this odd so that Gamma is sampled.
NQ_BACKGROUND = 41

# Number of local points per coordinate in every M patch.
# MUST be odd so that the exact M point is sampled.
#
# 41 gives 1681 samples per M patch before multiprocessing.
# Increase to 61 after the first run if needed.
NQ_LOCAL = 41

# Clustering exponent.
#
# local coordinate = half_width * sign(t) * |t|**CLUSTER_POWER
#
# CLUSTER_POWER = 1 -> uniform local grid
# CLUSTER_POWER = 3 -> strong clustering at M
CLUSTER_POWER = 3.0

# Number of coarse cells replaced around each M direction.
#
# For a coordinate where M lies at the reciprocal-cell edge,
# EDGE_CELLS gives this many coarse cells on EACH side of M.
#
# For a coordinate where M lies at the center (0), CENTER_CELLS
# gives this many cells on each side PLUS the center cell.
EDGE_CELLS = 4
CENTER_CELLS = 4

# Output/checkpoint files
CSV_FILE = "fig4c_around_800meV_adaptive.csv"
NPZ_FILE = "fig4c_around_800meV_adaptive.npz"

# ============================================================
# 3. PAPER GEOMETRY / CURRENT WORKING MATERIALS
# ============================================================

A_NM = 445.0
H_NM = 230.0

R_OVER_A = 0.405

N_SI = 3.48
N_SIO2 = 1.45

EPS_AIR = 1.0
EPS_SI = N_SI**2

# Corrected Khepri outgoing-wave convention:
# passive loss uses negative Im(epsilon).
EPS_SUB = N_SIO2**2 - 0.001j

# Fig. 4(c) working normalization:
REFERENCE_INDEX = 1.0

# x-oriented dipole
CURRENT = (1.0, 0.0, 0.0)

# hc in eV nm
HC_EV_NM = 1239.841984

H = H_NM / A_NM

# ============================================================
# 4. REAL-SPACE TRIANGULAR LATTICE
# ============================================================

a1 = np.array([np.sqrt(3.0)/2.0, +0.5])
a2 = np.array([np.sqrt(3.0)/2.0, -0.5])

lattice = np.array([a1, a2])

REAL_CELL_AREA = abs(
    a1[0]*a2[1] - a1[1]*a2[0]
)

# Current B-point coordinate assumption used in the reproduction.
B = (a1 + a2)/2.0

DIPOLE_POSITION = (
    float(B[0]),
    float(B[1]),
    H/2.0,
)

# ============================================================
# 5. RECIPROCAL LATTICE
# ============================================================

def reciprocal_vectors(a1, a2):
    det = a1[0]*a2[1] - a1[1]*a2[0]

    b1 = (2*np.pi/det) * np.array([
        a2[1],
        -a2[0],
    ])

    b2 = (2*np.pi/det) * np.array([
        -a1[1],
        a1[0],
    ])

    check = np.array([
        [np.dot(a1, b1), np.dot(a1, b2)],
        [np.dot(a2, b1), np.dot(a2, b2)],
    ])

    if not np.allclose(
        check,
        2*np.pi*np.eye(2),
        atol=1e-11,
    ):
        raise RuntimeError(
            "Reciprocal-lattice construction failed."
        )

    return b1, b2


b1, b2 = reciprocal_vectors(a1, a2)

# Three inequivalent M points on the reciprocal torus.
#
# We represent +1/2 as -1/2 so every point lies in
# [-1/2, 1/2).
M_POINTS_UV = [
    (-0.5,  0.0),   # -b1/2  == +b1/2 modulo b1
    ( 0.0, -0.5),   # -b2/2  == +b2/2 modulo b2
    (-0.5, -0.5),   # -(b1+b2)/2
]

# ============================================================
# 6. PERIODIC FRACTIONAL-COORDINATE HELPERS
# ============================================================

def wrap_fractional(x):
    """Map a fractional reciprocal coordinate to [-0.5, 0.5)."""
    return ((x + 0.5) % 1.0) - 0.5


def torus_delta(x, x0):
    """
    Shortest signed difference x-x0 on a unit-periodic coordinate.
    Result lies in [-0.5, 0.5).
    """
    return ((x - x0 + 0.5) % 1.0) - 0.5


def uv_to_q(u, v):
    q = u*b1 + v*b2
    return float(q[0]), float(q[1])


# ============================================================
# 7. M-PATCH GEOMETRY
# ============================================================

if NQ_BACKGROUND % 2 == 0:
    raise ValueError("NQ_BACKGROUND must be odd.")

if NQ_LOCAL % 2 == 0:
    raise ValueError("NQ_LOCAL must be odd.")

du_bg = 1.0 / NQ_BACKGROUND

# If M is at a periodic edge, there is no central coarse cell:
# replace EDGE_CELLS cells on either side.
HALF_EDGE = EDGE_CELLS * du_bg

# If M is at coordinate 0, there IS a central coarse cell.
# Replacing CENTER_CELLS cells on either side plus the center cell
# gives total width (2*CENTER_CELLS+1)/N.
HALF_CENTER = (CENTER_CELLS + 0.5) * du_bg

# Exact rectangular replacement patches.
#
# Each tuple:
#   (u0, v0, half_u, half_v, label)
M_PATCHES = [
    (-0.5,  0.0, HALF_EDGE,   HALF_CENTER, "M1"),
    ( 0.0, -0.5, HALF_CENTER, HALF_EDGE,   "M2"),
    (-0.5, -0.5, HALF_EDGE,   HALF_EDGE,   "M3"),
]


def inside_patch(u, v, patch):
    u0, v0, half_u, half_v, _ = patch

    du = abs(torus_delta(u, u0))
    dv = abs(torus_delta(v, v0))

    return (
        du < half_u - 1e-14
        and
        dv < half_v - 1e-14
    )


def inside_any_M_patch(u, v):
    return any(
        inside_patch(u, v, patch)
        for patch in M_PATCHES
    )


# ============================================================
# 8. BUILD THE CRYSTAL
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

    # Inverse photonic crystal = air hole in silicon.
    d.disc(
        (0.0, 0.0),
        R_OVER_A,
        EPS_AIR,
    )

    pattern = d.canvas()

    # Split at the dipole plane.
    c.add_layer_pixmap(
        "L",
        pattern,
        H/2.0,
    )

    c.add_layer_pixmap(
        "R",
        pattern,
        H/2.0,
    )

    c.set_device(
        ["L", "R"],
        fields_mask=[True, False],
    )

    return c


# ============================================================
# 9. ONE q-POINT
# ============================================================

def wavelength_over_a(energy_mev):
    wavelength_nm = (
        HC_EV_NM
        /
        (energy_mev/1000.0)
    )

    return wavelength_nm / A_NM


def solve_q(crystal, energy_mev, qx, qy, details=False):

    wavelength = wavelength_over_a(energy_mev)

    return dipole_power_q(
        crystal=crystal,
        source_after="L",
        wavelength=wavelength,
        kp=(float(qx), float(qy)),
        position=DIPOLE_POSITION,
        current=CURRENT,
        backend="dense",
        return_details=details,
    )


# ============================================================
# 10. WORKER POOL
# ============================================================

_worker_crystal = None


def init_worker():
    global _worker_crystal
    _worker_crystal = build_crystal()


def worker(task):
    """
    task = (energy_mev, u, v)

    Returns:
        P_total
    """
    energy_mev, u, v = task

    qx, qy = uv_to_q(u, v)

    result = solve_q(
        _worker_crystal,
        energy_mev,
        qx,
        qy,
        details=False,
    )

    return float(result["P_total"])


# ============================================================
# 11. BACKGROUND GRID
# ============================================================

def background_tasks(energy_mev):

    s = (
        (np.arange(NQ_BACKGROUND) + 0.5)
        /
        NQ_BACKGROUND
        -
        0.5
    )

    tasks = []
    kept_uv = []

    excluded = 0

    for v in s:
        for u in s:

            if inside_any_M_patch(u, v):
                excluded += 1
                continue

            tasks.append(
                (
                    float(energy_mev),
                    float(u),
                    float(v),
                )
            )

            kept_uv.append((u, v))

    return tasks, kept_uv, excluded


# ============================================================
# 12. NONUNIFORM LOCAL GRID AROUND M
# ============================================================

def clustered_axis(half_width):
    """
    Odd nonuniform axis on [-half_width, +half_width],
    strongly clustered at zero and including zero exactly.
    """
    t = np.linspace(
        -1.0,
        1.0,
        NQ_LOCAL,
    )

    x = (
        np.sign(t)
        *
        half_width
        *
        np.abs(t)**CLUSTER_POWER
    )

    # Force exact center.
    x[NQ_LOCAL//2] = 0.0

    return x


def local_patch_tasks(energy_mev, patch):

    u0, v0, half_u, half_v, label = patch

    du = clustered_axis(half_u)
    dv = clustered_axis(half_v)

    tasks = []

    for dv_i in dv:
        for du_i in du:

            u = wrap_fractional(
                u0 + du_i
            )

            v = wrap_fractional(
                v0 + dv_i
            )

            tasks.append(
                (
                    float(energy_mev),
                    float(u),
                    float(v),
                )
            )

    return tasks, du, dv, label


# ============================================================
# 13. NUMERICAL INTEGRATION AT ONE ENERGY
# ============================================================

def integrate_energy(executor, energy_mev):

    # --------------------------------------------------------
    # A. smooth background
    # --------------------------------------------------------

    bg_tasks, bg_uv, excluded = background_tasks(
        energy_mev
    )

    chunksize_bg = max(
        1,
        len(bg_tasks)//(WORKERS*20),
    )

    bg_values = list(
        executor.map(
            worker,
            bg_tasks,
            chunksize=chunksize_bg,
        )
    )

    # One background midpoint cell contributes:
    #    du*dv / A_real = 1/(N^2 A_real)
    background_integral = (
        np.sum(bg_values)
        /
        (
            NQ_BACKGROUND**2
            *
            REAL_CELL_AREA
        )
    )

    # --------------------------------------------------------
    # B. refined M patches
    # --------------------------------------------------------

    patch_integrals = {}
    patch_M_values = {}

    local_total = 0.0

    for patch in M_PATCHES:

        tasks, du_axis, dv_axis, label = (
            local_patch_tasks(
                energy_mev,
                patch,
            )
        )

        chunksize_local = max(
            1,
            len(tasks)//(WORKERS*20),
        )

        values = np.array(
            list(
                executor.map(
                    worker,
                    tasks,
                    chunksize=chunksize_local,
                )
            ),
            dtype=float,
        ).reshape(
            NQ_LOCAL,
            NQ_LOCAL,
        )

        # Integrate P(u,v) du dv / A_real.
        #
        # Axis 1 = du
        # Axis 0 = dv
        integ_u = np.trapezoid(
            values,
            x=du_axis,
            axis=1,
        )

        integ_uv = np.trapezoid(
            integ_u,
            x=dv_axis,
            axis=0,
        )

        patch_power = (
            integ_uv
            /
            REAL_CELL_AREA
        )

        patch_integrals[label] = float(
            patch_power
        )

        local_total += patch_power

        # Exact M is the center of the clustered local grid.
        c = NQ_LOCAL//2

        patch_M_values[label] = float(
            values[c, c]
        )

    # --------------------------------------------------------
    # C. total BZ-integrated structured power
    # --------------------------------------------------------

    P_structure = (
        background_integral
        +
        local_total
    )

    # --------------------------------------------------------
    # D. homogeneous reference
    # --------------------------------------------------------

    wavelength = wavelength_over_a(
        energy_mev
    )

    P_reference = (
        homogeneous_reference_power_xy(
            wavelength=wavelength,
            refractive_index=REFERENCE_INDEX,
            current=CURRENT,
        )
    )

    Fp = P_structure / P_reference

    return {
        "energy_mev": float(energy_mev),
        "P_structure": float(P_structure),
        "P_reference": float(P_reference),
        "Fp": float(Fp),
        "background": float(background_integral),
        "M1_patch": patch_integrals["M1"],
        "M2_patch": patch_integrals["M2"],
        "M3_patch": patch_integrals["M3"],
        "P_M1": patch_M_values["M1"],
        "P_M2": patch_M_values["M2"],
        "P_M3": patch_M_values["M3"],
        "excluded_background_cells": int(excluded),
    }


# ============================================================
# 14. CHECKPOINTING
# ============================================================

CSV_FIELDS = [
    "energy_mev",
    "P_structure",
    "P_reference",
    "Fp",
    "background",
    "M1_patch",
    "M2_patch",
    "M3_patch",
    "P_M1",
    "P_M2",
    "P_M3",
    "excluded_background_cells",
]


def load_existing_results():

    if not os.path.exists(CSV_FILE):
        return {}

    results = {}

    with open(
        CSV_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            e = float(row["energy_mev"])

            parsed = {}

            for key in CSV_FIELDS:

                if key == "excluded_background_cells":
                    parsed[key] = int(
                        float(row[key])
                    )
                else:
                    parsed[key] = float(
                        row[key]
                    )

            results[e] = parsed

    return results


def write_all_results(results):

    energies = sorted(results.keys())

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for e in energies:
            writer.writerow(
                results[e]
            )

    np.savez(
        NPZ_FILE,
        energy_mev=np.array(
            energies,
            dtype=float,
        ),
        Fp=np.array(
            [
                results[e]["Fp"]
                for e in energies
            ],
            dtype=float,
        ),
        P_structure=np.array(
            [
                results[e]["P_structure"]
                for e in energies
            ],
            dtype=float,
        ),
        P_reference=np.array(
            [
                results[e]["P_reference"]
                for e in energies
            ],
            dtype=float,
        ),
        background=np.array(
            [
                results[e]["background"]
                for e in energies
            ],
            dtype=float,
        ),
        M1_patch=np.array(
            [
                results[e]["M1_patch"]
                for e in energies
            ],
            dtype=float,
        ),
        M2_patch=np.array(
            [
                results[e]["M2_patch"]
                for e in energies
            ],
            dtype=float,
        ),
        M3_patch=np.array(
            [
                results[e]["M3_patch"]
                for e in energies
            ],
            dtype=float,
        ),
        P_M1=np.array(
            [
                results[e]["P_M1"]
                for e in energies
            ],
            dtype=float,
        ),
        P_M2=np.array(
            [
                results[e]["P_M2"]
                for e in energies
            ],
            dtype=float,
        ),
        P_M3=np.array(
            [
                results[e]["P_M3"]
                for e in energies
            ],
            dtype=float,
        ),
    )


# ============================================================
# 15. SANITY CHECKS
# ============================================================

def sanity_checks():

    c = build_crystal()

    # Source jump at Gamma, 800 meV.
    g = solve_q(
        c,
        800.0,
        0.0,
        0.0,
        details=True,
    )

    print(
        "\nGamma source check:"
        f" P={g['P_total']:.6e},"
        f" jump={g['jump_relative_error']:.3e}"
    )

    if g["jump_relative_error"] > 1e-8:
        raise RuntimeError(
            "Dipole source-jump check failed."
        )

    # Reciprocity at a generic q.
    qtest = (
        0.173*b1
        +
        0.287*b2
    )

    p1 = solve_q(
        c,
        800.0,
        qtest[0],
        qtest[1],
    )["P_total"]

    p2 = solve_q(
        c,
        800.0,
        -qtest[0],
        -qtest[1],
    )["P_total"]

    rel = (
        abs(p1-p2)
        /
        max(
            abs(p1),
            abs(p2),
            1e-14,
        )
    )

    print(
        f"Reciprocity check:"
        f" rel.diff={rel:.3e}"
    )

    if rel > 2e-5:
        raise RuntimeError(
            "Reciprocity check failed."
        )

    # Print the three M points in Cartesian q coordinates.
    print("\nM points used:")

    for i, (u, v) in enumerate(
        M_POINTS_UV,
        start=1,
    ):

        qx, qy = uv_to_q(u, v)

        print(
            f"  M{i}:"
            f" (u,v)=({u:+.3f},{v:+.3f})"
            f" -> q=({qx:+.9f},{qy:+.9f})"
        )

    # Confirm exact area replacement.
    s = (
        (np.arange(NQ_BACKGROUND)+0.5)
        /
        NQ_BACKGROUND
        -
        0.5
    )

    excluded = 0

    for v in s:
        for u in s:
            if inside_any_M_patch(u, v):
                excluded += 1

    coarse_replaced_area = (
        excluded
        /
        NQ_BACKGROUND**2
    )

    exact_patch_area = sum(
        4.0*patch[2]*patch[3]
        for patch in M_PATCHES
    )

    print(
        "\nM-patch replacement check:"
        f"\n  excluded coarse cells = {excluded}"
        f"\n  coarse replaced uv area = {coarse_replaced_area:.12f}"
        f"\n  exact local patch area  = {exact_patch_area:.12f}"
    )

    if not np.isclose(
        coarse_replaced_area,
        exact_patch_area,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Local M-patch area does not exactly match "
            "the excluded coarse-grid area."
        )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("="*78)
    print("FIG. 4(c): PURCELL FACTOR AROUND 800 meV")
    print("BACKGROUND + M-LOCAL REFINED q INTEGRATION")
    print("="*78)

    print(f"PW                         = {PW}")
    print(f"energies (meV)             = {ENERGIES_MEV}")
    print(f"workers                    = {WORKERS}")
    print(f"NQ background              = {NQ_BACKGROUND}")
    print(f"NQ local per M patch       = {NQ_LOCAL}")
    print(f"local clustering exponent  = {CLUSTER_POWER}")
    print(f"B/a                        = {B}")
    print(f"z/a                        = {H/2:.9f}")
    print(f"substrate epsilon          = {EPS_SUB}")
    print(f"reference n                = {REFERENCE_INDEX}")

    sanity_checks()

    existing = load_existing_results()

    if existing:
        print(
            f"\nLoaded {len(existing)} completed "
            f"energy point(s) from {CSV_FILE}"
        )

    t_all = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
    ) as executor:

        for energy in ENERGIES_MEV:

            energy = float(energy)

            if energy in existing:

                print(
                    f"\n{energy:.3f} meV already completed:"
                    f" Fp={existing[energy]['Fp']:.8f}"
                )

                continue

            print("\n" + "-"*78)
            print(
                f"Energy = {energy:.3f} meV"
            )

            t0 = time.perf_counter()

            result = integrate_energy(
                executor,
                energy,
            )

            dt = time.perf_counter() - t0

            existing[energy] = result
            write_all_results(existing)

            print(
                f"P_structure = "
                f"{result['P_structure']:.12e}"
            )

            print(
                f"P_reference = "
                f"{result['P_reference']:.12e}"
            )

            print(
                f"Fp          = "
                f"{result['Fp']:.8f}"
            )

            print(
                f"background  = "
                f"{result['background']:.6e}"
            )

            print(
                "M-patch contributions:"
                f"  M1={result['M1_patch']:.6e},"
                f" M2={result['M2_patch']:.6e},"
                f" M3={result['M3_patch']:.6e}"
            )

            print(
                "Exact-M sampled P(q):"
                f"  M1={result['P_M1']:.6e},"
                f" M2={result['P_M2']:.6e},"
                f" M3={result['P_M3']:.6e}"
            )

            print(
                f"time = {dt/60:.2f} min"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    energies_done = sorted(
        e
        for e in existing
        if (
            e >= ENERGY_MIN_MEV - 1e-12
            and
            e <= ENERGY_MAX_MEV + 1e-12
        )
    )

    if energies_done:

        fp_done = np.array(
            [
                existing[e]["Fp"]
                for e in energies_done
            ]
        )

        i = int(np.argmax(fp_done))

        print("\n" + "="*78)
        print("SUMMARY")
        print("="*78)

        for e, fp in zip(
            energies_done,
            fp_done,
        ):
            print(
                f"{e:8.3f} meV   Fp={fp:.8f}"
            )

        print(
            "\nLargest sampled Purcell factor:"
            f"\n  E  = {energies_done[i]:.3f} meV"
            f"\n  Fp = {fp_done[i]:.8f}"
        )

        print(
            f"\nSaved -> {CSV_FILE}"
        )

        print(
            f"Saved -> {NPZ_FILE}"
        )

    print(
        f"\nTotal wall time = "
        f"{(time.perf_counter()-t_all)/60:.2f} min"
    )


if __name__ == "__main__":
    main()
