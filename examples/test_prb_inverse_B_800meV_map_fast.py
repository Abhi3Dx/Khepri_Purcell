"""
PRB 108, 155416 - inverse triangular photonic crystal
800 meV, B-point, x-oriented dipole

Updated q-space map:
- pw fixed to (7, 7)
- Nq increased to 81
- lattice rotated by 90 degrees relative to Khepri's built-in hexagonal orientation
- true hexagonal first Brillouin zone
- red colormap
- smoother tricontourf visualization
- multiprocessing for q-point solves
- reciprocity P(q)=P(-q) used to compute only half of the FBZ, but only
  after an explicit numerical validation

This script is still a q-space MAP diagnostic.  Do not use the coarse map
itself as the final converged Purcell-factor integral.
"""

# ============================================================================
# IMPORTANT: set BLAS/OpenMP threads BEFORE importing NumPy.
#
# We parallelize over independent q points using Python processes.  Restricting
# each worker to one BLAS thread prevents CPU oversubscription.
# ============================================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import dipole_power_q


# ============================================================================
# 1. NUMERICAL SETTINGS
# ============================================================================

# User requested (7,7) only.
PW = (7, 7)

# Increased from the old 31 x 31 diagnostic grid.
NQ = 101

# This controls only the real-space pixmap used to represent the circular hole.
# The visible q-map smoothness is controlled by NQ, not this number.
PIXMAP_RESOLUTION = (512, 512)

BACKEND = "dense"

# Parallel q-point evaluation.
USE_MULTIPROCESSING = True

# Leave a little CPU headroom.  Because every worker uses one BLAS thread,
# several workers can run efficiently in parallel.
_cpu_count = os.cpu_count() or 2
MAX_WORKERS = max(1, min(8, _cpu_count // 2))

# Reciprocity can nearly halve the number of q solves:
#
#       P(q) = P(-q)
#
# for this reciprocal in-plane system and the same linearly polarized source.
# We do NOT assume it blindly: the script tests it first.
TRY_RECIPROCITY_SPEEDUP = True

RECIPROCITY_RTOL = 2.0e-5
RECIPROCITY_ATOL = 1.0e-10


# ============================================================================
# 2. PAPER PARAMETERS
# ============================================================================

A_NM = 445.0
H_NM = 230.0
R_OVER_A = 0.405

ENERGY_EV = 0.800
HC_EV_NM = 1239.841984

WAVELENGTH_NM = HC_EV_NM / ENERGY_EV

# Normalize all lengths to a = 1.
A = 1.0
H = H_NM / A_NM
HALF_H = 0.5 * H
R = R_OVER_A
WAVELENGTH = WAVELENGTH_NM / A_NM

# Same material values used in the validated planar benchmark.
N_SI = 3.48
N_SIO2 = 1.45

EPS_AIR = 1.0
EPS_SI = N_SI**2
EPS_SIO2 = N_SIO2**2

# Corrected Khepri passive-loss convention.
EPS_SUBSTRATE = EPS_SIO2 - 0.001j

# x-oriented electric dipole.
CURRENT = (1.0, 0.0, 0.0)


# ============================================================================
# 3. TRIANGULAR LATTICE - ROTATED ORIENTATION
# ============================================================================

# Khepri's built-in hexagonal lattice was visually rotated by 90 degrees
# relative to the orientation we want for comparison with the paper.
#
# We therefore define the primitive vectors explicitly:
#
#            a1        a2
#             \        /
#              \      /
#               \    /
#                \  /
#
# a1 = (-1/2, sqrt(3)/2)
# a2 = (+1/2, sqrt(3)/2)
#
# This is exactly the same triangular Bravais lattice, rotated by 90 degrees.

A1 = np.array(
    [
        -0.5,
        np.sqrt(3.0) / 2.0,
    ],
    dtype=float,
)

A2 = np.array(
    [
        +0.5,
        np.sqrt(3.0) / 2.0,
    ],
    dtype=float,
)

PAPER_LATTICE = np.array(
    [
        A1,
        A2,
    ],
    dtype=float,
)


# ============================================================================
# 4. REAL-SPACE A/B/C POSITIONS
# ============================================================================

# These are defined relative to the primitive vectors, so the source rotates
# together with the lattice.
#
# The paper draws A, B and C graphically rather than printing their Cartesian
# coordinates.  This assignment follows the standard triangular-cell
# high-symmetry construction used in our first reproduction:
#
# A = centroid of the triangular void
# B = midpoint of the edge joining neighboring pore centers a1 and a2
# C = midpoint of the edge joining 0 and a1

RHO_A = (A1 + A2) / 3.0
RHO_B = (A1 + A2) / 2.0
RHO_C = A1 / 2.0

DIPOLE_POSITION = (
    float(RHO_B[0]),
    float(RHO_B[1]),
    HALF_H,
)


# ============================================================================
# 5. REAL-SPACE GEOMETRY CHECK
# ============================================================================

def nearest_pore_distance(rho):
    centers = []

    for m in range(-2, 3):
        for n in range(-2, 3):
            centers.append(
                m * A1
                +
                n * A2
            )

    return min(
        np.linalg.norm(
            rho - center
        )
        for center in centers
    )


def check_real_space_points():
    for label, rho in (
        ("A", RHO_A),
        ("B", RHO_B),
        ("C", RHO_C),
    ):
        distance = nearest_pore_distance(
            rho
        )

        if distance <= R:
            raise RuntimeError(
                f"Point {label} lies inside an air pore: "
                f"nearest distance={distance:.12f}, "
                f"hole radius={R:.12f}"
            )


# ============================================================================
# 6. BUILD THE PHOTONIC CRYSTAL
# ============================================================================

def build_crystal():
    crystal = Crystal(
        pw=PW,
        lattice=PAPER_LATTICE,
        lattice_pitch=A,
        epsi=EPS_AIR,
        epse=EPS_SUBSTRATE,
    )

    drawing = Drawing(
        PIXMAP_RESOLUTION,
        EPS_SI,
        crystal.lattice,
    )

    # One air hole centered at the Bravais-lattice origin.
    drawing.disc(
        (0.0, 0.0),
        R,
        EPS_AIR,
    )

    epsilon_pattern = drawing.canvas()

    # The dipole lies at z = h/2, so split the slab there.
    crystal.add_layer_pixmap(
        "phc_left",
        epsilon_pattern,
        HALF_H,
    )

    crystal.add_layer_pixmap(
        "phc_right",
        epsilon_pattern,
        HALF_H,
    )

    crystal.set_device(
        [
            "phc_left",
            "phc_right",
        ],
        fields_mask=[
            True,
            False,
        ],
    )

    return crystal


# ============================================================================
# 7. RECIPROCAL LATTICE
# ============================================================================

def reciprocal_vectors(a1, a2):
    det = (
        a1[0] * a2[1]
        -
        a1[1] * a2[0]
    )

    b1 = (
        2.0
        *
        np.pi
        /
        det
        *
        np.array(
            [
                a2[1],
                -a2[0],
            ],
            dtype=float,
        )
    )

    b2 = (
        2.0
        *
        np.pi
        /
        det
        *
        np.array(
            [
                -a1[1],
                a1[0],
            ],
            dtype=float,
        )
    )

    return b1, b2


B1, B2 = reciprocal_vectors(
    A1,
    A2,
)

# Six shortest reciprocal vectors.
NEAREST_G = np.array(
    [
        +B1,
        -B1,
        +B2,
        -B2,
        +(B1 + B2),
        -(B1 + B2),
    ],
    dtype=float,
)


def inside_first_bz(qx, qy, tol=1e-12):
    """
    Wigner-Seitz test:

        |q| <= |q-G|

    for every nearest reciprocal lattice vector G.

    This is equivalent to

        q.G <= |G|^2/2.
    """
    q = np.array(
        [
            qx,
            qy,
        ],
        dtype=float,
    )

    for G in NEAREST_G:
        if (
            np.dot(q, G)
            >
            0.5
            *
            np.dot(G, G)
            +
            tol
        ):
            return False

    return True


# High-symmetry points for the rotated orientation.
GAMMA = np.array(
    [
        0.0,
        0.0,
    ]
)

M_POINT = np.array(
    [
        0.0,
        2.0 * np.pi / np.sqrt(3.0),
    ]
)

K_POINT = np.array(
    [
        4.0 * np.pi / 3.0,
        0.0,
    ]
)

# True hexagonal first-BZ vertices.
BZ_VERTICES = np.array(
    [
        [+4.0 * np.pi / 3.0, 0.0],
        [+2.0 * np.pi / 3.0, +2.0 * np.pi / np.sqrt(3.0)],
        [-2.0 * np.pi / 3.0, +2.0 * np.pi / np.sqrt(3.0)],
        [-4.0 * np.pi / 3.0, 0.0],
        [-2.0 * np.pi / 3.0, -2.0 * np.pi / np.sqrt(3.0)],
        [+2.0 * np.pi / 3.0, -2.0 * np.pi / np.sqrt(3.0)],
    ],
    dtype=float,
)

BZ_VERTICES_CLOSED = np.vstack(
    [
        BZ_VERTICES,
        BZ_VERTICES[0],
    ]
)


# ============================================================================
# 8. ONE q-POINT SOLVE
# ============================================================================

def solve_power(crystal, qx, qy, return_details=False):
    return dipole_power_q(
        crystal=crystal,
        source_after="phc_left",
        wavelength=WAVELENGTH,
        kp=(
            float(qx),
            float(qy),
        ),
        position=DIPOLE_POSITION,
        current=CURRENT,
        backend=BACKEND,
        return_details=return_details,
    )


# ============================================================================
# 9. MULTIPROCESSING WORKER
# ============================================================================

_WORKER_CRYSTAL = None


def _worker_init():
    """
    Each process owns its own Crystal object.

    Crystal is stateful because every q solve updates the source and scattering
    matrices, so separate worker-local objects are safer than threads.
    """
    global _WORKER_CRYSTAL

    _WORKER_CRYSTAL = build_crystal()


def _worker_power(task):
    """
    task = (iy, ix, qx, qy)
    """
    iy, ix, qx, qy = task

    result = solve_power(
        _WORKER_CRYSTAL,
        qx,
        qy,
        return_details=False,
    )

    return (
        iy,
        ix,
        float(
            result["P_total"]
        ),
    )


# ============================================================================
# 10. RECIPROCITY VALIDATION
# ============================================================================

def validate_reciprocity(crystal):
    """
    Numerically verify P(q) = P(-q) before using that relation as a speedup.

    If the test fails, the code simply computes the full FBZ.
    """

    test_points = [
        np.array(
            [
                0.37 * K_POINT[0],
                0.22 * M_POINT[1],
            ]
        ),
        np.array(
            [
                -0.24 * K_POINT[0],
                0.53 * M_POINT[1],
            ]
        ),
        np.array(
            [
                +0.19 * K_POINT[0],
                -0.44 * M_POINT[1],
            ]
        ),
    ]

    print("\nRECIPROCITY CHECK")

    passed = True

    for index, q in enumerate(
        test_points,
        start=1,
    ):
        if not inside_first_bz(
            q[0],
            q[1],
        ):
            continue

        plus = solve_power(
            crystal,
            q[0],
            q[1],
            return_details=False,
        )["P_total"]

        minus = solve_power(
            crystal,
            -q[0],
            -q[1],
            return_details=False,
        )["P_total"]

        difference = abs(
            plus
            -
            minus
        )

        scale = max(
            abs(plus),
            abs(minus),
            RECIPROCITY_ATOL,
        )

        relative = (
            difference
            /
            scale
        )

        local_pass = np.isclose(
            plus,
            minus,
            rtol=RECIPROCITY_RTOL,
            atol=RECIPROCITY_ATOL,
        )

        print(
            f"  test {index}: "
            f"P(q)={plus:.8e}  "
            f"P(-q)={minus:.8e}  "
            f"rel.diff={relative:.3e}  "
            f"{'PASS' if local_pass else 'FAIL'}"
        )

        passed = (
            passed
            and
            local_pass
        )

    return passed


# ============================================================================
# 11. q GRID
# ============================================================================

def build_q_grid():
    # Bounding rectangle of the horizontally oriented regular hexagon.
    qx_max = (
        4.0
        *
        np.pi
        /
        3.0
    )

    qy_max = (
        2.0
        *
        np.pi
        /
        np.sqrt(3.0)
    )

    qx_values = np.linspace(
        -qx_max,
        +qx_max,
        NQ,
    )

    qy_values = np.linspace(
        -qy_max,
        +qy_max,
        NQ,
    )

    QX, QY = np.meshgrid(
        qx_values,
        qy_values,
        indexing="xy",
    )

    mask = np.zeros(
        QX.shape,
        dtype=bool,
    )

    for iy in range(NQ):
        for ix in range(NQ):
            mask[iy, ix] = inside_first_bz(
                QX[iy, ix],
                QY[iy, ix],
            )

    return (
        qx_values,
        qy_values,
        QX,
        QY,
        mask,
    )


def build_tasks(
    QX,
    QY,
    mask,
    use_reciprocity,
):
    """
    If reciprocity is enabled, compute only one representative of each pair

        (qx,qy) <-> (-qx,-qy)

    and later mirror the result.
    """

    tasks = []

    for iy in range(NQ):
        for ix in range(NQ):
            if not mask[iy, ix]:
                continue

            if use_reciprocity:
                mirror_iy = (
                    NQ
                    -
                    1
                    -
                    iy
                )

                mirror_ix = (
                    NQ
                    -
                    1
                    -
                    ix
                )

                # Only keep one member of the inversion pair.
                if (
                    iy > mirror_iy
                    or
                    (
                        iy == mirror_iy
                        and
                        ix > mirror_ix
                    )
                ):
                    continue

            tasks.append(
                (
                    iy,
                    ix,
                    float(QX[iy, ix]),
                    float(QY[iy, ix]),
                )
            )

    return tasks


# ============================================================================
# 12. MAP CALCULATION
# ============================================================================

def calculate_map(
    main_crystal,
    QX,
    QY,
    mask,
    use_reciprocity,
):
    P = np.full(
        QX.shape,
        np.nan,
        dtype=float,
    )

    tasks = build_tasks(
        QX,
        QY,
        mask,
        use_reciprocity,
    )

    full_count = int(
        np.sum(
            mask
        )
    )

    print(
        f"\nFBZ grid contains {full_count} valid q points."
    )

    if use_reciprocity:
        print(
            f"Reciprocity reduces this to {len(tasks)} actual RCWA solves."
        )
    else:
        print(
            f"Computing all {len(tasks)} q points."
        )

    start = time.perf_counter()

    progress_step = max(
        1,
        len(tasks) // 20,
    )

    if (
        USE_MULTIPROCESSING
        and
        MAX_WORKERS > 1
    ):
        print(
            f"Using {MAX_WORKERS} worker processes."
        )

        # chunksize > 1 reduces Windows inter-process communication overhead.
        chunksize = max(
            1,
            len(tasks)
            //
            (
                MAX_WORKERS
                *
                20
            ),
        )

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=_worker_init,
        ) as executor:

            results = executor.map(
                _worker_power,
                tasks,
                chunksize=chunksize,
            )

            for counter, (
                iy,
                ix,
                value,
            ) in enumerate(
                results,
                start=1,
            ):
                P[iy, ix] = value

                if use_reciprocity:
                    mirror_iy = (
                        NQ
                        -
                        1
                        -
                        iy
                    )

                    mirror_ix = (
                        NQ
                        -
                        1
                        -
                        ix
                    )

                    P[
                        mirror_iy,
                        mirror_ix,
                    ] = value

                if (
                    counter == 1
                    or
                    counter == len(tasks)
                    or
                    counter % progress_step == 0
                ):
                    elapsed = (
                        time.perf_counter()
                        -
                        start
                    )

                    rate = (
                        counter
                        /
                        elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    remaining = (
                        len(tasks)
                        -
                        counter
                    )

                    eta = (
                        remaining
                        /
                        rate
                        if rate > 0
                        else np.nan
                    )

                    print(
                        f"  {counter:5d}/{len(tasks):5d} "
                        f"({100.0 * counter / len(tasks):5.1f}%)  "
                        f"elapsed={elapsed:8.1f}s  "
                        f"ETA={eta:8.1f}s"
                    )

    else:
        print(
            "Multiprocessing disabled; using one process."
        )

        for counter, task in enumerate(
            tasks,
            start=1,
        ):
            iy, ix, qx, qy = task

            result = solve_power(
                main_crystal,
                qx,
                qy,
                return_details=False,
            )

            value = float(
                result["P_total"]
            )

            P[
                iy,
                ix,
            ] = value

            if use_reciprocity:
                mirror_iy = (
                    NQ
                    -
                    1
                    -
                    iy
                )

                mirror_ix = (
                    NQ
                    -
                    1
                    -
                    ix
                )

                P[
                    mirror_iy,
                    mirror_ix,
                ] = value

            if (
                counter == 1
                or
                counter == len(tasks)
                or
                counter % progress_step == 0
            ):
                elapsed = (
                    time.perf_counter()
                    -
                    start
                )

                print(
                    f"  {counter:5d}/{len(tasks):5d} "
                    f"({100.0 * counter / len(tasks):5.1f}%)  "
                    f"elapsed={elapsed:8.1f}s"
                )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"\nMap calculation finished in {elapsed:.2f} s."
    )

    return P


# ============================================================================
# 13. PLOT
# ============================================================================

def save_plot(
    QX,
    QY,
    P,
    mask,
):
    # Import matplotlib only in the main process after all heavy RCWA work.
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    valid_P = P[
        mask
    ]

    positive = valid_P[
        valid_P > 0
    ]

    if len(positive) == 0:
        raise RuntimeError(
            "No positive q-resolved powers found."
        )

    Pmax = float(
        np.max(
            positive
        )
    )

    # Visualization only.  The raw P values remain unchanged.
    floor = (
        Pmax
        *
        1.0e-6
    )

    logP = np.log10(
        np.maximum(
            valid_P,
            floor,
        )
        /
        Pmax
    )

    scale_q = (
        2.0
        *
        np.pi
    )

    xplot = (
        QX[mask]
        /
        scale_q
    )

    yplot = (
        QY[mask]
        /
        scale_q
    )

    triangulation = mtri.Triangulation(
        xplot,
        yplot,
    )

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            6.3,
        )
    )

    # 121 contour levels gives a visually smooth map while the true physical
    # resolution still comes from the 81x81 q sampling.
    levels = np.linspace(
        -6.0,
        0.0,
        121,
    )

    contour = ax.tricontourf(
        triangulation,
        logP,
        levels=levels,
        cmap="hot",
        extend="min",
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
        pad=0.025,
    )

    cbar.set_label(
        r"$\log_{10}[P(\mathbf{q})/P_{\max}]$"
    )

    ax.plot(
        BZ_VERTICES_CLOSED[:, 0]
        /
        scale_q,
        BZ_VERTICES_CLOSED[:, 1]
        /
        scale_q,
        linewidth=1.6,
    )

    # High-symmetry points.
    ax.scatter(
        [
            GAMMA[0]
            /
            scale_q
        ],
        [
            GAMMA[1]
            /
            scale_q
        ],
        s=28,
    )

    ax.text(
        GAMMA[0] / scale_q,
        GAMMA[1] / scale_q,
        r"$\Gamma$",
        ha="left",
        va="bottom",
    )

    ax.scatter(
        [
            M_POINT[0]
            /
            scale_q
        ],
        [
            M_POINT[1]
            /
            scale_q
        ],
        s=28,
    )

    ax.text(
        M_POINT[0] / scale_q,
        M_POINT[1] / scale_q,
        "M",
        ha="left",
        va="bottom",
    )

    ax.scatter(
        [
            K_POINT[0]
            /
            scale_q
        ],
        [
            K_POINT[1]
            /
            scale_q
        ],
        s=28,
    )

    ax.text(
        K_POINT[0] / scale_q,
        K_POINT[1] / scale_q,
        "K",
        ha="left",
        va="bottom",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        r"$q_x a/(2\pi)$"
    )

    ax.set_ylabel(
        r"$q_y a/(2\pi)$"
    )

    ax.set_title(
        "Inverse triangular PhC, 800 meV\n"
        "B-point, x-oriented dipole"
    )

    fig.tight_layout()

    output_png = (
        "prb_inverse_B_x_800meV_fbz_map_red.png"
    )

    fig.savefig(
        output_png,
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved figure: {os.path.abspath(output_png)}"
    )


# ============================================================================
# 14. MAIN
# ============================================================================

def main():
    print("=" * 92)
    print("PRB INVERSE TRIANGULAR PhC - 800 meV - B POINT - x DIPOLE")
    print("=" * 92)

    check_real_space_points()

    print("\nPHYSICAL PARAMETERS")
    print(f"a                                  = {A_NM:.6f} nm")
    print(f"h                                  = {H_NM:.6f} nm")
    print(f"r                                  = {R_OVER_A * A_NM:.6f} nm")
    print(f"energy                             = {ENERGY_EV * 1000.0:.3f} meV")
    print(f"vacuum wavelength                  = {WAVELENGTH_NM:.9f} nm")

    print("\nNORMALIZED PARAMETERS")
    print(f"h/a                                = {H:.12f}")
    print(f"r/a                                = {R:.12f}")
    print(f"lambda/a                           = {WAVELENGTH:.12f}")

    print("\nMATERIALS")
    print(f"eps_air                            = {EPS_AIR}")
    print(f"eps_Si                             = {EPS_SI}")
    print(f"eps_substrate                      = {EPS_SUBSTRATE}")

    print("\nNUMERICS")
    print(f"pw                                 = {PW}")
    print(f"retained rectangular harmonics     = {PW[0] * PW[1]}")
    print(f"Nq                                 = {NQ}")
    print(f"pixmap resolution                  = {PIXMAP_RESOLUTION}")
    print(f"backend                            = {BACKEND}")
    print(f"multiprocessing                    = {USE_MULTIPROCESSING}")
    print(f"worker processes                   = {MAX_WORKERS}")

    print("\nROTATED REAL-SPACE LATTICE")
    print(f"a1                                 = {A1}")
    print(f"a2                                 = {A2}")

    print("\nREAL-SPACE HIGH-SYMMETRY POSITIONS")
    print(f"A/a                                = {RHO_A}")
    print(f"B/a                                = {RHO_B}")
    print(f"C/a                                = {RHO_C}")
    print(f"selected dipole position           = {DIPOLE_POSITION}")
    print(
        f"nearest pore distance at B         = "
        f"{nearest_pore_distance(RHO_B):.12f}"
    )

    print("\nRECIPROCAL LATTICE")
    print(f"b1                                 = {B1}")
    print(f"b2                                 = {B2}")
    print(f"Gamma                              = {GAMMA}")
    print(f"M                                  = {M_POINT}")
    print(f"K                                  = {K_POINT}")

    print("\nBuilding main Crystal object...")
    crystal = build_crystal()

    # ------------------------------------------------------------------------
    # Special q points
    # ------------------------------------------------------------------------

    print("\n" + "=" * 92)
    print("STEP 1 - GAMMA / M / K SANITY CHECK")
    print("=" * 92)

    for name, q in (
        ("Gamma", GAMMA),
        ("M", M_POINT),
        ("K", K_POINT),
    ):
        result = solve_power(
            crystal,
            q[0],
            q[1],
            return_details=True,
        )

        print(
            f"{name:5s}: "
            f"P_left={result['P_left']:+.8e}  "
            f"P_right={result['P_right']:+.8e}  "
            f"P_total={result['P_total']:+.8e}  "
            f"jump_err={result['jump_relative_error']:.3e}"
        )

        if not np.isfinite(
            result["P_total"]
        ):
            raise RuntimeError(
                f"{name} power is non-finite."
            )

        if (
            result["jump_relative_error"]
            >
            1.0e-8
        ):
            raise RuntimeError(
                f"{name} source jump failed."
            )

        if (
            result["P_total"]
            <
            -1.0e-8
        ):
            raise RuntimeError(
                f"{name} has significant negative emitted power."
            )

    # ------------------------------------------------------------------------
    # Validate reciprocity speedup
    # ------------------------------------------------------------------------

    use_reciprocity = False

    if TRY_RECIPROCITY_SPEEDUP:
        use_reciprocity = validate_reciprocity(
            crystal
        )

        if use_reciprocity:
            print(
                "[PASS] Reciprocity verified. "
                "The map will compute only half of the FBZ."
            )
        else:
            print(
                "[WARNING] Reciprocity test did not meet tolerance. "
                "The full FBZ will be calculated."
            )

    # ------------------------------------------------------------------------
    # q-space map
    # ------------------------------------------------------------------------

    print("\n" + "=" * 92)
    print("STEP 2 - HIGHER-RESOLUTION HEXAGONAL FBZ MAP")
    print("=" * 92)

    (
        qx_values,
        qy_values,
        QX,
        QY,
        mask,
    ) = build_q_grid()

    P = calculate_map(
        main_crystal=crystal,
        QX=QX,
        QY=QY,
        mask=mask,
        use_reciprocity=use_reciprocity,
    )

    valid_P = P[
        mask
    ]

    if not np.all(
        np.isfinite(
            valid_P
        )
    ):
        raise RuntimeError(
            "FBZ map contains NaN or infinity."
        )

    print("\nMAP DIAGNOSTICS")
    print(
        f"min(P_total)                       = "
        f"{np.min(valid_P):.12e}"
    )
    print(
        f"max(P_total)                       = "
        f"{np.max(valid_P):.12e}"
    )
    print(
        f"mean(P_total)                      = "
        f"{np.mean(valid_P):.12e}"
    )
    print(
        f"all sampled powers nonnegative     = "
        f"{np.min(valid_P) >= -1.0e-8}"
    )

    # ------------------------------------------------------------------------
    # Strongest sampled points
    # ------------------------------------------------------------------------

    flat_indices = np.argwhere(
        mask
    )

    values = np.array(
        [
            P[iy, ix]
            for iy, ix in flat_indices
        ],
        dtype=float,
    )

    order = np.argsort(
        values
    )[::-1]

    print("\nSTRONGEST SAMPLED q POINTS")

    # All six M points = half the six shortest reciprocal vectors.
    m_points = (
        0.5
        *
        NEAREST_G
    )

    for rank in range(
        min(
            12,
            len(order),
        )
    ):
        iy, ix = flat_indices[
            order[rank]
        ]

        q = np.array(
            [
                QX[iy, ix],
                QY[iy, ix],
            ]
        )

        value = P[
            iy,
            ix,
        ]

        distance_to_m = min(
            np.linalg.norm(
                q - m
            )
            for m in m_points
        )

        print(
            f"{rank + 1:2d}: "
            f"q=({q[0]:+.8f},{q[1]:+.8f})  "
            f"P={value:.8e}  "
            f"d(nearest M)={distance_to_m:.4e}"
        )

    # ------------------------------------------------------------------------
    # Save raw data
    # ------------------------------------------------------------------------

    output_npz = (
        "prb_inverse_B_x_800meV_fbz_map_red.npz"
    )

    np.savez(
        output_npz,
        P_total=P,
        mask=mask,
        qx=QX,
        qy=QY,
        qx_values=qx_values,
        qy_values=qy_values,
        a1=A1,
        a2=A2,
        b1=B1,
        b2=B2,
        bz_vertices=BZ_VERTICES,
        rho_A=RHO_A,
        rho_B=RHO_B,
        rho_C=RHO_C,
        position=np.asarray(
            DIPOLE_POSITION
        ),
        current=np.asarray(
            CURRENT
        ),
        wavelength=WAVELENGTH,
        wavelength_nm=WAVELENGTH_NM,
        energy_eV=ENERGY_EV,
        pw=np.asarray(
            PW
        ),
        Nq=NQ,
        reciprocity_used=use_reciprocity,
    )

    print(
        f"\nSaved raw map: {os.path.abspath(output_npz)}"
    )

    # ------------------------------------------------------------------------
    # Smooth red visualization
    # ------------------------------------------------------------------------

    save_plot(
        QX=QX,
        QY=QY,
        P=P,
        mask=mask,
    )

    print("\n" + "=" * 92)
    print("DONE")
    print("=" * 92)
    print(
        "This is a higher-resolution diagnostic map. "
        "The next step is to compare its resonance topology with Fig. 3(a), "
        "then construct a resonance-aware 2D integration rather than simply "
        "summing this uniform grid."
    )


# ============================================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
