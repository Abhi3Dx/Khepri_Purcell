"""
PRB 108, 155416 (2023) - inverse triangular photonic crystal
800 meV, B-point, x-oriented dipole

Workflow
--------
1) Show the REAL-SPACE triangular lattice
   - air holes
   - primitive unit cell
   - B-point
   - x-oriented dipole at B
2) Compute or load the q-resolved near-field emissivity over the first BZ
3) Show the RECIPROCAL-SPACE near-field emissivity map

Important convention
--------------------
- The physical real-space lattice is kept in the rotated orientation:
      a1 = (sqrt(3)/2, +1/2)
      a2 = (sqrt(3)/2, -1/2)
- RCWA always uses the TRUE reciprocal lattice of this real-space lattice.
- ONLY FOR DISPLAY, reciprocal-space coordinates are rotated by -90 degrees
  so that the first Brillouin zone appears with horizontal top/bottom edges.

Numerics
--------
- backend = "dense"
- pw = (7, 7)
- Nq = 101
- multiprocessing over q points
- P(q)=P(-q) is used only after a numerical reciprocity check
- q-map is cached so plotting changes do not require recalculation

Figures are displayed only and are not saved.
"""

# ============================================================
# 0. LIMIT BLAS THREADS BEFORE IMPORTING NUMPY
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
from khepri.dipole import dipole_power_q


# ============================================================
# 2. USER SETTINGS
# ============================================================

PW = (7, 7)
NQ = 101
PIX = (512, 512)

WORKERS = max(
    1,
    min(8, (os.cpu_count() or 2) // 2)
)

USE_CACHE = True
CACHE_FILE = "qmap_cache_Bx_rotated_real_display_flat_top.npz"


# ============================================================
# 3. PAPER PARAMETERS
# ============================================================

# Inverse photonic crystal: air holes in silicon
a_nm = 445.0
h_nm = 230.0
r = 0.405

energy_eV = 0.800
hc_eV_nm = 1239.841984

# Normalize all lengths by a
wavelength = (hc_eV_nm / energy_eV) / a_nm
h = h_nm / a_nm

eps_air = 1.0
eps_si = 3.48**2

# Corrected passive-loss sign for Khepri's exp(-i*kz*z) convention
eps_sub = 1.45**2 - 0.001j

# x-oriented electric dipole
current = (1.0, 0.0, 0.0)


# ============================================================
# 4. REAL-SPACE PHYSICAL LATTICE
# ============================================================

# Rotated triangular lattice used for the physical calculation
a1 = np.array([
    np.sqrt(3.0) / 2.0,
    +0.5,
])

a2 = np.array([
    np.sqrt(3.0) / 2.0,
    -0.5,
])

lattice = np.array([
    a1,
    a2,
])

# B-point assumption used in this reproduction
B = (a1 + a2) / 2.0

# Dipole is in the slab mid-plane z=h/2
dipole_pos = (
    float(B[0]),
    float(B[1]),
    h / 2.0,
)

# Check that B lies in silicon and not inside an air hole
nearest_hole_distance = min(
    np.linalg.norm(
        B - (m * a1 + n * a2)
    )
    for m in range(-2, 3)
    for n in range(-2, 3)
)

if nearest_hole_distance <= r:
    raise RuntimeError(
        "Chosen B point lies inside an air hole."
    )


# ============================================================
# 5. TRUE RECIPROCAL LATTICE
# ============================================================

def reciprocal(u, v):
    """
    Return reciprocal primitive vectors b1,b2 satisfying
        ai . bj = 2*pi*delta_ij
    """
    det = (
        u[0] * v[1]
        -
        u[1] * v[0]
    )

    b1 = (
        2.0 * np.pi / det
    ) * np.array([
        v[1],
        -v[0],
    ])

    b2 = (
        2.0 * np.pi / det
    ) * np.array([
        -u[1],
        u[0],
    ])

    return b1, b2


b1, b2 = reciprocal(
    a1,
    a2,
)

# Six nearest reciprocal lattice vectors
Gs = np.array([
    +b1,
    -b1,
    +b2,
    -b2,
    +(b1 + b2),
    -(b1 + b2),
])


# ============================================================
# 6. TRUE FIRST BRILLOUIN ZONE IN PHYSICAL q COORDINATES
# ============================================================

# For this real-space orientation, the true FBZ is point-top.
BZ_phys = np.array([
    [0.0, +4.0 * np.pi / 3.0],
    [-2.0 * np.pi / np.sqrt(3.0), +2.0 * np.pi / 3.0],
    [-2.0 * np.pi / np.sqrt(3.0), -2.0 * np.pi / 3.0],
    [0.0, -4.0 * np.pi / 3.0],
    [+2.0 * np.pi / np.sqrt(3.0), -2.0 * np.pi / 3.0],
    [+2.0 * np.pi / np.sqrt(3.0), +2.0 * np.pi / 3.0],
])

BZ_phys_closed = np.vstack([
    BZ_phys,
    BZ_phys[0],
])

# High-symmetry points in the PHYSICAL reciprocal coordinates
Gamma_phys = np.array([
    0.0,
    0.0,
])

K_phys = np.array([
    0.0,
    4.0 * np.pi / 3.0,
])

M_phys = np.array([
    -np.pi / np.sqrt(3.0),
    np.pi,
])


def in_bz(qx, qy):
    """
    Wigner-Seitz test for the first Brillouin zone.
    """
    q = np.array([
        qx,
        qy,
    ])

    return np.all(
        Gs @ q
        <=
        0.5 * np.sum(Gs * Gs, axis=1)
        + 1e-12
    )


# ============================================================
# 7. RECIPROCAL-SPACE DISPLAY ROTATION ONLY
# ============================================================

def q_to_plot(qx, qy):
    """
    Rotate reciprocal-space coordinates by -90 degrees ONLY FOR DISPLAY:

        qx_plot =  qy_phys
        qy_plot = -qx_phys

    RCWA calculations are NOT performed in these rotated coordinates.
    """
    return (
        np.asarray(qy),
        -np.asarray(qx),
    )


Gamma_plot = np.array(
    q_to_plot(*Gamma_phys)
)

M_plot = np.array(
    q_to_plot(*M_phys)
)

K_plot = np.array(
    q_to_plot(*K_phys)
)


# ============================================================
# 8. BUILD KHEPRI STRUCTURE
# ============================================================

def build_crystal():
    """
    Build inverse triangular PhC slab:
    air holes in silicon on silica substrate.
    """

    crystal = Crystal(
        pw=PW,
        lattice=lattice,
        lattice_pitch=1.0,
        epsi=eps_air,
        epse=eps_sub,
    )

    drawing = Drawing(
        PIX,
        eps_si,
        crystal.lattice,
    )

    # Air hole at the Bravais-lattice point
    drawing.disc(
        (0.0, 0.0),
        r,
        eps_air,
    )

    pattern = drawing.canvas()

    # Split slab at z=h/2, where the dipole is located
    crystal.add_layer_pixmap(
        "L",
        pattern,
        h / 2.0,
    )

    crystal.add_layer_pixmap(
        "R",
        pattern,
        h / 2.0,
    )

    crystal.set_device(
        ["L", "R"],
        fields_mask=[
            True,
            False,
        ],
    )

    return crystal


# ============================================================
# 9. q-RESOLVED NEAR-FIELD EMISSIVITY / POWER
# ============================================================

def power(crystal, qx, qy, details=False):
    """
    Compute q-resolved total near-field emitted power for the B-point dipole.
    """

    return dipole_power_q(
        crystal=crystal,
        source_after="L",
        wavelength=wavelength,
        kp=(
            float(qx),
            float(qy),
        ),
        position=dipole_pos,
        current=current,
        backend="dense",
        return_details=details,
    )


# ============================================================
# 10. MULTIPROCESSING HELPERS
# ============================================================

_worker_crystal = None


def init_worker():
    """
    Each process receives its own Crystal object because Crystal is stateful.
    """
    global _worker_crystal

    _worker_crystal = build_crystal()


def worker(task):
    """
    task = (iy, ix, qx, qy)
    """

    iy, ix, qx, qy = task

    result = power(
        _worker_crystal,
        qx,
        qy,
    )

    return (
        iy,
        ix,
        float(
            result["P_total"]
        ),
    )


# ============================================================
# 11. VERIFY P(q)=P(-q) BEFORE USING SYMMETRY SPEEDUP
# ============================================================

def reciprocity_ok(crystal):
    """
    Verify inversion symmetry of the q-resolved emitted power.
    """

    tests = [
        (0.4, 0.8),
        (-1.0, 1.0),
        (1.1, -0.7),
    ]

    ok = True

    print("\nReciprocity check:")

    for qx, qy in tests:

        if not in_bz(
            qx,
            qy,
        ):
            continue

        p_plus = power(
            crystal,
            qx,
            qy,
        )["P_total"]

        p_minus = power(
            crystal,
            -qx,
            -qy,
        )["P_total"]

        relative_difference = (
            abs(
                p_plus
                -
                p_minus
            )
            /
            max(
                abs(p_plus),
                abs(p_minus),
                1e-14,
            )
        )

        passed = (
            relative_difference
            <
            2e-5
        )

        print(
            f"  q=({qx:+.2f},{qy:+.2f})  "
            f"rel.diff={relative_difference:.2e}  "
            f"{'PASS' if passed else 'FAIL'}"
        )

        ok &= passed

    return ok


# ============================================================
# 12. CALCULATE q-MAP
# ============================================================

def calculate_qmap(crystal):
    """
    Calculate near-field emissivity over the TRUE physical FBZ.
    """

    # Bounding rectangle of the physical point-top FBZ
    qx_values = np.linspace(
        -2.0 * np.pi / np.sqrt(3.0),
        +2.0 * np.pi / np.sqrt(3.0),
        NQ,
    )

    qy_values = np.linspace(
        -4.0 * np.pi / 3.0,
        +4.0 * np.pi / 3.0,
        NQ,
    )

    QX, QY = np.meshgrid(
        qx_values,
        qy_values,
        indexing="xy",
    )

    # Vectorized Wigner-Seitz FBZ mask
    dots = np.stack([
        G[0] * QX
        +
        G[1] * QY
        for G in Gs
    ])

    limits = (
        0.5
        *
        np.sum(
            Gs * Gs,
            axis=1,
        )
    )[:, None, None]

    mask = np.all(
        dots
        <=
        limits
        +
        1e-12,
        axis=0,
    )

    use_symmetry = reciprocity_ok(
        crystal
    )

    tasks = []

    for iy, ix in np.argwhere(mask):

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

        # Compute only one point from each q <-> -q pair
        if use_symmetry:
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
                QX[iy, ix],
                QY[iy, ix],
            )
        )

    P = np.full(
        (NQ, NQ),
        np.nan,
        dtype=float,
    )

    print(
        f"\nFBZ points = {mask.sum()}, "
        f"actual RCWA solves = {len(tasks)}, "
        f"workers = {WORKERS}"
    )

    start = time.perf_counter()

    if WORKERS > 1:

        chunksize = max(
            1,
            len(tasks)
            //
            (WORKERS * 20),
        )

        with ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=init_worker,
        ) as executor:

            for n, (
                iy,
                ix,
                value,
            ) in enumerate(
                executor.map(
                    worker,
                    tasks,
                    chunksize=chunksize,
                ),
                start=1,
            ):

                P[
                    iy,
                    ix,
                ] = value

                if use_symmetry:

                    P[
                        NQ - 1 - iy,
                        NQ - 1 - ix,
                    ] = value

                if (
                    n
                    %
                    max(
                        1,
                        len(tasks) // 10,
                    )
                    ==
                    0
                ):
                    print(
                        f"  "
                        f"{100.0*n/len(tasks):5.1f}%"
                    )

    else:

        for (
            iy,
            ix,
            qx,
            qy,
        ) in tasks:

            value = float(
                power(
                    crystal,
                    qx,
                    qy,
                )["P_total"]
            )

            P[
                iy,
                ix,
            ] = value

            if use_symmetry:

                P[
                    NQ - 1 - iy,
                    NQ - 1 - ix,
                ] = value

    values = P[
        mask
    ]

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise RuntimeError(
            "Non-finite values found in q-map."
        )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    print(
        f"\nFinished q-map in "
        f"{elapsed:.1f} s"
    )

    print(
        f"P statistics: "
        f"min={values.min():.4e}, "
        f"max={values.max():.4e}, "
        f"mean={values.mean():.4e}"
    )

    return (
        QX,
        QY,
        mask,
        P,
    )


# ============================================================
# 13. CACHE NUMERICAL q-MAP
# ============================================================

def load_cache():
    """
    Load cached q-map only if its geometry/numerics match this script.
    """

    if not (
        USE_CACHE
        and
        os.path.exists(
            CACHE_FILE
        )
    ):
        return None

    data = np.load(
        CACHE_FILE
    )

    valid = (
        int(data["NQ"])
        ==
        NQ

        and
        np.array_equal(
            data["PW"],
            np.asarray(PW),
        )

        and
        np.allclose(
            data["a1"],
            a1,
        )

        and
        np.allclose(
            data["a2"],
            a2,
        )

        and
        np.allclose(
            data["B"],
            B,
        )

        and
        np.isclose(
            float(data["wavelength"]),
            wavelength,
        )
    )

    if valid:

        print(
            f"\nLoaded cached q-map "
            f"from {CACHE_FILE}"
        )

        return (
            data["QX"],
            data["QY"],
            data["mask"],
            data["P"],
        )

    print(
        "\nCache exists but does not "
        "match current settings. Recalculating."
    )

    return None


def save_cache(
    QX,
    QY,
    mask,
    P,
):
    """
    Save numerical data only. No image is saved.
    """

    np.savez(
        CACHE_FILE,
        QX=QX,
        QY=QY,
        mask=mask,
        P=P,
        NQ=NQ,
        PW=np.asarray(PW),
        a1=a1,
        a2=a2,
        B=B,
        wavelength=wavelength,
    )

    print(
        f"Cached numerical q-map -> "
        f"{CACHE_FILE}"
    )


# ============================================================
# 14. REAL-SPACE FIGURE
# ============================================================

def show_real_space():
    """
    Display physical real-space lattice first:
      - air holes
      - primitive unit cell
      - B point
      - x-oriented dipole
    """

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(6.5, 6.0)
    )

    # Nearby air holes
    for m in range(-2, 3):
        for n in range(-2, 3):

            center = (
                m * a1
                +
                n * a2
            )

            hole = plt.Circle(
                center,
                r,
                fill=False,
                linewidth=1.1,
            )

            ax.add_patch(
                hole
            )

    # Primitive unit cell
    cell = np.array([
        [0.0, 0.0],
        a1,
        a1 + a2,
        a2,
        [0.0, 0.0],
    ])

    ax.plot(
        cell[:, 0],
        cell[:, 1],
        "k--",
        linewidth=1.5,
        label="primitive unit cell",
    )

    # B point
    ax.scatter(
        B[0],
        B[1],
        s=80,
        zorder=5,
    )

    ax.text(
        B[0],
        B[1] + 0.10,
        "B",
        ha="center",
        fontsize=13,
    )

    # x-oriented dipole arrow
    ax.annotate(
        "",
        xy=(
            B[0] + 0.23,
            B[1],
        ),
        xytext=(
            B[0] - 0.23,
            B[1],
        ),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2.0,
        ),
    )

    ax.text(
        B[0],
        B[1] - 0.15,
        "x-dipole",
        ha="center",
    )

    ax.set_xlabel(
        r"$x/a$"
    )

    ax.set_ylabel(
        r"$y/a$"
    )

    ax.set_title(
        "Real space: lattice, unit cell, and B-point dipole"
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlim(
        -1.0,
        2.5,
    )

    ax.set_ylim(
        -1.6,
        1.6,
    )

    ax.legend()

    fig.tight_layout()

    # DISPLAY ONLY
    plt.show()

    plt.close(
        fig
    )


# ============================================================
# 15. RECIPROCAL-SPACE NEAR-FIELD EMISSIVITY FIGURE
# ============================================================

def show_qmap(
    QX,
    QY,
    mask,
    P,
):
    """
    Display q-resolved near-field emissivity.

    RCWA data are calculated in physical reciprocal coordinates.
    The coordinates are rotated ONLY FOR DISPLAY.
    """

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    values = P[mask]

    # Avoid log10(0)
    positive = values[values > 0]
    floor = positive.min()

    log_emissivity = np.log10(
        np.maximum(values, floor)
    )
        

    # Rotate ONLY displayed reciprocal coordinates
    Xplot, Yplot = q_to_plot(
        QX[mask],
        QY[mask],
    )

    Xplot = (
        Xplot
        /
        (2.0 * np.pi)
    )

    Yplot = (
        Yplot
        /
        (2.0 * np.pi)
    )

    triangulation = mtri.Triangulation(
        Xplot,
        Yplot,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 6.0)
    )

    levels = np.linspace(
    log_emissivity.min(),
    log_emissivity.max(),
    121
    )

    contour = ax.tricontourf(
    triangulation,
    log_emissivity,
    levels=levels,
    cmap="hot",
    )

    fig.colorbar(
    contour,
    ax=ax,
    label=r"$\log_{10}[P(\mathbf{q})]$",
    )

    # Rotate FBZ outline with the same plotting transform
    bx, by = q_to_plot(
        BZ_phys_closed[:, 0],
        BZ_phys_closed[:, 1],
    )

    ax.plot(
        bx / (2.0 * np.pi),
        by / (2.0 * np.pi),
        "w-",
        linewidth=1.5,
    )

    # High-symmetry points in displayed coordinates
    Gp = (
        Gamma_plot
        /
        (2.0 * np.pi)
    )

    Mp = (
        M_plot
        /
        (2.0 * np.pi)
    )

    Kp = (
        K_plot
        /
        (2.0 * np.pi)
    )

    ax.scatter(
        [
            Gp[0],
            Mp[0],
            Kp[0],
        ],
        [
            Gp[1],
            Mp[1],
            Kp[1],
        ],
        s=35,
        c="cyan",
        zorder=5,
    )

    ax.text(
        Gp[0] + 0.015,
        Gp[1] + 0.015,
        r"$\Gamma$",
        color="cyan",
    )

    ax.text(
        Mp[0] + 0.015,
        Mp[1] + 0.015,
        "M",
        color="cyan",
    )

    ax.text(
        Kp[0] + 0.015,
        Kp[1] + 0.015,
        "K",
        color="cyan",
    )

    ax.set_xlabel(
        r"$q_x a/(2\pi)$"
    )

    ax.set_ylabel(
        r"$q_y a/(2\pi)$"
    )

    ax.set_title(
        "Reciprocal space: near-field emissivity"
    )

    ax.set_aspect(
        "equal"
    )

    # Flat-top display bounds
    ax.set_xlim(
        -0.72,
        0.72,
    )

    ax.set_ylim(
        -0.62,
        0.62,
    )

    fig.tight_layout()

    # DISPLAY ONLY
    plt.show()

    plt.close(
        fig
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print(
        "PRB inverse triangular PhC | "
        "800 meV | B-point x-dipole"
    )

    print(
        f"pw = {PW}, "
        f"Nq = {NQ}, "
        f"workers = {WORKERS}"
    )

    print(
        f"a1 = {a1}"
    )

    print(
        f"a2 = {a2}"
    )

    print(
        f"B = {B}"
    )

    print(
        f"dipole position = "
        f"{dipole_pos}"
    )

    print(
        f"wavelength/a = "
        f"{wavelength:.9f}"
    )

    print(
        f"h/a = "
        f"{h:.9f}"
    )

    print(
        f"eps_sub = "
        f"{eps_sub}"
    )

    print(
        "\nReciprocal-space DISPLAY convention:"
    )

    print(
        "  qx_plot =  qy_phys"
    )

    print(
        "  qy_plot = -qx_phys"
    )

    print(
        f"Gamma_plot = "
        f"{Gamma_plot}"
    )

    print(
        f"M_plot = "
        f"{M_plot}"
    )

    print(
        f"K_plot = "
        f"{K_plot}"
    )

    # --------------------------------------------------------
    # STEP 1: REAL SPACE FIRST
    # --------------------------------------------------------

    show_real_space()

    # --------------------------------------------------------
    # STEP 2: LOAD OR CALCULATE NEAR-FIELD EMISSIVITY
    # --------------------------------------------------------

    cached = load_cache()

    if cached is not None:

        (
            QX,
            QY,
            mask,
            P,
        ) = cached

    else:

        crystal = build_crystal()

        # Source-jump sanity check at Gamma
        gamma_result = power(
            crystal,
            0.0,
            0.0,
            details=True,
        )

        print(
            f"\nGamma source check: "
            f"P = "
            f"{gamma_result['P_total']:.6e}, "
            f"jump error = "
            f"{gamma_result['jump_relative_error']:.2e}"
        )

        if (
            gamma_result["jump_relative_error"]
            >
            1e-8
        ):
            raise RuntimeError(
                "Dipole source-jump check failed."
            )

        (
            QX,
            QY,
            mask,
            P,
        ) = calculate_qmap(
            crystal
        )

        save_cache(
            QX,
            QY,
            mask,
            P,
        )

    # --------------------------------------------------------
    # STEP 3: RECIPROCAL-SPACE NEAR-FIELD EMISSIVITY
    # --------------------------------------------------------

    show_qmap(
        QX,
        QY,
        mask,
        P,
    )


# ============================================================
# 17. WINDOWS MULTIPROCESSING ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
