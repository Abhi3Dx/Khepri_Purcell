"""
Reproduce Fig. 2(a) and Fig. 2(e) of Dyakov et al.,
Phys. Rev. B 108, 155416 (2023): dipole near-field emissivity in vacuum.

Panels reproduced
-----------------
Fig. 2(a): vertical z-oriented electric dipole in vacuum.
Fig. 2(e): horizontal dipole in vacuum, defined in the paper as
            the average of x- and y-oriented dipoles:
                f_parallel = (f_x + f_y)/2

Paper frequency
---------------
hbar*omega = 850 meV

Notes
-----
- This script uses the actual Khepri dipole source implementation.
- The vacuum source plane is represented by two artificial vacuum layers
  with identical epsilon=1. This does not create a physical interface.
- The map is plotted versus qx/k0 and qy/k0.
- The dashed circle is the vacuum light cone:
      sqrt(qx^2 + qy^2) / k0 = 1
- Each panel is normalized independently before taking log10, consistent
  with the paper's statement that the color scales differ between panels.
- Raw unnormalized power maps are also saved to NPZ.

Run from the repository root:
    python examples/reproduce_fig2ae_vacuum.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from khepri.crystal import Crystal
from khepri.dipole import dipole_power_q


# ============================================================
# PAPER / NUMERICAL PARAMETERS
# ============================================================

ENERGY_MEV = 850.0
HC_EV_NM = 1239.841984

# Fig. 2 uses a = 625 nm.  Vacuum itself is non-periodic, but using
# this normalization keeps q and wavelength units consistent with
# the later Fig. 2(b-h) calculations.
A_NM = 625.0

WAVELENGTH_NM = HC_EV_NM / (ENERGY_MEV / 1000.0)
WAVELENGTH_OVER_A = WAVELENGTH_NM / A_NM

K0 = 2.0 * np.pi / WAVELENGTH_OVER_A

# Only the zero Fourier order is needed in homogeneous vacuum.
PW = (1, 1)

# q/k0 map range.
Q_OVER_K0_MAX = 1.15

# Midpoint grid. 151 avoids sampling the exact vacuum light line
# for this chosen range.
NQ = 151

# Artificial source-plane split in vacuum.
# Its precise value is physically irrelevant because all media are epsilon=1.
SOURCE_PLANE_Z_OVER_A = 80.0 / A_NM
VACUUM_HALF_DEPTH = SOURCE_PLANE_Z_OVER_A

OUTPUT_DIR = Path("fig2_vacuum_output")


# ============================================================
# CRYSTAL
# ============================================================

def build_vacuum_crystal():
    """
    Homogeneous vacuum everywhere, artificially split at the source plane.
    """

    crystal = Crystal(
        pw=PW,
        lattice="square",
        lattice_pitch=1.0,
        epsi=1.0,
        epse=1.0,
    )

    crystal.add_layer_uniform(
        "minus_z_half",
        epsilon=1.0,
        depth=VACUUM_HALF_DEPTH,
    )

    crystal.add_layer_uniform(
        "plus_z_half",
        epsilon=1.0,
        depth=VACUUM_HALF_DEPTH,
    )

    crystal.set_device(
        [
            "minus_z_half",
            "plus_z_half",
        ],
        fields_mask=[
            True,
            False,
        ],
    )

    return crystal


# ============================================================
# q GRID
# ============================================================

def build_q_grid():

    ds = 2.0 * Q_OVER_K0_MAX / NQ

    s = (
        -Q_OVER_K0_MAX
        +
        (np.arange(NQ) + 0.5) * ds
    )

    SX, SY = np.meshgrid(
        s,
        s,
        indexing="xy",
    )

    QX = K0 * SX
    QY = K0 * SY

    # Safety check against exact grazing incidence.
    distance_to_light_line = np.abs(
        1.0
        -
        SX**2
        -
        SY**2
    )

    min_distance = float(
        np.min(distance_to_light_line)
    )

    if min_distance < 1e-12:
        raise RuntimeError(
            "The chosen q grid hits the exact vacuum light line. "
            "Change NQ slightly."
        )

    return s, SX, SY, QX, QY, min_distance


# ============================================================
# SINGLE MAP
# ============================================================

def calculate_map(
    crystal,
    QX,
    QY,
    current,
    label,
):

    result = np.empty(
        QX.shape,
        dtype=float,
    )

    total = QX.size
    counter = 0

    t0 = perf_counter()

    for iy in range(QX.shape[0]):

        for ix in range(QX.shape[1]):

            counter += 1

            qx = float(QX[iy, ix])
            qy = float(QY[iy, ix])

            power = dipole_power_q(
                crystal=crystal,
                source_after="minus_z_half",
                wavelength=WAVELENGTH_OVER_A,
                kp=(qx, qy),
                position=(
                    0.0,
                    0.0,
                    SOURCE_PLANE_Z_OVER_A,
                ),
                current=current,
                backend="dense",
                return_details=False,
            )["P_total"]

            result[iy, ix] = float(
                np.real(power)
            )

        if (iy + 1) % 10 == 0 or iy == QX.shape[0] - 1:
            elapsed = perf_counter() - t0
            print(
                f"{label}: row {iy+1:3d}/{QX.shape[0]}  "
                f"points {counter:6d}/{total:6d}  "
                f"time={elapsed:.1f} s"
            )

    return result


# ============================================================
# PLOTTING
# ============================================================

def normalized_log_map(P):
    """
    Independent logarithmic normalization for visual comparison
    with the paper.

    Returns log10(P / Pmax), clipped to [-4, 0].
    """

    P = np.asarray(
        P,
        dtype=float,
    )

    # Numerical roundoff can produce tiny negative values near zero.
    Pclean = np.maximum(
        P,
        0.0,
    )

    pmax = float(
        np.max(Pclean)
    )

    if pmax <= 0.0:
        raise RuntimeError(
            "Power map contains no positive values."
        )

    floor = pmax * 1e-12

    logmap = np.log10(
        np.maximum(Pclean, floor)
        /
        pmax
    )

    return np.clip(
        logmap,
        -4.0,
        0.0,
    )


def plot_fig2ae(
    s,
    Pz,
    Phorizontal,
):

    Zz = normalized_log_map(Pz)
    Zh = normalized_log_map(Phorizontal)

    extent = [
        s[0],
        s[-1],
        s[0],
        s[-1],
    ]

    phi = np.linspace(
        0.0,
        2.0*np.pi,
        500,
    )

    circle_x = np.cos(phi)
    circle_y = np.sin(phi)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.7),
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            Zz,
            "(a) vertical dipole in vacuum",
        ),
        (
            axes[1],
            Zh,
            "(e) horizontal dipole in vacuum",
        ),
    ]

    for ax, data, title in panels:

        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            vmin=-4.0,
            vmax=0.0,
        )

        ax.plot(
            circle_x,
            circle_y,
            "--",
            linewidth=1.2,
            label="vacuum light line",
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            r"$q_x/k_0$"
        )

        ax.set_ylabel(
            r"$q_y/k_0$"
        )

        ax.set_xlim(
            -Q_OVER_K0_MAX,
            +Q_OVER_K0_MAX,
        )

        ax.set_ylim(
            -Q_OVER_K0_MAX,
            +Q_OVER_K0_MAX,
        )

        ax.set_aspect(
            "equal"
        )

        ax.legend(
            loc="upper right",
            fontsize=8,
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            shrink=0.87,
        )

        cbar.set_label(
            r"$\log_{10}[f(\mathbf{q})/f_{\max}]$"
        )

    fig.suptitle(
        "Dyakov et al. Fig. 2(a,e) baseline — 850 meV"
    )

    return fig


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(
    s,
    Pz,
    Px,
    Py,
    Phorizontal,
):

    # Find grid point closest to q=0.
    i0 = int(
        np.argmin(
            np.abs(s)
        )
    )

    center_z = Pz[
        i0,
        i0,
    ]

    center_h = Phorizontal[
        i0,
        i0,
    ]

    xy_difference = np.max(
        np.abs(
            Px
            -
            Py.T
        )
    )

    xy_scale = max(
        np.max(np.abs(Px)),
        np.max(np.abs(Py)),
        1e-30,
    )

    xy_relative = (
        xy_difference
        /
        xy_scale
    )

    print()
    print("="*78)
    print("DIAGNOSTICS")
    print("="*78)

    print(
        f"closest q/k0 to origin = "
        f"({s[i0]:+.6e}, {s[i0]:+.6e})"
    )

    print(
        f"Pz near q=0             = {center_z:.12e}"
    )

    print(
        f"Phorizontal near q=0    = {center_h:.12e}"
    )

    print(
        "x/y 90-degree symmetry check "
        f"max relative difference = {xy_relative:.3e}"
    )

    print(
        f"max(Pz)                 = {np.max(Pz):.12e}"
    )

    print(
        f"max(Px)                 = {np.max(Px):.12e}"
    )

    print(
        f"max(Py)                 = {np.max(Py):.12e}"
    )

    print(
        f"max(Phorizontal)        = {np.max(Phorizontal):.12e}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("="*78)
    print("REPRODUCE FIG. 2(a) AND FIG. 2(e): VACUUM")
    print("="*78)

    print(
        f"energy                  = {ENERGY_MEV:.3f} meV"
    )

    print(
        f"wavelength              = {WAVELENGTH_NM:.6f} nm"
    )

    print(
        f"a                       = {A_NM:.3f} nm"
    )

    print(
        f"wavelength/a            = {WAVELENGTH_OVER_A:.12f}"
    )

    print(
        f"k0 (normalized)         = {K0:.12f}"
    )

    print(
        f"PW                      = {PW}"
    )

    print(
        f"NQ                      = {NQ}"
    )

    print(
        f"q/k0 range              = "
        f"[-{Q_OVER_K0_MAX}, +{Q_OVER_K0_MAX}]"
    )

    print(
        f"source z/a              = {SOURCE_PLANE_Z_OVER_A:.12f}"
    )

    (
        s,
        SX,
        SY,
        QX,
        QY,
        min_lightline_distance,
    ) = build_q_grid()

    print(
        "minimum |1-(qx/k0)^2-(qy/k0)^2| "
        f"= {min_lightline_distance:.3e}"
    )

    crystal = build_vacuum_crystal()

    print()
    print("Calculating Fig. 2(a): z dipole ...")

    Pz = calculate_map(
        crystal=crystal,
        QX=QX,
        QY=QY,
        current=(
            0.0,
            0.0,
            1.0,
        ),
        label="Jz",
    )

    print()
    print("Calculating x part of Fig. 2(e) ...")

    Px = calculate_map(
        crystal=crystal,
        QX=QX,
        QY=QY,
        current=(
            1.0,
            0.0,
            0.0,
        ),
        label="Jx",
    )

    print()
    print("Calculating y part of Fig. 2(e) ...")

    Py = calculate_map(
        crystal=crystal,
        QX=QX,
        QY=QY,
        current=(
            0.0,
            1.0,
            0.0,
        ),
        label="Jy",
    )

    Phorizontal = 0.5 * (
        Px
        +
        Py
    )

    print_diagnostics(
        s=s,
        Pz=Pz,
        Px=Px,
        Py=Py,
        Phorizontal=Phorizontal,
    )

    npz_path = OUTPUT_DIR / "fig2ae_vacuum_850meV_raw.npz"

    np.savez_compressed(
        npz_path,
        energy_meV=ENERGY_MEV,
        wavelength_nm=WAVELENGTH_NM,
        wavelength_over_a=WAVELENGTH_OVER_A,
        q_over_k0=s,
        SX=SX,
        SY=SY,
        QX=QX,
        QY=QY,
        Pz=Pz,
        Px=Px,
        Py=Py,
        Phorizontal=Phorizontal,
    )

    fig = plot_fig2ae(
        s=s,
        Pz=Pz,
        Phorizontal=Phorizontal,
    )

    png_path = OUTPUT_DIR / "fig2ae_vacuum_850meV.png"
    pdf_path = OUTPUT_DIR / "fig2ae_vacuum_850meV.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print()
    print("="*78)
    print("FINISHED")
    print("="*78)

    print(
        f"raw data: {npz_path}"
    )

    print(
        f"PNG     : {png_path}"
    )

    print(
        f"PDF     : {pdf_path}"
    )

    print()
    print(
        "Expected qualitative result:"
    )

    print(
        "  Fig. 2(a): circular z-dipole map with a dark center."
    )

    print(
        "  Fig. 2(e): circular horizontal-dipole map."
    )

    print(
        "  Both maps should be confined to the vacuum light cone."
    )

    plt.show()


if __name__ == "__main__":
    main()
