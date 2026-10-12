"""
Reproduce the vacuum baseline panels corresponding to Fig. 2(a) and Fig. 2(e)
of Dyakov et al., Phys. Rev. B 108, 155416 (2023), but plot a
dimensionless Purcell-emissivity density rather than raw Khepri flux.

Definition
----------
Let

    sx = qx / k0
    sy = qy / k0

and let P_K(qx,qy) be Khepri's raw near-source total emitted power.

We define

    rho_F(sx,sy)
        = [k0^2 / (2*pi)^2] * P_K(qx,qy) / P_ref_K

so that

    F_P = integral rho_F(sx,sy) dsx dsy.

For vacuum, the exact result is F_P = 1 for x, y, and z dipoles.

Panels
------
(a) vertical z dipole in vacuum
(e) horizontal dipole = average of x and y dipoles

Plotting
--------
- logarithmic color scale
- "hot" colormap
- raw normalized density is saved unchanged in NPZ
- non-positive values are only masked for log plotting

Run from repository root:
    python examples/reproduce_fig2ae_vacuum_normalized.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

from khepri.crystal import Crystal
from khepri.dipole import (
    dipole_power_q,
    homogeneous_reference_power,
)


# ============================================================
# PAPER / NUMERICAL PARAMETERS
# ============================================================

ENERGY_MEV = 850.0
HC_EV_NM = 1239.841984
A_NM = 625.0

WAVELENGTH_NM = HC_EV_NM / (ENERGY_MEV / 1000.0)
WAVELENGTH_OVER_A = WAVELENGTH_NM / A_NM

# Khepri normalized vacuum wave number.
K0 = 2.0 * np.pi / WAVELENGTH_OVER_A

# Homogeneous vacuum needs only the zero Fourier harmonic.
PW = (1, 1)

# Plot range in sx = qx/k0 and sy = qy/k0.
Q_OVER_K0_MAX = 1.15
NQ = 151

# The exact z position is irrelevant in uniform vacuum, but this
# retains the paper's z=80 nm value for consistency.
SOURCE_Z_OVER_A = 80.0 / A_NM
VACUUM_HALF_DEPTH = SOURCE_Z_OVER_A

OUTPUT_DIR = Path("fig2_vacuum_output")


# ============================================================
# CRYSTAL
# ============================================================

def build_vacuum_crystal():
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

    distance_to_light_line = np.abs(
        1.0 - SX**2 - SY**2
    )

    min_distance = float(
        np.min(distance_to_light_line)
    )

    if min_distance < 1e-12:
        raise RuntimeError(
            "The q grid lands too close to the exact vacuum light line. "
            "Change NQ slightly."
        )

    return (
        s,
        ds,
        SX,
        SY,
        QX,
        QY,
        min_distance,
    )


# ============================================================
# RAW KHEPRI MAP
# ============================================================

def calculate_raw_power_map(
    crystal,
    QX,
    QY,
    current,
    label,
):
    P = np.empty(
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

            result = dipole_power_q(
                crystal=crystal,
                source_after="minus_z_half",
                wavelength=WAVELENGTH_OVER_A,
                kp=(qx, qy),
                position=(
                    0.0,
                    0.0,
                    SOURCE_Z_OVER_A,
                ),
                current=current,
                backend="dense",
                return_details=False,
            )

            P[iy, ix] = float(
                np.real(
                    result["P_total"]
                )
            )

        if (
            (iy + 1) % 10 == 0
            or
            iy == QX.shape[0] - 1
        ):
            elapsed = perf_counter() - t0
            print(
                f"{label}: row {iy+1:3d}/{QX.shape[0]}  "
                f"points {counter:6d}/{total:6d}  "
                f"time={elapsed:.1f} s"
            )

    return P


# ============================================================
# DIMENSIONLESS PURCELL-EMISSIVITY DENSITY
# ============================================================

def purcell_density_from_raw_power(
    raw_power,
    current,
):
    """
    Convert Khepri raw q-resolved power into a dimensionless density
    in coordinates sx=qx/k0, sy=qy/k0:

        rho_F = k0^2/(2*pi)^2 * P_K / P_ref_K

    so that

        F_P = integral rho_F dsx dsy.
    """

    P_ref = homogeneous_reference_power(
        wavelength=WAVELENGTH_OVER_A,
        refractive_index=1.0,
        current=current,
    )

    rho_F = (
        K0**2
        /
        (2.0 * np.pi)**2
        *
        np.asarray(raw_power, dtype=float)
        /
        P_ref
    )

    return rho_F, float(P_ref)


def integrate_density_midpoint(
    density,
    ds,
):
    """
    Midpoint quadrature in sx=q_x/k0 and sy=q_y/k0.
    """

    return float(
        np.sum(density)
        *
        ds
        *
        ds
    )


# ============================================================
# LOG PLOT
# ============================================================

def log_plot_data(data):
    """
    Preserve the numerical data exactly, but mask non-positive
    entries for LogNorm plotting.
    """

    data = np.asarray(
        data,
        dtype=float,
    )

    positive = data[
        data > 0.0
    ]

    if positive.size == 0:
        raise RuntimeError(
            "No positive values are available for logarithmic plotting."
        )

    masked = np.ma.masked_less_equal(
        data,
        0.0,
    )

    vmin = float(
        np.min(positive)
    )

    vmax = float(
        np.max(positive)
    )

    return (
        masked,
        vmin,
        vmax,
    )


def plot_maps(
    s,
    rho_z,
    rho_h,
):
    extent = [
        s[0],
        s[-1],
        s[0],
        s[-1],
    ]

    phi = np.linspace(
        0.0,
        2.0 * np.pi,
        600,
    )

    light_x = np.cos(phi)
    light_y = np.sin(phi)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.7),
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            rho_z,
            "(a) vertical dipole in vacuum",
        ),
        (
            axes[1],
            rho_h,
            "(e) horizontal dipole in vacuum",
        ),
    ]

    for ax, density, title in panels:

        plotted, vmin, vmax = log_plot_data(
            density
        )

        im = ax.imshow(
            plotted,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            cmap="hot",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
        )

        ax.plot(
            light_x,
            light_y,
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
            format=LogFormatterMathtext(),
        )

        cbar.set_label(
            r"$\rho_F(q_x/k_0,q_y/k_0)$"
        )

        print(
            f"{title}: "
            f"log plot range [{vmin:.6e}, {vmax:.6e}]"
        )

    fig.suptitle(
        "Vacuum Purcell-emissivity density at 850 meV"
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("FIG. 2(a,e) VACUUM — DIMENSIONLESS PURCELL-EMISSIVITY DENSITY")
    print("=" * 78)

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
        f"k0 (Khepri normalized)  = {K0:.12f}"
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

    (
        s,
        ds,
        SX,
        SY,
        QX,
        QY,
        min_lightline_distance,
    ) = build_q_grid()

    print(
        f"grid spacing ds         = {ds:.12e}"
    )

    print(
        "minimum distance to exact light line "
        f"= {min_lightline_distance:.3e}"
    )

    crystal = build_vacuum_crystal()

    # --------------------------------------------------------
    # z dipole
    # --------------------------------------------------------

    print()
    print("Calculating z-dipole raw Khepri map ...")

    Pz_raw = calculate_raw_power_map(
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

    rho_z, Pref_z = purcell_density_from_raw_power(
        raw_power=Pz_raw,
        current=(
            0.0,
            0.0,
            1.0,
        ),
    )

    Fz = integrate_density_midpoint(
        rho_z,
        ds,
    )

    # --------------------------------------------------------
    # x/y dipoles and horizontal average
    # --------------------------------------------------------

    print()
    print("Calculating x-dipole raw Khepri map ...")

    Px_raw = calculate_raw_power_map(
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
    print("Calculating y-dipole raw Khepri map ...")

    Py_raw = calculate_raw_power_map(
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

    rho_x, Pref_x = purcell_density_from_raw_power(
        raw_power=Px_raw,
        current=(
            1.0,
            0.0,
            0.0,
        ),
    )

    rho_y, Pref_y = purcell_density_from_raw_power(
        raw_power=Py_raw,
        current=(
            0.0,
            1.0,
            0.0,
        ),
    )

    rho_h = 0.5 * (
        rho_x
        +
        rho_y
    )

    F_x = integrate_density_midpoint(
        rho_x,
        ds,
    )

    F_y = integrate_density_midpoint(
        rho_y,
        ds,
    )

    F_h = integrate_density_midpoint(
        rho_h,
        ds,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    i0 = int(
        np.argmin(
            np.abs(s)
        )
    )

    light_mask = (
        SX**2
        +
        SY**2
        <=
        1.0
    )

    outside_z = float(
        np.sum(
            np.abs(
                rho_z[~light_mask]
            )
        )
        *
        ds
        *
        ds
    )

    outside_h = float(
        np.sum(
            np.abs(
                rho_h[~light_mask]
            )
        )
        *
        ds
        *
        ds
    )

    print()
    print("=" * 78)
    print("NORMALIZATION CHECK")
    print("=" * 78)

    print(
        f"P_ref(z)                = {Pref_z:.12e}"
    )

    print(
        f"P_ref(x)                = {Pref_x:.12e}"
    )

    print(
        f"P_ref(y)                = {Pref_y:.12e}"
    )

    print()
    print(
        f"Integrated Fp(z)        = {Fz:.12f}"
    )

    print(
        f"Integrated Fp(x)        = {F_x:.12f}"
    )

    print(
        f"Integrated Fp(y)        = {F_y:.12f}"
    )

    print(
        f"Integrated Fp(horizontal average) = {F_h:.12f}"
    )

    print()
    print(
        f"z error from 1          = {100.0*abs(Fz-1.0):.6f}%"
    )

    print(
        f"x error from 1          = {100.0*abs(F_x-1.0):.6f}%"
    )

    print(
        f"y error from 1          = {100.0*abs(F_y-1.0):.6f}%"
    )

    print(
        f"horizontal error from 1 = {100.0*abs(F_h-1.0):.6f}%"
    )

    print()
    print(
        f"rho_z at grid center    = {rho_z[i0, i0]:.12e}"
    )

    print(
        f"rho_h at grid center    = {rho_h[i0, i0]:.12e}"
    )

    print(
        f"integrated |rho_z| outside light cone = {outside_z:.6e}"
    )

    print(
        f"integrated |rho_h| outside light cone = {outside_h:.6e}"
    )

    print()
    print(
        f"rho_z min/max           = "
        f"{np.min(rho_z):.6e} / {np.max(rho_z):.6e}"
    )

    print(
        f"rho_h min/max           = "
        f"{np.min(rho_h):.6e} / {np.max(rho_h):.6e}"
    )

    # --------------------------------------------------------
    # Save raw + normalized maps
    # --------------------------------------------------------

    npz_path = (
        OUTPUT_DIR
        /
        "fig2ae_vacuum_850meV_normalized.npz"
    )

    np.savez_compressed(
        npz_path,
        energy_meV=ENERGY_MEV,
        wavelength_nm=WAVELENGTH_NM,
        wavelength_over_a=WAVELENGTH_OVER_A,
        K0=K0,
        q_over_k0=s,
        ds=ds,
        SX=SX,
        SY=SY,
        QX=QX,
        QY=QY,
        Pz_raw=Pz_raw,
        Px_raw=Px_raw,
        Py_raw=Py_raw,
        rho_z=rho_z,
        rho_x=rho_x,
        rho_y=rho_y,
        rho_horizontal=rho_h,
        Pref_z=Pref_z,
        Pref_x=Pref_x,
        Pref_y=Pref_y,
        Fp_z=Fz,
        Fp_x=F_x,
        Fp_y=F_y,
        Fp_horizontal=F_h,
    )

    fig = plot_maps(
        s=s,
        rho_z=rho_z,
        rho_h=rho_h,
    )

    png_path = (
        OUTPUT_DIR
        /
        "fig2ae_vacuum_850meV_normalized_log_hot.png"
    )

    pdf_path = (
        OUTPUT_DIR
        /
        "fig2ae_vacuum_850meV_normalized_log_hot.pdf"
    )

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
    print("=" * 78)
    print("FINISHED")
    print("=" * 78)

    print(
        f"NPZ : {npz_path}"
    )

    print(
        f"PNG : {png_path}"
    )

    print(
        f"PDF : {pdf_path}"
    )

    print()
    print(
        "For a correct vacuum calculation, all integrated Fp values "
        "should be close to 1."
    )

    plt.show()


if __name__ == "__main__":
    main()
