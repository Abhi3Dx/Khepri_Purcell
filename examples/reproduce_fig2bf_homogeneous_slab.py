"""
Reproduce Fig. 2(b) and Fig. 2(f) of Dyakov et al.,
Phys. Rev. B 108, 155416 (2023):

    (b) vertical z dipole in a homogeneous Si slab on SiO2
    (f) horizontal dipole average in the same slab

The paper states that these homogeneous-slab maps are axially symmetric,
so this script computes a dense 1D radial q scan and reconstructs the 2D
map from that scan. This is much faster and resolves the guided-mode rings
better than a coarse 2D brute-force grid.

IMPORTANT KHEPRI CONVENTIONS
----------------------------
Khepri stack order is:

    incidence side  ->  emergence side
    air             ->  Si slab       ->  SiO2 substrate

The paper's z coordinate is measured upward from the Si/SiO2 interface.
For h = 218 nm and z0 = 80 nm:

    air-side Si thickness      = h - z0 = 138 nm
    substrate-side Si thickness = z0    = 80 nm

So the source plane is inserted between those two Si pieces.

For the small substrate loss used by the paper for total-Purcell
calculations, the paper writes +0.001 i in its convention. Khepri uses
the opposite time-harmonic sign convention, so here we use:

    eps_SiO2 = n_SiO2^2 - 0.001j

The Si source layer remains lossless.

NORMALIZATION
-------------
The plotted quantity is the dimensionless Purcell-emissivity density

    rho_F(sx,sy)
        = [k0^2/(2*pi)^2] * P_Khepri(qx,qy) / P_ref,Khepri

with

    sx = qx/k0
    sy = qy/k0

and the reference medium is homogeneous silicon:

    P_ref,Khepri = k0^2*n_Si/(6*pi) * |J|^2

so that

    F_P = integral rho_F dsx dsy.

PLOTTING
--------
- logarithmic color scale
- "hot" colormap
- vacuum light line: q/k0 = 1
- substrate light line: q/k0 = n_SiO2

Run from the repository root:

    python examples/reproduce_fig2bf_homogeneous_slab.py
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
# PAPER PARAMETERS
# ============================================================

ENERGY_MEV = 850.0
HC_EV_NM = 1239.841984

A_NM = 625.0
H_NM = 218.0
Z0_FROM_SUBSTRATE_NM = 80.0

WAVELENGTH_NM = HC_EV_NM / (ENERGY_MEV / 1000.0)

# Khepri uses normalized geometric units here: a = 1.
WAVELENGTH = WAVELENGTH_NM / A_NM
K0 = 2.0 * np.pi / WAVELENGTH

H = H_NM / A_NM
Z0_FROM_SUBSTRATE = Z0_FROM_SUBSTRATE_NM / A_NM

# Khepri stack goes air -> substrate.
DEPTH_AIR_TO_SOURCE = (H_NM - Z0_FROM_SUBSTRATE_NM) / A_NM
DEPTH_SOURCE_TO_SUBSTRATE = Z0_FROM_SUBSTRATE_NM / A_NM


# ============================================================
# MATERIALS
# ============================================================

# These are reasonable near-1.46-um values.
# If you later identify the exact optical constants used by the
# authors, change only these two numbers.
N_SI = 3.48
N_SIO2 = 1.45

EPS_SI = N_SI**2

# Small substrate loss for finite-width guided-mode peaks.
# Khepri passive-loss convention: negative imaginary epsilon.
EPS_SIO2 = N_SIO2**2 - 0.001j

EPS_AIR = 1.0


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

# Homogeneous slab: only G=0 is needed.
PW = (1, 1)

# Fig. 2(b,f) must extend beyond the substrate light line to show
# guided slab modes. Silicon has n ~ 3.48, so 0...3.7 captures the
# relevant guided region.
Q_OVER_K0_MAX = 3.70

# Dense radial sampling is important because the guided-mode rings
# can be very narrow.
N_RADIAL = 4001

# 2D display grid reconstructed from radial symmetry.
N_PLOT = 501

# Show at most this many decades in each log-color panel.
# The numerical values themselves are NOT renormalized.
LOG_DECADES = 5.0

OUTPUT_DIR = Path("fig2_homogeneous_slab_output")


# ============================================================
# STRUCTURE
# ============================================================

def build_homogeneous_slab():
    """
    Air / Si(138 nm) / source / Si(80 nm) / SiO2.
    """

    crystal = Crystal(
        pw=PW,
        lattice="square",
        lattice_pitch=1.0,
        epsi=EPS_AIR,
        epse=EPS_SIO2,
    )

    crystal.add_layer_uniform(
        "si_air_side",
        epsilon=EPS_SI,
        depth=DEPTH_AIR_TO_SOURCE,
    )

    crystal.add_layer_uniform(
        "si_substrate_side",
        epsilon=EPS_SI,
        depth=DEPTH_SOURCE_TO_SUBSTRATE,
    )

    # The Jz implementation needs source-layer IC = 1/eps_Si.
    # Retaining the air-side Si piece provides that operator and
    # the partial S matrix at the source plane.
    crystal.set_device(
        [
            "si_air_side",
            "si_substrate_side",
        ],
        fields_mask=[
            True,
            False,
        ],
    )

    return crystal


# ============================================================
# SINGLE q POINT
# ============================================================

def power_at_q(
    crystal,
    q_over_k0,
    current,
):
    qx = float(q_over_k0 * K0)

    result = dipole_power_q(
        crystal=crystal,
        source_after="si_air_side",
        wavelength=WAVELENGTH,
        kp=(qx, 0.0),
        position=(
            0.0,
            0.0,
            DEPTH_AIR_TO_SOURCE,
        ),
        current=current,
        backend="dense",
        return_details=False,
    )

    return float(
        np.real(
            result["P_total"]
        )
    )


# ============================================================
# RADIAL SCANS
# ============================================================

def radial_scan(
    crystal,
    s,
    current,
    label,
):
    result = np.empty_like(
        s,
        dtype=float,
    )

    t0 = perf_counter()

    for i, si in enumerate(s):
        result[i] = power_at_q(
            crystal=crystal,
            q_over_k0=si,
            current=current,
        )

        if (
            (i + 1) % 250 == 0
            or
            i == len(s) - 1
        ):
            print(
                f"{label}: {i+1:5d}/{len(s):5d}  "
                f"time={perf_counter()-t0:.1f} s"
            )

    return result


# ============================================================
# PURCELL-DENSITY NORMALIZATION
# ============================================================

def to_purcell_density(
    raw_power,
    current,
):
    P_ref = homogeneous_reference_power(
        wavelength=WAVELENGTH,
        refractive_index=N_SI,
        current=current,
    )

    rho = (
        K0**2
        /
        (2.0 * np.pi)**2
        *
        np.asarray(
            raw_power,
            dtype=float,
        )
        /
        P_ref
    )

    return (
        rho,
        float(P_ref),
    )


# ============================================================
# AXIAL-SYMMETRY RECONSTRUCTION
# ============================================================

def reconstruct_2d_from_radial(
    s_radial,
    rho_radial,
):
    axis = np.linspace(
        -Q_OVER_K0_MAX,
        +Q_OVER_K0_MAX,
        N_PLOT,
    )

    SX, SY = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    R = np.sqrt(
        SX**2
        +
        SY**2
    )

    # Values outside the scanned radial range are set to NaN.
    map2d = np.interp(
        R.ravel(),
        s_radial,
        rho_radial,
        left=np.nan,
        right=np.nan,
    ).reshape(
        R.shape
    )

    return (
        axis,
        SX,
        SY,
        map2d,
    )


# ============================================================
# LOG COLOR RANGE
# ============================================================

def log_limits(data):
    positive = np.asarray(data)[
        np.isfinite(data)
        &
        (np.asarray(data) > 0.0)
    ]

    if positive.size == 0:
        raise RuntimeError(
            "No positive values available for logarithmic plotting."
        )

    vmax = float(
        np.max(positive)
    )

    vmin = max(
        float(np.min(positive)),
        vmax * 10.0**(-LOG_DECADES),
    )

    return (
        vmin,
        vmax,
    )


# ============================================================
# 2D FIGURE
# ============================================================

def plot_2d_maps(
    axis,
    rho_z_map,
    rho_h_map,
):
    extent = [
        axis[0],
        axis[-1],
        axis[0],
        axis[-1],
    ]

    phi = np.linspace(
        0.0,
        2.0*np.pi,
        800,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.9),
        constrained_layout=True,
    )

    panels = [
        (
            axes[0],
            rho_z_map,
            "(b) vertical dipole — homogeneous Si slab",
        ),
        (
            axes[1],
            rho_h_map,
            "(f) horizontal dipole — homogeneous Si slab",
        ),
    ]

    for ax, data, title in panels:
        vmin, vmax = log_limits(
            data
        )

        masked = np.ma.masked_invalid(
            data
        )

        masked = np.ma.masked_less_equal(
            masked,
            0.0,
        )

        im = ax.imshow(
            masked,
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

        # Vacuum light line.
        ax.plot(
            np.cos(phi),
            np.sin(phi),
            "--",
            linewidth=1.2,
            label="vacuum light line",
        )

        # Substrate light line.
        ax.plot(
            N_SIO2*np.cos(phi),
            N_SIO2*np.sin(phi),
            "--",
            linewidth=1.2,
            label="SiO$_2$ light line",
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
            f"{title}: log range "
            f"[{vmin:.6e}, {vmax:.6e}]"
        )

    fig.suptitle(
        "Homogeneous Si slab on SiO$_2$ — 850 meV"
    )

    return fig


# ============================================================
# RADIAL-CUT FIGURE
# ============================================================

def plot_radial_cuts(
    s,
    rho_z,
    rho_h,
):
    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.semilogy(
        s,
        rho_z,
        label="vertical z dipole",
    )

    ax.semilogy(
        s,
        rho_h,
        label="horizontal x/y average",
    )

    ax.axvline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="vacuum light line",
    )

    ax.axvline(
        N_SIO2,
        linestyle="--",
        linewidth=1.0,
        label="SiO$_2$ light line",
    )

    ax.set_xlabel(
        r"$q_\parallel/k_0$"
    )

    ax.set_ylabel(
        r"$\rho_F$"
    )

    ax.set_xlim(
        0.0,
        Q_OVER_K0_MAX,
    )

    ax.set_title(
        "Radial near-field emissivity"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ============================================================
# RADIAL INTEGRAL
# ============================================================

def radial_purcell_integral(
    s,
    rho,
):
    """
    Because the homogeneous slab is axially symmetric:

        Fp = 2*pi * integral rho(s) * s ds
    """

    return float(
        2.0
        *
        np.pi
        *
        np.trapezoid(
            rho * s,
            s,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("FIG. 2(b,f): HOMOGENEOUS Si SLAB ON SiO2")
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
        f"h                       = {H_NM:.3f} nm"
    )

    print(
        f"paper z0 from substrate = {Z0_FROM_SUBSTRATE_NM:.3f} nm"
    )

    print(
        f"air -> source depth     = "
        f"{H_NM-Z0_FROM_SUBSTRATE_NM:.3f} nm"
    )

    print(
        f"source -> substrate     = "
        f"{Z0_FROM_SUBSTRATE_NM:.3f} nm"
    )

    print(
        f"n_Si                    = {N_SI}"
    )

    print(
        f"n_SiO2                  = {N_SIO2}"
    )

    print(
        f"eps_Si                  = {EPS_SI}"
    )

    print(
        f"eps_SiO2 (Khepri)       = {EPS_SIO2}"
    )

    print(
        f"q/k0 scan               = 0 ... {Q_OVER_K0_MAX}"
    )

    print(
        f"N radial                = {N_RADIAL}"
    )

    print()

    crystal = build_homogeneous_slab()

    # Avoid landing exactly on the light lines.
    ds = Q_OVER_K0_MAX / N_RADIAL

    s = (
        np.arange(N_RADIAL)
        +
        0.5
    ) * ds

    # --------------------------------------------------------
    # Vertical dipole
    # --------------------------------------------------------

    print("Calculating vertical Jz radial scan ...")

    Pz = radial_scan(
        crystal=crystal,
        s=s,
        current=(
            0.0,
            0.0,
            1.0,
        ),
        label="Jz",
    )

    rho_z, Pref_z = to_purcell_density(
        raw_power=Pz,
        current=(
            0.0,
            0.0,
            1.0,
        ),
    )

    # --------------------------------------------------------
    # Horizontal x/y average
    # --------------------------------------------------------

    print()
    print("Calculating Jx radial scan ...")

    Px = radial_scan(
        crystal=crystal,
        s=s,
        current=(
            1.0,
            0.0,
            0.0,
        ),
        label="Jx",
    )

    print()
    print("Calculating Jy radial scan ...")

    Py = radial_scan(
        crystal=crystal,
        s=s,
        current=(
            0.0,
            1.0,
            0.0,
        ),
        label="Jy",
    )

    rho_x, Pref_x = to_purcell_density(
        raw_power=Px,
        current=(
            1.0,
            0.0,
            0.0,
        ),
    )

    rho_y, Pref_y = to_purcell_density(
        raw_power=Py,
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

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    Fz_truncated = radial_purcell_integral(
        s,
        rho_z,
    )

    Fh_truncated = radial_purcell_integral(
        s,
        rho_h,
    )

    print()
    print("=" * 78)
    print("DIAGNOSTICS")
    print("=" * 78)

    print(
        f"P_ref(z, bulk Si)       = {Pref_z:.12e}"
    )

    print(
        f"P_ref(x, bulk Si)       = {Pref_x:.12e}"
    )

    print(
        f"P_ref(y, bulk Si)       = {Pref_y:.12e}"
    )

    print()
    print(
        f"max rho_z               = {np.max(rho_z):.12e}"
    )

    print(
        f"max rho_horizontal      = {np.max(rho_h):.12e}"
    )

    print()
    print(
        f"truncated radial Fp(z)  = {Fz_truncated:.8f}"
    )

    print(
        f"truncated radial Fp(h)  = {Fh_truncated:.8f}"
    )

    print(
        "(These Fp values are integrated only up to "
        f"q/k0={Q_OVER_K0_MAX}, so do not treat them as final total Fp.)"
    )

    # --------------------------------------------------------
    # Reconstruct 2D maps
    # --------------------------------------------------------

    (
        axis,
        SX,
        SY,
        rho_z_map,
    ) = reconstruct_2d_from_radial(
        s_radial=s,
        rho_radial=rho_z,
    )

    (
        _axis2,
        _SX2,
        _SY2,
        rho_h_map,
    ) = reconstruct_2d_from_radial(
        s_radial=s,
        rho_radial=rho_h,
    )

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    npz_path = (
        OUTPUT_DIR
        /
        "fig2bf_homogeneous_slab_850meV.npz"
    )

    np.savez_compressed(
        npz_path,
        energy_meV=ENERGY_MEV,
        wavelength_nm=WAVELENGTH_NM,
        a_nm=A_NM,
        h_nm=H_NM,
        z0_from_substrate_nm=Z0_FROM_SUBSTRATE_NM,
        n_si=N_SI,
        n_sio2=N_SIO2,
        eps_si=EPS_SI,
        eps_sio2=EPS_SIO2,
        q_over_k0_radial=s,
        Pz_raw=Pz,
        Px_raw=Px,
        Py_raw=Py,
        rho_z=rho_z,
        rho_x=rho_x,
        rho_y=rho_y,
        rho_horizontal=rho_h,
        plot_axis=axis,
        rho_z_map=rho_z_map,
        rho_horizontal_map=rho_h_map,
        Fp_z_truncated=Fz_truncated,
        Fp_horizontal_truncated=Fh_truncated,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    fig2d = plot_2d_maps(
        axis=axis,
        rho_z_map=rho_z_map,
        rho_h_map=rho_h_map,
    )

    map_png = (
        OUTPUT_DIR
        /
        "fig2bf_homogeneous_slab_log_hot.png"
    )

    map_pdf = (
        OUTPUT_DIR
        /
        "fig2bf_homogeneous_slab_log_hot.pdf"
    )

    fig2d.savefig(
        map_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig2d.savefig(
        map_pdf,
        bbox_inches="tight",
    )

    radial_fig = plot_radial_cuts(
        s=s,
        rho_z=rho_z,
        rho_h=rho_h,
    )

    radial_png = (
        OUTPUT_DIR
        /
        "fig2bf_homogeneous_slab_radial.png"
    )

    radial_pdf = (
        OUTPUT_DIR
        /
        "fig2bf_homogeneous_slab_radial.pdf"
    )

    radial_fig.savefig(
        radial_png,
        dpi=300,
        bbox_inches="tight",
    )

    radial_fig.savefig(
        radial_pdf,
        bbox_inches="tight",
    )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"NPZ        : {npz_path}"
    )

    print(
        f"2D PNG     : {map_png}"
    )

    print(
        f"2D PDF     : {map_pdf}"
    )

    print(
        f"radial PNG : {radial_png}"
    )

    print(
        f"radial PDF : {radial_pdf}"
    )

    plt.show()


if __name__ == "__main__":
    main()
