"""
Validate the Khepri vacuum dipole emissivity against the exact analytical
momentum-space solution.

This script uses the NPZ file produced by:

    examples/reproduce_fig2ae_vacuum_normalized.py

Expected input:
    fig2_vacuum_output/fig2ae_vacuum_850meV_normalized.npz

It compares a radial cut along q_y = 0 with the exact vacuum expressions

    rho_z(s) =
        3/(4*pi) * s^2 / sqrt(1-s^2)

    rho_parallel(s) =
        3/(4*pi) * (1 - s^2/2) / sqrt(1-s^2)

where

    s = q_parallel / k0

and rho_parallel is the average of x- and y-oriented dipoles.

The dimensionless density is defined so that

    F_P = integral rho_F(sx,sy) dsx dsy

and the exact vacuum integral is F_P = 1.

Run from the repository root:

    python examples/validate_vacuum_radial_analytic.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "fig2_vacuum_output/"
    "fig2ae_vacuum_850meV_normalized.npz"
)

OUTPUT_DIR = Path(
    "fig2_vacuum_output"
)


# ============================================================
# ANALYTICAL VACUUM SOLUTIONS
# ============================================================

def rho_z_analytic(s):
    """
    Exact dimensionless q-resolved Purcell-emissivity density
    for a z-oriented electric dipole in vacuum.
    """
    s = np.asarray(s, dtype=float)

    result = np.zeros_like(s)

    inside = (
        (s >= 0.0)
        &
        (s < 1.0)
    )

    result[inside] = (
        3.0
        /
        (4.0 * np.pi)
        *
        s[inside]**2
        /
        np.sqrt(
            1.0
            -
            s[inside]**2
        )
    )

    return result


def rho_horizontal_analytic(s):
    """
    Exact dimensionless q-resolved Purcell-emissivity density
    for the average of x- and y-oriented electric dipoles in vacuum.
    """
    s = np.asarray(s, dtype=float)

    result = np.zeros_like(s)

    inside = (
        (s >= 0.0)
        &
        (s < 1.0)
    )

    result[inside] = (
        3.0
        /
        (4.0 * np.pi)
        *
        (
            1.0
            -
            0.5 * s[inside]**2
        )
        /
        np.sqrt(
            1.0
            -
            s[inside]**2
        )
    )

    return result


# ============================================================
# ERROR METRICS
# ============================================================

def relative_error(
    numerical,
    analytical,
):
    numerical = np.asarray(
        numerical,
        dtype=float,
    )

    analytical = np.asarray(
        analytical,
        dtype=float,
    )

    return (
        np.abs(
            numerical
            -
            analytical
        )
        /
        np.maximum(
            np.abs(analytical),
            1e-15,
        )
    )


def report_errors(
    name,
    s,
    numerical,
    analytical,
):
    """
    Report errors away from the integrable light-line singularity.
    """

    # At s=0 the z-dipole analytical value is exactly zero,
    # so exclude points whose reference value is too small
    # from relative-error statistics.
    valid = (
        (s < 0.95)
        &
        (np.abs(analytical) > 1e-12)
    )

    rel = relative_error(
        numerical[valid],
        analytical[valid],
    )

    abs_err = np.abs(
        numerical[valid]
        -
        analytical[valid]
    )

    print()
    print(name)
    print("-" * len(name))

    print(
        f"points used (s < 0.95)     = {np.count_nonzero(valid)}"
    )

    if np.count_nonzero(valid) > 0:
        print(
            f"maximum relative error     = {np.max(rel):.6e}"
        )

        print(
            f"mean relative error        = {np.mean(rel):.6e}"
        )

        print(
            f"maximum absolute error     = {np.max(abs_err):.6e}"
        )


# ============================================================
# PLOTS
# ============================================================

def plot_vertical(
    s,
    numerical,
    analytical,
):

    fig, ax = plt.subplots(
        figsize=(6.2, 5.0)
    )

    ax.plot(
        s,
        analytical,
        linewidth=2.0,
        label="Analytical vacuum",
    )

    ax.plot(
        s,
        numerical,
        "o",
        markersize=4.0,
        markerfacecolor="none",
        label="Khepri",
    )

    ax.set_xlabel(
        r"$q_\parallel/k_0$"
    )

    ax.set_ylabel(
        r"$\rho_z$"
    )

    ax.set_title(
        "Vertical dipole in vacuum"
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    return fig


def plot_horizontal(
    s,
    numerical,
    analytical,
):

    fig, ax = plt.subplots(
        figsize=(6.2, 5.0)
    )

    ax.plot(
        s,
        analytical,
        linewidth=2.0,
        label="Analytical vacuum",
    )

    ax.plot(
        s,
        numerical,
        "o",
        markersize=4.0,
        markerfacecolor="none",
        label="Khepri",
    )

    ax.set_xlabel(
        r"$q_\parallel/k_0$"
    )

    ax.set_ylabel(
        r"$\rho_{\parallel}$"
    )

    ax.set_title(
        "Horizontal dipole average in vacuum"
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}\n\n"
            "First run:\n"
            "python examples/reproduce_fig2ae_vacuum_normalized.py"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = np.load(
        INPUT_FILE
    )

    s = np.asarray(
        data["q_over_k0"],
        dtype=float,
    )

    rho_z = np.asarray(
        data["rho_z"],
        dtype=float,
    )

    rho_h = np.asarray(
        data["rho_horizontal"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Extract the q_y = 0 radial cut.
    # --------------------------------------------------------

    iy0 = int(
        np.argmin(
            np.abs(s)
        )
    )

    sy_used = float(
        s[iy0]
    )

    rho_z_cut = rho_z[
        iy0,
        :
    ]

    rho_h_cut = rho_h[
        iy0,
        :
    ]

    # Keep only q_x >= 0 and strictly inside the light cone.
    mask = (
        (s >= 0.0)
        &
        (s < 1.0)
    )

    s_cut = s[
        mask
    ]

    rho_z_khepri = rho_z_cut[
        mask
    ]

    rho_h_khepri = rho_h_cut[
        mask
    ]

    rho_z_exact = rho_z_analytic(
        s_cut
    )

    rho_h_exact = rho_horizontal_analytic(
        s_cut
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("=" * 78)
    print("VACUUM RADIAL-CUT ANALYTICAL VALIDATION")
    print("=" * 78)

    print(
        f"input file               = {INPUT_FILE}"
    )

    print(
        f"q_y/k0 used              = {sy_used:+.12e}"
    )

    print(
        f"number of radial points  = {len(s_cut)}"
    )

    print()
    print("Exact center values:")

    print(
        "  rho_z(0)               = 0"
    )

    print(
        "  rho_horizontal(0)      = "
        f"{3.0/(4.0*np.pi):.12e}"
    )

    print()
    print("Khepri center values:")

    i0 = int(
        np.argmin(
            np.abs(s)
        )
    )

    print(
        "  rho_z(0)               = "
        f"{rho_z[i0, i0]:.12e}"
    )

    print(
        "  rho_horizontal(0)      = "
        f"{rho_h[i0, i0]:.12e}"
    )

    report_errors(
        "Vertical z dipole",
        s_cut,
        rho_z_khepri,
        rho_z_exact,
    )

    report_errors(
        "Horizontal x/y average",
        s_cut,
        rho_h_khepri,
        rho_h_exact,
    )

    # --------------------------------------------------------
    # Save numerical comparison table.
    # --------------------------------------------------------

    table = np.column_stack(
        [
            s_cut,
            rho_z_khepri,
            rho_z_exact,
            rho_h_khepri,
            rho_h_exact,
        ]
    )

    csv_path = (
        OUTPUT_DIR
        /
        "vacuum_radial_analytic_comparison.csv"
    )

    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=(
            "q_over_k0,"
            "rho_z_khepri,"
            "rho_z_analytic,"
            "rho_horizontal_khepri,"
            "rho_horizontal_analytic"
        ),
        comments="",
    )

    # --------------------------------------------------------
    # Vertical plot.
    # --------------------------------------------------------

    fig_z = plot_vertical(
        s_cut,
        rho_z_khepri,
        rho_z_exact,
    )

    z_png = (
        OUTPUT_DIR
        /
        "vacuum_radial_z_khepri_vs_analytic.png"
    )

    z_pdf = (
        OUTPUT_DIR
        /
        "vacuum_radial_z_khepri_vs_analytic.pdf"
    )

    fig_z.savefig(
        z_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig_z.savefig(
        z_pdf,
        bbox_inches="tight",
    )

    # --------------------------------------------------------
    # Horizontal plot.
    # --------------------------------------------------------

    fig_h = plot_horizontal(
        s_cut,
        rho_h_khepri,
        rho_h_exact,
    )

    h_png = (
        OUTPUT_DIR
        /
        "vacuum_radial_horizontal_khepri_vs_analytic.png"
    )

    h_pdf = (
        OUTPUT_DIR
        /
        "vacuum_radial_horizontal_khepri_vs_analytic.pdf"
    )

    fig_h.savefig(
        h_png,
        dpi=300,
        bbox_inches="tight",
    )

    fig_h.savefig(
        h_pdf,
        bbox_inches="tight",
    )

    print()
    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)

    print(
        f"CSV             : {csv_path}"
    )

    print(
        f"vertical PNG    : {z_png}"
    )

    print(
        f"vertical PDF    : {z_pdf}"
    )

    print(
        f"horizontal PNG  : {h_png}"
    )

    print(
        f"horizontal PDF  : {h_pdf}"
    )

    plt.show()


if __name__ == "__main__":
    main()
