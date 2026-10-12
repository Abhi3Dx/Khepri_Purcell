"""
Reproduce Fig. 2(a) and Fig. 2(e) of Dyakov et al.,
Phys. Rev. B 108, 155416 (2023): dipole near-field emissivity in vacuum.

UPDATED VERSION
---------------
- Uses a LOG color scale
- Uses the "hot" colormap
- Colorbar is shown in powers of 10

Panels reproduced
-----------------
Fig. 2(a): vertical z-oriented electric dipole in vacuum.
Fig. 2(e): horizontal dipole in vacuum, defined in the paper as
            the average of x- and y-oriented dipoles:
                f_parallel = (f_x + f_y)/2

Run from the repository root:
    python examples/reproduce_fig2ae_vacuum_log_hot.py
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
from khepri.dipole import dipole_power_q


ENERGY_MEV = 850.0
HC_EV_NM = 1239.841984
A_NM = 625.0

WAVELENGTH_NM = HC_EV_NM / (ENERGY_MEV / 1000.0)
WAVELENGTH_OVER_A = WAVELENGTH_NM / A_NM
K0 = 2.0 * np.pi / WAVELENGTH_OVER_A

PW = (1, 1)
Q_OVER_K0_MAX = 1.15
NQ = 151

SOURCE_PLANE_Z_OVER_A = 80.0 / A_NM
VACUUM_HALF_DEPTH = SOURCE_PLANE_Z_OVER_A

OUTPUT_DIR = Path("fig2_vacuum_output")


def build_vacuum_crystal():
    crystal = Crystal(
        pw=PW,
        lattice="square",
        lattice_pitch=1.0,
        epsi=1.0,
        epse=1.0,
    )
    crystal.add_layer_uniform("minus_z_half", epsilon=1.0, depth=VACUUM_HALF_DEPTH)
    crystal.add_layer_uniform("plus_z_half", epsilon=1.0, depth=VACUUM_HALF_DEPTH)
    crystal.set_device(["minus_z_half", "plus_z_half"], fields_mask=[True, False])
    return crystal


def build_q_grid():
    ds = 2.0 * Q_OVER_K0_MAX / NQ
    s = -Q_OVER_K0_MAX + (np.arange(NQ) + 0.5) * ds
    SX, SY = np.meshgrid(s, s, indexing="xy")
    QX = K0 * SX
    QY = K0 * SY

    distance_to_light_line = np.abs(1.0 - SX**2 - SY**2)
    min_distance = float(np.min(distance_to_light_line))
    if min_distance < 1e-12:
        raise RuntimeError("The chosen q grid hits the exact vacuum light line. Change NQ slightly.")

    return s, SX, SY, QX, QY, min_distance


def calculate_map(crystal, QX, QY, current, label):
    result = np.empty(QX.shape, dtype=float)
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
                position=(0.0, 0.0, SOURCE_PLANE_Z_OVER_A),
                current=current,
                backend="dense",
                return_details=False,
            )["P_total"]

            result[iy, ix] = float(np.real(power))

        if (iy + 1) % 10 == 0 or iy == QX.shape[0] - 1:
            elapsed = perf_counter() - t0
            print(f"{label}: row {iy+1:3d}/{QX.shape[0]}  points {counter:6d}/{total:6d}  time={elapsed:.1f} s")

    return result


def prepare_positive_for_log(data):
    """
    Prepare raw emissivity data for log plotting.

    We do not change the saved raw data. This function only prepares
    a positive plotting array and a suitable lower cutoff.
    """
    raw = np.asarray(data, dtype=float)
    positive = raw[raw > 0]

    if positive.size == 0:
        raise RuntimeError("No positive values found; cannot plot on a log scale.")

    vmin = float(np.min(positive))
    vmax = float(np.max(positive))

    # Replace non-positive values by the smallest positive plotted value
    # so the full map can still be displayed on a log color scale.
    plot_data = raw.copy()
    plot_data[plot_data <= 0] = vmin

    return plot_data, vmin, vmax


def plot_fig2ae_log_hot(s, Pz, Phorizontal):
    extent = [s[0], s[-1], s[0], s[-1]]

    phi = np.linspace(0.0, 2.0*np.pi, 500)
    circle_x = np.cos(phi)
    circle_y = np.sin(phi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)

    panels = [
        (axes[0], Pz, "(a) vertical dipole in vacuum"),
        (axes[1], Phorizontal, "(e) horizontal dipole in vacuum"),
    ]

    for ax, data, title in panels:
        plot_data, vmin, vmax = prepare_positive_for_log(data)

        im = ax.imshow(
            plot_data,
            origin="lower",
            extent=extent,
            aspect="equal",
            interpolation="nearest",
            cmap="hot",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )

        ax.plot(circle_x, circle_y, "--", linewidth=1.2, color="cyan", label="vacuum light line")
        ax.set_title(title)
        ax.set_xlabel(r"$q_x/k_0$")
        ax.set_ylabel(r"$q_y/k_0$")
        ax.set_xlim(-Q_OVER_K0_MAX, +Q_OVER_K0_MAX)
        ax.set_ylim(-Q_OVER_K0_MAX, +Q_OVER_K0_MAX)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.87, format=LogFormatterMathtext())
        cbar.set_label(r"$f(\mathbf{q})$")

        print(f"{title}: log-scale range = [{vmin:.6e}, {vmax:.6e}]")

    fig.suptitle("Dyakov et al. Fig. 2(a,e) baseline — 850 meV")
    return fig


def print_diagnostics(s, Pz, Px, Py, Phorizontal):
    i0 = int(np.argmin(np.abs(s)))
    center_z = Pz[i0, i0]
    center_h = Phorizontal[i0, i0]
    xy_difference = np.max(np.abs(Px - Py.T))
    xy_scale = max(np.max(np.abs(Px)), np.max(np.abs(Py)), 1e-30)
    xy_relative = xy_difference / xy_scale

    print()
    print("="*78)
    print("DIAGNOSTICS")
    print("="*78)
    print(f"closest q/k0 to origin = ({s[i0]:+.6e}, {s[i0]:+.6e})")
    print(f"Pz near q=0             = {center_z:.12e}")
    print(f"Phorizontal near q=0    = {center_h:.12e}")
    print(f"x/y 90-degree symmetry check max relative difference = {xy_relative:.3e}")
    print(f"min(Pz)                 = {np.min(Pz):.12e}")
    print(f"max(Pz)                 = {np.max(Pz):.12e}")
    print(f"min(Px)                 = {np.min(Px):.12e}")
    print(f"max(Px)                 = {np.max(Px):.12e}")
    print(f"min(Py)                 = {np.min(Py):.12e}")
    print(f"max(Py)                 = {np.max(Py):.12e}")
    print(f"min(Phorizontal)        = {np.min(Phorizontal):.12e}")
    print(f"max(Phorizontal)        = {np.max(Phorizontal):.12e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*78)
    print("REPRODUCE FIG. 2(a) AND FIG. 2(e): VACUUM")
    print("LOG COLOR SCALE + HOT COLORMAP")
    print("="*78)

    print(f"energy                  = {ENERGY_MEV:.3f} meV")
    print(f"wavelength              = {WAVELENGTH_NM:.6f} nm")
    print(f"a                       = {A_NM:.3f} nm")
    print(f"wavelength/a            = {WAVELENGTH_OVER_A:.12f}")
    print(f"k0 (normalized)         = {K0:.12f}")
    print(f"PW                      = {PW}")
    print(f"NQ                      = {NQ}")
    print(f"q/k0 range              = [-{Q_OVER_K0_MAX}, +{Q_OVER_K0_MAX}]")
    print(f"source z/a              = {SOURCE_PLANE_Z_OVER_A:.12f}")

    s, SX, SY, QX, QY, min_lightline_distance = build_q_grid()
    print(f"minimum |1-(qx/k0)^2-(qy/k0)^2| = {min_lightline_distance:.3e}")

    crystal = build_vacuum_crystal()

    print()
    print("Calculating Fig. 2(a): z dipole ...")
    Pz = calculate_map(crystal, QX, QY, current=(0.0, 0.0, 1.0), label="Jz")

    print()
    print("Calculating x part of Fig. 2(e) ...")
    Px = calculate_map(crystal, QX, QY, current=(1.0, 0.0, 0.0), label="Jx")

    print()
    print("Calculating y part of Fig. 2(e) ...")
    Py = calculate_map(crystal, QX, QY, current=(0.0, 1.0, 0.0), label="Jy")

    Phorizontal = 0.5 * (Px + Py)

    print_diagnostics(s, Pz, Px, Py, Phorizontal)

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

    fig = plot_fig2ae_log_hot(s, Pz, Phorizontal)

    png_path = OUTPUT_DIR / "fig2ae_vacuum_850meV_log_hot.png"
    pdf_path = OUTPUT_DIR / "fig2ae_vacuum_850meV_log_hot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print()
    print("="*78)
    print("FINISHED")
    print("="*78)
    print(f"raw data: {npz_path}")
    print(f"PNG     : {png_path}")
    print(f"PDF     : {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
