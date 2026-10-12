"""
Dyakov et al., PRB 108, 155416 (2023): Fig. 3(c)
Inverse triangular PhC, B-point, x-dipole.

Paper geometry:
    a = 445 nm
    h = 230 nm
    r = 0.405 a
    z = h/2
    air holes in silicon on SiO2

Calculated quantity:
    near-field dipole power P_total(E,q)
along the paper-style high-symmetry path
    M -> Gamma -> K -> M

Paper notes used here:
    - 85 Fourier harmonics
    - substrate loss 0.001 i for total near-field/Purcell calculations
    - blue = air light line
    - green = silica light line

Khepri conventions:
    - backend="dense"
    - passive loss => negative Im(epsilon)
    - PW=(7,7) gives 49 rectangular harmonics; this is intentionally used for speed

The exact dispersive material model used by the paper is not tabulated in
the main text, so this reproduction uses nSi=3.48, nSiO2=1.45.

The main plot uses LINEAR P/Pmax because Fig. 3 does not state that its
color scale is logarithmic. A log diagnostic is also saved.

Run:
    python examples/reproduce_fig3c_inverse_Bx_MGKM_7x7_axisfixed.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from khepri.crystal import Crystal
from khepri.dipole import dipole_power_q, homogeneous_reference_power


# ============================================================
# SETTINGS
# ============================================================

FAST_MODE = True

# Use 7x7 plane waves ONLY for all calculations.
PW = (7, 7)

if FAST_MODE:
    # Higher-resolution first pass.
    POINTS_PER_SEGMENT = 81
    ENERGY_MIN_MEV = 600.0
    ENERGY_MAX_MEV = 1600.0
    ENERGY_STEP_MEV = 2.5
else:
    # Higher-resolution scan, still using PW=(7,7).
    POINTS_PER_SEGMENT = 121
    ENERGY_MIN_MEV = 600.0
    ENERGY_MAX_MEV = 1600.0
    ENERGY_STEP_MEV = 2.0

WORKERS = 10
CHUNKSIZE = 1
CHECKPOINT_EVERY = 250
RUN_PW_SPOT_CHECK = True

OUT = Path("fig3c_inverse_Bx_output")
CHECKPOINT = OUT / (
    "fig3c_checkpoint_MGKM_7x7_axisfixed_v2.npz"
    if FAST_MODE else
    "fig3c_checkpoint_MGKM_paperlike_highres.npz"
)


# ============================================================
# GEOMETRY / MATERIALS
# ============================================================

A_NM = 445.0
H_NM = 230.0
R = 0.405
H = H_NM / A_NM
HALF_H = H / 2.0

N_SI = 3.48
N_SIO2 = 1.45

EPS_AIR = 1.0
EPS_SI = N_SI**2
EPS_SUB = N_SIO2**2 - 0.001j

CURRENT = (1.0, 0.0, 0.0)
HC_EV_NM = 1239.841984
HBARC_MEV_NM = 197326.9804

# Conventional triangular real-space lattice: flat-top reciprocal hexagon.
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3.0)/2.0])
LATTICE = np.array([A1, A2])

# Same B-point choice used in our Fig. 3(a) reproduction.
B_REAL = (A1 + A2) / 2.0
POSITION = (float(B_REAL[0]), float(B_REAL[1]), float(HALF_H))

nearest_B_hole = min(
    np.linalg.norm(B_REAL - (m*A1 + n*A2))
    for m in range(-2, 3)
    for n in range(-2, 3)
)
if nearest_B_hole <= R:
    raise RuntimeError("B point lies inside an air hole.")


# ============================================================
# PAPER-STYLE HIGH-SYMMETRY PATH
# ============================================================

def reciprocal_vectors(a1, a2):
    det = a1[0]*a2[1] - a1[1]*a2[0]

    b1 = (2.0*np.pi/det) * np.array([
        a2[1],
        -a2[0],
    ])

    b2 = (2.0*np.pi/det) * np.array([
        -a1[1],
        a1[0],
    ])

    return b1, b2


B1, B2 = reciprocal_vectors(A1, A2)

GAMMA = np.array([0.0, 0.0])

# Derived from the SAME reciprocal lattice used by the RCWA geometry.
# M: midpoint of the upper-right FBZ edge.
M_POINT = 0.5 * (B1 + B2)

# K: right-hand FBZ corner.
K_POINT = (2.0*B1 + B2) / 3.0


def make_path(nseg):
    """
    Paper path:
        M -> Gamma -> K -> M

    The horizontal coordinate is the cumulative reciprocal-space distance,
    not an arbitrary equal-width segment index. This is the standard band-
    diagram convention and preserves the actual relative lengths of M-Gamma,
    Gamma-K and K-M.
    """

    vertices = [
        M_POINT,
        GAMMA,
        K_POINT,
        M_POINT,
    ]

    q = []
    x = []

    cumulative = 0.0

    for seg in range(3):
        q0 = vertices[seg]
        q1 = vertices[seg + 1]

        segment_vector = q1 - q0
        segment_length = np.linalg.norm(segment_vector)

        tvals = np.linspace(
            0.0,
            1.0,
            nseg,
            endpoint=False,
        )

        for t in tvals:
            q.append(
                (1.0 - t) * q0
                +
                t * q1
            )

            x.append(
                cumulative
                +
                t * segment_length
            )

        cumulative += segment_length

    q.append(
        M_POINT.copy()
    )

    x.append(
        cumulative
    )

    return (
        np.asarray(q),
        np.asarray(x),
    )


Q_PATH, PATH_X = make_path(POINTS_PER_SEGMENT)
N_K = len(Q_PATH)


def centers_to_edges(values):
    """
    Convert strictly increasing sample-center coordinates to cell edges.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two 1D centers.")

    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Centers must be strictly increasing.")

    edges = np.empty(values.size + 1, dtype=float)

    edges[1:-1] = 0.5 * (
        values[:-1]
        +
        values[1:]
    )

    edges[0] = (
        values[0]
        -
        0.5*(values[1] - values[0])
    )

    edges[-1] = (
        values[-1]
        +
        0.5*(values[-1] - values[-2])
    )

    return edges


# Exact x-axis locations of M, Gamma, K, M for the cumulative-distance axis.
L_MG = np.linalg.norm(GAMMA - M_POINT)
L_GK = np.linalg.norm(K_POINT - GAMMA)
L_KM = np.linalg.norm(M_POINT - K_POINT)

PATH_TICKS = np.array(
    [
        0.0,
        L_MG,
        L_MG + L_GK,
        L_MG + L_GK + L_KM,
    ],
    dtype=float,
)

PATH_XMAX = float(PATH_TICKS[-1])

# Exact column indices of the high-symmetry points in Q_PATH.
IDX_M_START = 0
IDX_GAMMA = POINTS_PER_SEGMENT
IDX_K = 2 * POINTS_PER_SEGMENT
IDX_M_END = 3 * POINTS_PER_SEGMENT

if not np.allclose(Q_PATH[IDX_M_START], M_POINT, atol=1e-12):
    raise RuntimeError("Start of path is not exactly M.")

if not np.allclose(Q_PATH[IDX_GAMMA], GAMMA, atol=1e-12):
    raise RuntimeError("Gamma column is misaligned.")

if not np.allclose(Q_PATH[IDX_K], K_POINT, atol=1e-12):
    raise RuntimeError("K column is misaligned.")

if not np.allclose(Q_PATH[IDX_M_END], M_POINT, atol=1e-12):
    raise RuntimeError("End of path is not exactly M.")

ENERGIES = np.arange(
    ENERGY_MIN_MEV,
    ENERGY_MAX_MEV + 0.5*ENERGY_STEP_MEV,
    ENERGY_STEP_MEV,
)
N_E = len(ENERGIES)


# ============================================================
# CRYSTAL
# ============================================================

def islands():
    return [{
        "type": "disc",
        "params": [0.0, 0.0, R],
        "epsilon": EPS_AIR,
    }]


def build_crystal(pw=PW):
    c = Crystal(
        pw=pw,
        lattice=LATTICE,
        lattice_pitch=1.0,
        epsi=EPS_AIR,
        epse=EPS_SUB,
    )

    geom = islands()

    c.add_layer_analytical(
        "phc_air_side",
        epsilon=geom,
        epsilon_host=EPS_SI,
        depth=HALF_H,
    )

    c.add_layer_analytical(
        "phc_substrate_side",
        epsilon=geom,
        epsilon_host=EPS_SI,
        depth=HALF_H,
    )

    c.set_device(
        ["phc_air_side", "phc_substrate_side"],
        fields_mask=[True, False],
    )

    return c


SOURCE_AFTER = "phc_air_side"


def wavelength_over_a(E_meV):
    E_eV = E_meV / 1000.0
    return (HC_EV_NM / E_eV) / A_NM


def power(c, E_meV, q):
    result = dipole_power_q(
        crystal=c,
        source_after=SOURCE_AFTER,
        wavelength=wavelength_over_a(E_meV),
        kp=(float(q[0]), float(q[1])),
        position=POSITION,
        current=CURRENT,
        backend="dense",
        return_details=False,
    )
    return float(np.real(result["P_total"]))


# ============================================================
# SMALL PW CHECK
# ============================================================

def pw_spot_check():
    """
    Small source/geometry diagnostic at 800 meV using PW=(7,7) only.
    """

    print("\n" + "="*78)
    print("7x7 PLANE-WAVE SPOT CHECK AT 800 meV")
    print("="*78)

    tests = {
        "Gamma": GAMMA,
        "M": M_POINT,
        "K": K_POINT,
        "0.98 M": 0.98*M_POINT,
    }

    c = build_crystal(PW)

    print(f"\nPW={PW} ({PW[0]*PW[1]} harmonics)")

    for name, q in tests.items():
        print(
            f"  {name:8s}: "
            f"P={power(c, 800.0, q):.12e}"
        )


# ============================================================
# MULTIPROCESSING
# ============================================================

_WORKER_CRYSTAL = None


def worker_init():
    global _WORKER_CRYSTAL
    _WORKER_CRYSTAL = build_crystal(PW)


def worker_task(task):
    global _WORKER_CRYSTAL

    if _WORKER_CRYSTAL is None:
        worker_init()

    ie, ik, E, qx, qy = task
    p = power(_WORKER_CRYSTAL, E, np.array([qx, qy]))
    return ie, ik, p


# ============================================================
# CHECKPOINTING
# ============================================================

def signature():
    return np.array([
        PW[0], PW[1],
        POINTS_PER_SEGMENT,
        ENERGY_MIN_MEV,
        ENERGY_MAX_MEV,
        ENERGY_STEP_MEV,
        A_NM, H_NM, R,
    ], dtype=float)


def save_checkpoint(P):
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT,
        signature=signature(),
        power=P,
        energies=ENERGIES,
        q_path=Q_PATH,
        path_x=PATH_X,
    )


def load_checkpoint():
    empty = np.full((N_E, N_K), np.nan, dtype=float)

    if not CHECKPOINT.exists():
        return empty

    try:
        d = np.load(CHECKPOINT)

        if not np.allclose(
            np.asarray(d["signature"], float),
            signature(),
            rtol=0.0,
            atol=1e-12,
        ):
            print("Checkpoint settings differ; starting fresh.")
            return empty

        P = np.asarray(d["power"], float)

        if P.shape != (N_E, N_K):
            return empty

        done = np.count_nonzero(np.isfinite(P))
        print(f"Loaded checkpoint: {done}/{N_E*N_K} points complete.")
        return P

    except Exception as exc:
        print(f"Checkpoint could not be loaded: {exc}")
        return empty


# ============================================================
# FULL ENERGY-k MAP
# ============================================================

def calculate():
    P = load_checkpoint()

    tasks = []
    for ie, E in enumerate(ENERGIES):
        for ik, q in enumerate(Q_PATH):
            if not np.isfinite(P[ie, ik]):
                tasks.append((ie, ik, float(E), float(q[0]), float(q[1])))

    print("\n" + "="*78)
    print("FIG. 3(c) ENERGY-k MAP")
    print("="*78)
    print(f"PW                  = {PW} ({PW[0]*PW[1]} harmonics)")
    print(f"E range             = {ENERGY_MIN_MEV} ... {ENERGY_MAX_MEV} meV")
    print(f"E step              = {ENERGY_STEP_MEV} meV")
    print(f"k-path points       = {N_K}")
    print(f"total points        = {N_E*N_K}")
    print(f"remaining           = {len(tasks)}")
    print(f"workers             = {WORKERS}")

    if not tasks:
        return P

    t0 = perf_counter()

    def store(result, count):
        ie, ik, val = result
        P[ie, ik] = val
        if count % CHECKPOINT_EVERY == 0:
            save_checkpoint(P)

    context = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=context,
        initializer=worker_init,
    ) as executor:

        for count, result in enumerate(
            executor.map(worker_task, tasks, chunksize=CHUNKSIZE),
            start=1,
        ):
            store(result, count)

            if count % 100 == 0 or count == len(tasks):
                elapsed = perf_counter() - t0
                rate = count / max(elapsed, 1e-12)
                eta = (len(tasks)-count) / max(rate, 1e-12)

                print(
                    f"{count:6d}/{len(tasks):6d}  "
                    f"elapsed={elapsed/60:7.2f} min  "
                    f"ETA={eta/60:7.2f} min"
                )

    save_checkpoint(P)
    return P


# ============================================================
# SECOND, DIMENSIONLESS NORMALIZATION
# ============================================================

def make_rho(P):
    rho = np.empty_like(P)

    for ie, E in enumerate(ENERGIES):
        lam = wavelength_over_a(E)
        k0 = 2.0*np.pi/lam

        pref = homogeneous_reference_power(
            wavelength=lam,
            refractive_index=N_SI,
            current=CURRENT,
        )

        rho[ie, :] = (
            k0**2/(2.0*np.pi)**2
            *
            P[ie, :]
            /
            pref
        )

    return rho


# ============================================================
# LIGHT LINES
# ============================================================

def light_line(n):
    qabs = np.linalg.norm(Q_PATH, axis=1)
    return HBARC_MEV_NM*qabs/(n*A_NM)


E_AIR = light_line(1.0)
E_SIO2 = light_line(N_SIO2)


# ============================================================
# PLOTTING
# ============================================================

def plot_map(P, log=False):
    """
    Plot against the TRUE cumulative reciprocal-space distance.

    The previous version used imshow with a single [0, PATH_XMAX] extent.
    That incorrectly assumed every image column had the same physical
    k-spacing, even though M-Gamma, Gamma-K and K-M have different lengths
    but were sampled with the same number of points.

    This pcolormesh version uses the actual PATH_X coordinates and therefore
    keeps resonances aligned with the M, Gamma and K tick marks.
    """

    positive = P[
        np.isfinite(P)
        &
        (P > 0.0)
    ]

    if positive.size == 0:
        raise RuntimeError(
            "No positive emissivity values."
        )

    pmax = float(
        np.max(positive)
    )

    x_edges = centers_to_edges(
        PATH_X
    )

    e_edges = centers_to_edges(
        ENERGIES
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 5.7)
    )

    if log:
        floor = pmax * 1e-6

        im = ax.pcolormesh(
            x_edges,
            e_edges,
            np.maximum(P, floor),
            shading="flat",
            cmap="hot",
            norm=LogNorm(
                vmin=floor,
                vmax=pmax,
            ),
            rasterized=True,
        )

        cbar_label = (
            "near-field emissivity "
            "(log diagnostic)"
        )

    else:
        im = ax.pcolormesh(
            x_edges,
            e_edges,
            np.maximum(P, 0.0) / pmax,
            shading="flat",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )

        cbar_label = (
            r"near-field emissivity "
            r"$P/P_{\max}$"
        )

    # Paper caption: blue = air, green = silica.
    ax.plot(
        PATH_X,
        E_AIR,
        color="tab:blue",
        lw=1.5,
        label="air light line",
        zorder=4,
    )

    ax.plot(
        PATH_X,
        E_SIO2,
        color="tab:green",
        lw=1.5,
        label="silica light line",
        zorder=4,
    )

    for xpos in PATH_TICKS[1:-1]:
        ax.axvline(
            xpos,
            color="0.65",
            lw=0.8,
            alpha=0.65,
            zorder=3,
        )

    ax.set_xticks(
        PATH_TICKS
    )

    ax.set_xticklabels(
        [
            "M",
            r"$\Gamma$",
            "K",
            "M",
        ]
    )

    ax.set_xlim(
        PATH_TICKS[0],
        PATH_TICKS[-1],
    )

    ax.set_ylim(
        ENERGY_MIN_MEV,
        ENERGY_MAX_MEV,
    )

    ax.set_ylabel(
        "Photon energy (meV)"
    )

    ax.set_title(
        "Fig. 3(c): air holes in Si, "
        "B-point, x-dipole"
    )

    ax.legend(
        loc="upper left",
        fontsize=8,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        cbar_label
    )

    fig.tight_layout()

    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("="*78)
    print("DYAKOV PRB 108, 155416 — FIG. 3(c)")
    print("="*78)
    print(f"a={A_NM} nm, h={H_NM} nm, r/a={R}")
    print(f"B={B_REAL}, z/a={HALF_H:.8f}")
    print(f"Gamma={GAMMA}")
    print(f"M={M_POINT}")
    print(f"K={K_POINT}")
    print(f"b1={B1}")
    print(f"b2={B2}")
    print("path = M -> Gamma -> K -> M")
    print(
        "high-symmetry column indices: "
        f"M={IDX_M_START}, "
        f"Gamma={IDX_GAMMA}, "
        f"K={IDX_K}, "
        f"M={IDX_M_END}"
    )
    print(
        "high-symmetry cumulative distances: "
        f"{PATH_TICKS}"
    )
    print(f"points/segment = {POINTS_PER_SEGMENT}")
    print(f"total k points = {N_K}")
    print(f"energy step = {ENERGY_STEP_MEV} meV")
    print(f"number of energies = {N_E}")
    print(f"eps_sub={EPS_SUB}")

    if RUN_PW_SPOT_CHECK:
        pw_spot_check()

    P = calculate()
    rho = make_rho(P)

    npz = OUT / (
        "fig3c_inverse_Bx_MGKM_7x7_axisfixed_v2.npz"
        if FAST_MODE else
        "fig3c_inverse_Bx_MGKM_7x7_axisfixed_v2_highres.npz"
    )

    np.savez_compressed(
        npz,
        power=P,
        rho=rho,
        energies_mev=ENERGIES,
        q_path=Q_PATH,
        path_x=PATH_X,
        Gamma=GAMMA,
        M=M_POINT,
        K=K_POINT,
        B=B_REAL,
        air_light_line_mev=E_AIR,
        silica_light_line_mev=E_SIO2,
        pw=np.array(PW),
        a_nm=A_NM,
        h_nm=H_NM,
        r_over_a=R,
    )

    fig = plot_map(P, log=False)
    fig.savefig(OUT/"fig3c_inverse_Bx_MGKM_7x7_axisfixed_v2_linear.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT/"fig3c_inverse_Bx_MGKM_7x7_axisfixed_v2_linear.pdf", bbox_inches="tight")

    fig_log = plot_map(P, log=True)
    fig_log.savefig(OUT/"fig3c_inverse_Bx_MGKM_7x7_axisfixed_v2_log.png", dpi=300, bbox_inches="tight")
    fig_log.savefig(OUT/"fig3c_inverse_Bx_MGKM_7x7_axisfixed_v2_log.pdf", bbox_inches="tight")

    print("\nFINISHED")
    print(f"data: {npz}")
    print("The ~800 meV band-edge feature should appear near M.")

    plt.show()


if __name__ == "__main__":
    mp.freeze_support()
    main()
