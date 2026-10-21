"""
Reproduce the BLACK total-Purcell-factor curve of Fig. 4(c)
Dyakov et al., Phys. Rev. B 108, 155416 (2023)

Structure:
    inverse triangular PhC = air holes in Si on SiO2
    a = 445 nm
    r = 0.405 a
    h = 230 nm
    B-point, x-oriented dipole
    z = h/2

Method:
    For each photon energy E,
        1) calculate the dipole near-field power P(q,E)
           throughout one reciprocal primitive cell,
        2) integrate
               P_struct(E) = ∫ P(q,E) d²q/(2π)²,
        3) divide by the homogeneous-Si reference power,
               Fp(E) = P_struct(E)/P_ref(E).

Notes:
    * backend="dense" only.
    * pw=(7,7), as requested in the current reproduction work.
      The paper reports 85 Fourier harmonics, so pw=(9,9) is a useful
      later convergence check.
    * The q integration is the difficult part.  Increase NQ from 31 to
      41/51 and verify convergence of the 800-meV peak before treating
      the result as quantitatively final.
    * A small substrate loss is included, as in the paper.  In the corrected
      Khepri exp(-i kz z) convention, passive loss uses -0.001j.
    * The paper allows the Purcell reference medium to be chosen.  Here
      REFERENCE_INDEX = n_Si because the emitter is hosted in silicon.
"""

# ============================================================
# 0. THREAD LIMITS - MUST COME BEFORE NUMPY
# ============================================================

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ============================================================
# 1. IMPORTS
# ============================================================

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import (
    dipole_power_q,
    homogeneous_reference_power_xy,
)

# ============================================================
# 2. NUMERICAL SETTINGS
# ============================================================

PW = (7, 7)
NQ = 31                      # q points along each reciprocal primitive vector
PIX = (384, 384)
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))

# Spectrum: coarse everywhere + finer sampling around the 800-meV resonance.
COARSE_STEP_MEV = 10.0
FINE_MIN_MEV = 740.0
FINE_MAX_MEV = 860.0
FINE_STEP_MEV = 2.0

RESULTS_FILE = "fig4c_total_purcell_checkpoint.npz"

# ============================================================
# 3. PAPER GEOMETRY / MATERIALS
# ============================================================

A_NM = 445.0
H_NM = 230.0
R_OVER_A = 0.405

N_SI = 3.48
N_SIO2 = 1.45

EPS_AIR = 1.0
EPS_SI = N_SI**2
EPS_SUB = N_SIO2**2 - 0.001j

REFERENCE_INDEX = N_SI
CURRENT = (1.0, 0.0, 0.0)   # x-oriented dipole

HC_EV_NM = 1239.841984
HBARC_EV_NM = 197.3269804

# Normalize lengths by a.
H = H_NM / A_NM
R = R_OVER_A

# Physical real-space orientation used in our Fig. 3 reproduction.
a1 = np.array([np.sqrt(3.0)/2.0, +0.5])
a2 = np.array([np.sqrt(3.0)/2.0, -0.5])
lattice = np.array([a1, a2])

# B-point used in the present reproduction.
B = (a1 + a2) / 2.0
DIPOLE_POSITION = (float(B[0]), float(B[1]), H/2.0)

REAL_CELL_AREA = abs(a1[0]*a2[1] - a1[1]*a2[0])

# ============================================================
# 4. RECIPROCAL LATTICE
# ============================================================

def reciprocal_vectors(u, v):
    det = u[0]*v[1] - u[1]*v[0]

    b1 = (2*np.pi/det) * np.array([v[1], -v[0]])
    b2 = (2*np.pi/det) * np.array([-u[1], u[0]])

    return b1, b2

b1, b2 = reciprocal_vectors(a1, a2)

# ============================================================
# 5. ENERGY GRID
# ============================================================

coarse_energies = np.arange(
    600.0,
    1300.0 + 0.5*COARSE_STEP_MEV,
    COARSE_STEP_MEV,
)

fine_energies = np.arange(
    FINE_MIN_MEV,
    FINE_MAX_MEV + 0.5*FINE_STEP_MEV,
    FINE_STEP_MEV,
)

ENERGIES_MEV = np.unique(
    np.round(
        np.concatenate([coarse_energies, fine_energies]),
        10,
    )
)

# ============================================================
# 6. RECIPROCAL PRIMITIVE-CELL q GRID
# ============================================================

def make_q_grid(N):
    """
    Midpoint grid over one reciprocal primitive cell:

        q = u b1 + v b2,
        -1/2 <= u,v < 1/2.

    Integrating over this primitive reciprocal cell is equivalent to
    integrating over the Wigner-Seitz FBZ for a reciprocal-periodic P(q).

    For a=1:
        ∫ P(q) d²q/(2π)²
        = mean(P) / A_real.
    """

    if N % 2 == 0:
        raise ValueError("Use an odd NQ so q=0 and q<->-q pairing are exact.")

    u = (np.arange(N) + 0.5)/N - 0.5
    v = (np.arange(N) + 0.5)/N - 0.5

    U, V = np.meshgrid(u, v, indexing="xy")

    QX = U*b1[0] + V*b2[0]
    QY = U*b1[1] + V*b2[1]

    return U, V, QX, QY

U, V, QX, QY = make_q_grid(NQ)

# ============================================================
# 7. BUILD THE KHEPRI CRYSTAL
# ============================================================

def build_crystal():
    crystal = Crystal(
        pw=PW,
        lattice=lattice,
        lattice_pitch=1.0,
        epsi=EPS_AIR,
        epse=EPS_SUB,
    )

    drawing = Drawing(
        PIX,
        EPS_SI,
        crystal.lattice,
    )

    drawing.disc(
        (0.0, 0.0),
        R,
        EPS_AIR,
    )

    pattern = drawing.canvas()

    # Split at the dipole plane z=h/2.
    crystal.add_layer_pixmap("L", pattern, H/2.0)
    crystal.add_layer_pixmap("R", pattern, H/2.0)

    crystal.set_device(
        ["L", "R"],
        fields_mask=[True, False],
    )

    return crystal

# ============================================================
# 8. SINGLE-q SOLVE
# ============================================================

def solve_q(crystal, wavelength, qx, qy, details=False):
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
# 9. CHECK q -> -q RECIPROCITY
# ============================================================

def check_reciprocity(crystal):
    energy_meV = 800.0
    wavelength = (HC_EV_NM/(energy_meV/1000.0))/A_NM

    test_indices = [
        (NQ//2 + 3, NQ//2 + 5),
        (NQ//2 - 4, NQ//2 + 2),
        (NQ//2 + 6, NQ//2 - 3),
    ]

    print("\nReciprocity check at 800 meV:")

    passed = True

    for iy, ix in test_indices:
        qx = QX[iy, ix]
        qy = QY[iy, ix]

        p1 = solve_q(crystal, wavelength, qx, qy)["P_total"]
        p2 = solve_q(crystal, wavelength, -qx, -qy)["P_total"]

        rel = abs(p1-p2) / max(abs(p1), abs(p2), 1e-14)
        good = rel < 2e-5

        print(
            f"  q=({qx:+.5f},{qy:+.5f})  "
            f"rel.diff={rel:.3e}  "
            f"{'PASS' if good else 'FAIL'}"
        )

        passed &= good

    return passed

# ============================================================
# 10. BUILD THE q TASK LIST
# ============================================================

def make_q_tasks(use_reciprocity):
    """
    Returns (iy, ix, qx, qy, multiplicity).

    If P(q)=P(-q), only one point in each inversion pair is solved.
    multiplicity=2 accounts for the omitted partner.
    """

    tasks = []

    for iy in range(NQ):
        for ix in range(NQ):

            if not use_reciprocity:
                tasks.append(
                    (iy, ix, QX[iy, ix], QY[iy, ix], 1)
                )
                continue

            jy = NQ - 1 - iy
            jx = NQ - 1 - ix

            if iy > jy or (iy == jy and ix > jx):
                continue

            multiplicity = 1 if (iy == jy and ix == jx) else 2

            tasks.append(
                (iy, ix, QX[iy, ix], QY[iy, ix], multiplicity)
            )

    return tasks

# ============================================================
# 11. MULTIPROCESSING WORKER
# ============================================================

_worker_crystal = None

def init_worker():
    global _worker_crystal
    _worker_crystal = build_crystal()

def worker(task):
    """
    task =
        (qx, qy, multiplicity, wavelength)
    """

    qx, qy, multiplicity, wavelength = task

    p = solve_q(
        _worker_crystal,
        wavelength,
        qx,
        qy,
    )["P_total"]

    return float(p), int(multiplicity)

# ============================================================
# 12. ONE-ENERGY PURCELL FACTOR
# ============================================================

def purcell_at_energy(executor, q_tasks, energy_meV):
    """
    Calculate Fp(E) by BZ integration.

    Since the q grid uniformly samples one reciprocal primitive cell,

        P_structure
          = [sum_i multiplicity_i * P(q_i)] / (NQ^2 * A_real).
    """

    energy_eV = energy_meV / 1000.0

    wavelength_nm = HC_EV_NM / energy_eV
    wavelength = wavelength_nm / A_NM

    tasks = [
        (
            qx,
            qy,
            multiplicity,
            wavelength,
        )
        for _iy, _ix, qx, qy, multiplicity in q_tasks
    ]

    chunksize = max(
        1,
        len(tasks)//(WORKERS*20),
    )

    weighted_sum = 0.0
    p_max = -np.inf

    for p, multiplicity in executor.map(
        worker,
        tasks,
        chunksize=chunksize,
    ):
        weighted_sum += multiplicity * p
        p_max = max(p_max, p)

    # The multiplicities sum to NQ^2.
    P_structure = (
        weighted_sum
        /
        (NQ*NQ*REAL_CELL_AREA)
    )

    P_reference = homogeneous_reference_power_xy(
        wavelength=wavelength,
        refractive_index=REFERENCE_INDEX,
        current=CURRENT,
    )

    Fp = P_structure / P_reference

    return (
        float(Fp),
        float(P_structure),
        float(P_reference),
        float(p_max),
    )

# ============================================================
# 13. CHECKPOINT / RESUME
# ============================================================

def load_checkpoint():
    Fp = np.full(len(ENERGIES_MEV), np.nan)
    Pstruct = np.full(len(ENERGIES_MEV), np.nan)
    Pref = np.full(len(ENERGIES_MEV), np.nan)
    Pmax = np.full(len(ENERGIES_MEV), np.nan)

    if not os.path.exists(RESULTS_FILE):
        return Fp, Pstruct, Pref, Pmax

    data = np.load(RESULTS_FILE)

    compatible = (
        np.array_equal(data["energies_meV"], ENERGIES_MEV)
        and int(data["NQ"]) == NQ
        and np.array_equal(data["PW"], np.array(PW))
        and np.allclose(data["a1"], a1)
        and np.allclose(data["a2"], a2)
        and np.allclose(data["B"], B)
    )

    if not compatible:
        print("\nCheckpoint exists but settings changed; starting a new sweep.")
        return Fp, Pstruct, Pref, Pmax

    print(f"\nResuming from {RESULTS_FILE}")

    return (
        data["Fp"].copy(),
        data["P_structure"].copy(),
        data["P_reference"].copy(),
        data["P_q_max"].copy(),
    )

def save_checkpoint(Fp, Pstruct, Pref, Pmax):
    np.savez(
        RESULTS_FILE,
        energies_meV=ENERGIES_MEV,
        Fp=Fp,
        P_structure=Pstruct,
        P_reference=Pref,
        P_q_max=Pmax,
        NQ=NQ,
        PW=np.array(PW),
        a1=a1,
        a2=a2,
        B=B,
        reference_index=REFERENCE_INDEX,
    )

# ============================================================
# 14. PAPER DIFFRACTION THRESHOLDS
# ============================================================

# Eq. (14) of the paper, using substrate index n_SiO2.
E_LOWER_MEV = (
    2*np.pi*np.sqrt(3.0)*HBARC_EV_NM
    /
    (3.0*N_SIO2*A_NM)
    *
    1000.0
)

E_UPPER_MEV = (
    4*np.pi*HBARC_EV_NM
    /
    (3.0*N_SIO2*A_NM)
    *
    1000.0
)

# ============================================================
# 15. PLOT THE BLACK CURVE
# ============================================================

def show_spectrum(Fp):
    import matplotlib.pyplot as plt

    valid = np.isfinite(Fp)

    fig, ax = plt.subplots(figsize=(7.2, 5.3))

    ax.plot(
        ENERGIES_MEV[valid],
        Fp[valid],
        "k-",
        linewidth=1.8,
        label="total Purcell factor",
    )

    ax.axvline(
        E_LOWER_MEV,
        linestyle="--",
        linewidth=1.2,
        label="lower diffraction threshold",
    )

    ax.axvline(
        E_UPPER_MEV,
        linestyle="--",
        linewidth=1.2,
        label="upper diffraction threshold",
    )

    ax.set_xlim(600, 1300)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Photon energy (meV)")
    ax.set_ylabel("Purcell factor")

    ax.set_title(
        "Inverse triangular PhC: B-point x-dipole"
    )

    ax.legend()
    fig.tight_layout()
    plt.show()

# ============================================================
# 16. MAIN
# ============================================================

def main():

    print("="*78)
    print("FIG. 4(c) BLACK CURVE - TOTAL PURCELL FACTOR")
    print("="*78)

    print(f"PW                  = {PW}")
    print(f"NQ                  = {NQ} x {NQ}")
    print(f"workers             = {WORKERS}")
    print(f"B/a                 = {B}")
    print(f"z/a                 = {H/2:.8f}")
    print(f"reference index     = {REFERENCE_INDEX}")
    print(f"substrate epsilon   = {EPS_SUB}")
    print(f"energy points       = {len(ENERGIES_MEV)}")
    print(f"lower threshold     = {E_LOWER_MEV:.2f} meV")
    print(f"upper threshold     = {E_UPPER_MEV:.2f} meV")

    # Main-process crystal used only for validation.
    crystal = build_crystal()

    # Source-jump sanity check.
    wavelength_800 = (
        HC_EV_NM/(800.0/1000.0)
    ) / A_NM

    check = solve_q(
        crystal,
        wavelength_800,
        0.0,
        0.0,
        details=True,
    )

    print(
        f"\nGamma source check: "
        f"P={check['P_total']:.6e}, "
        f"jump={check['jump_relative_error']:.3e}"
    )

    if check["jump_relative_error"] > 1e-8:
        raise RuntimeError(
            "Dipole Maxwell-jump validation failed."
        )

    use_reciprocity = check_reciprocity(crystal)

    if use_reciprocity:
        print("[PASS] Using P(q)=P(-q) to halve q solves.")
    else:
        print("[WARNING] Reciprocity check failed; using the full q grid.")

    q_tasks = make_q_tasks(use_reciprocity)

    print(
        f"\nActual q solves per energy = {len(q_tasks)} "
        f"(full grid would be {NQ*NQ})"
    )

    Fp, Pstruct, Pref, Pmax = load_checkpoint()

    start_total = time.perf_counter()

    # Keep one worker pool alive for the whole spectrum.
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=init_worker,
    ) as executor:

        for i, energy in enumerate(ENERGIES_MEV):

            if np.isfinite(Fp[i]):
                print(
                    f"[{i+1:3d}/{len(ENERGIES_MEV):3d}] "
                    f"{energy:7.1f} meV  "
                    f"Fp={Fp[i]:9.5f}  [cached]"
                )
                continue

            t0 = time.perf_counter()

            (
                Fp[i],
                Pstruct[i],
                Pref[i],
                Pmax[i],
            ) = purcell_at_energy(
                executor,
                q_tasks,
                energy,
            )

            elapsed = time.perf_counter() - t0

            print(
                f"[{i+1:3d}/{len(ENERGIES_MEV):3d}] "
                f"{energy:7.1f} meV  "
                f"Fp={Fp[i]:9.5f}  "
                f"Pstruct={Pstruct[i]:.4e}  "
                f"Pref={Pref[i]:.4e}  "
                f"max P(q)={Pmax[i]:.4e}  "
                f"{elapsed:6.1f}s"
            )

            # Save after every energy, so a long run can be resumed.
            save_checkpoint(
                Fp,
                Pstruct,
                Pref,
                Pmax,
            )

    total_time = time.perf_counter() - start_total

    print("\n" + "="*78)
    print(f"Sweep finished in {total_time/60.0:.2f} min")
    print("="*78)

    # Report the computed maximum.
    imax = np.nanargmax(Fp)

    print(
        f"\nMaximum sampled Fp = {Fp[imax]:.6f} "
        f"at E = {ENERGIES_MEV[imax]:.1f} meV"
    )

    # Explicit 800-meV value.
    i800 = np.where(np.isclose(ENERGIES_MEV, 800.0))[0]

    if len(i800):
        j = i800[0]
        print(
            f"Fp(800 meV) = {Fp[j]:.6f}"
        )

    show_spectrum(Fp)


# Required for Windows multiprocessing.
if __name__ == "__main__":
    main()
