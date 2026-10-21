"""
Compact PRB inverse-PhC dipole map
Dyakov et al., PRB 108, 155416 (2023)

Shows in one run:
  1) real-space triangular air-hole lattice + B-point x-dipole
  2) q-space near-field power map in the true hexagonal FBZ

Speed:
  - dense RCWA, pw=(7,7)
  - Nq=101
  - multiprocessing over q points
  - P(q)=P(-q) used only if a numerical reciprocity check passes

Images are DISPLAYED only; nothing is saved.
"""

# Limit BLAS threads before NumPy import: each worker gets one thread.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.dipole import dipole_power_q


# ============================================================
# 1. SETTINGS
# ============================================================

PW = (7, 7)
NQ = 101
PIX = (384, 384)
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))

# Paper geometry, normalized to a = 1
a_nm, h_nm, r = 445.0, 230.0, 0.405
energy_eV, hc_eV_nm = 0.800, 1239.841984
wavelength = (hc_eV_nm / energy_eV) / a_nm
h = h_nm / a_nm

eps_air = 1.0
eps_si = 3.48**2
eps_sub = 1.45**2 - 0.001j   # passive-loss convention in corrected Khepri

current = (1.0, 0.0, 0.0)    # x-oriented dipole


# ============================================================
# 2. REAL-SPACE TRIANGULAR LATTICE
# ============================================================

# Keep the physical lattice orientation; only the FBZ definition was corrected.
a1 = np.array([np.sqrt(3)/2, +0.5])
a2 = np.array([np.sqrt(3)/2, -0.5])
lattice = np.array([a1, a2])

# A/B/C assignment inferred from Fig. 1 geometry.
A = (a1 + a2) / 3
B = (a1 + a2) / 2
C = a1 / 2

dipole_pos = (float(B[0]), float(B[1]), h/2)

# B must remain in silicon, not inside an air hole.
if min(np.linalg.norm(B - (m*a1+n*a2))
       for m in range(-2, 3) for n in range(-2, 3)) <= r:
    raise RuntimeError("Chosen B point lies inside an air hole.")


# ============================================================
# 3. RECIPROCAL LATTICE + TRUE HORIZONTAL HEXAGONAL FBZ
# ============================================================

def reciprocal(u, v):
    d = u[0]*v[1] - u[1]*v[0]
    return ((2*np.pi/d) * np.array([v[1], -v[0]]),
            (2*np.pi/d) * np.array([-u[1], u[0]]))

b1, b2 = reciprocal(a1, a2)
Gs = np.array([b1, -b1, b2, -b2, b1+b2, -(b1+b2)])

Gamma = np.array([0.0, 0.0])
M = np.array([2*np.pi/np.sqrt(3), 0.0])
K = np.array([4*np.pi/3, 0.0])

BZ = np.array([
    [+4*np.pi/3, 0],
    [+2*np.pi/3, +2*np.pi/np.sqrt(3)],
    [-2*np.pi/3, +2*np.pi/np.sqrt(3)],
    [-4*np.pi/3, 0],
    [-2*np.pi/3, -2*np.pi/np.sqrt(3)],
    [+2*np.pi/3, -2*np.pi/np.sqrt(3)],
])
BZC = np.vstack([BZ, BZ[0]])

def in_bz(qx, qy):
    q = np.array([qx, qy])
    return np.all(Gs @ q <= 0.5*np.sum(Gs*Gs, axis=1) + 1e-12)


# ============================================================
# 4. BUILD / SOLVE KHEPRI
# ============================================================

def build_crystal():
    c = Crystal(
        pw=PW, lattice=lattice, lattice_pitch=1.0,
        epsi=eps_air, epse=eps_sub
    )
    d = Drawing(PIX, eps_si, c.lattice)
    d.disc((0.0, 0.0), r, eps_air)
    pat = d.canvas()

    # Split exactly at z=h/2, where the dipole is located.
    c.add_layer_pixmap("L", pat, h/2)
    c.add_layer_pixmap("R", pat, h/2)
    c.set_device(["L", "R"], fields_mask=[True, False])
    return c

def power(c, qx, qy, details=False):
    return dipole_power_q(
        crystal=c,
        source_after="L",
        wavelength=wavelength,
        kp=(float(qx), float(qy)),
        position=dipole_pos,
        current=current,
        backend="dense",
        return_details=details,
    )

# Each Windows worker gets its own stateful Crystal.
_worker_crystal = None

def init_worker():
    global _worker_crystal
    _worker_crystal = build_crystal()

def worker(task):
    iy, ix, qx, qy = task
    return iy, ix, float(power(_worker_crystal, qx, qy)["P_total"])


# ============================================================
# 5. CHECK P(q)=P(-q) BEFORE USING IT AS A SPEEDUP
# ============================================================

def reciprocity_ok(c):
    tests = [(0.7, 0.5), (-1.0, 1.1), (1.4, -0.8)]
    ok = True
    print("\nReciprocity check:")
    for qx, qy in tests:
        if not in_bz(qx, qy):
            continue
        p1 = power(c, qx, qy)["P_total"]
        p2 = power(c, -qx, -qy)["P_total"]
        rel = abs(p1-p2) / max(abs(p1), abs(p2), 1e-14)
        good = rel < 2e-5
        print(f"  ({qx:+.2f},{qy:+.2f})  rel.diff={rel:.2e}  "
              f"{'PASS' if good else 'FAIL'}")
        ok &= good
    return ok


# ============================================================
# 6. FAST q-MAP
# ============================================================

def qmap(c):
    qx = np.linspace(-4*np.pi/3, +4*np.pi/3, NQ)
    qy = np.linspace(-2*np.pi/np.sqrt(3), +2*np.pi/np.sqrt(3), NQ)
    QX, QY = np.meshgrid(qx, qy, indexing="xy")

    # Vectorized FBZ mask: much faster than Python-loop point testing.
    dots = np.stack([G[0]*QX + G[1]*QY for G in Gs])
    limits = (0.5*np.sum(Gs*Gs, axis=1))[:, None, None]
    mask = np.all(dots <= limits + 1e-12, axis=0)

    use_sym = reciprocity_ok(c)
    tasks = []

    for iy, ix in np.argwhere(mask):
        jy, jx = NQ-1-iy, NQ-1-ix
        if use_sym and (iy > jy or (iy == jy and ix > jx)):
            continue
        tasks.append((iy, ix, QX[iy, ix], QY[iy, ix]))

    P = np.full((NQ, NQ), np.nan)
    print(f"\nFBZ samples={mask.sum()}, RCWA solves={len(tasks)}, workers={WORKERS}")

    t0 = time.perf_counter()

    if WORKERS > 1:
        chunks = max(1, len(tasks)//(WORKERS*20))
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            initializer=init_worker
        ) as ex:
            for n, (iy, ix, val) in enumerate(
                ex.map(worker, tasks, chunksize=chunks), 1
            ):
                P[iy, ix] = val
                if use_sym:
                    P[NQ-1-iy, NQ-1-ix] = val
                if n % max(1, len(tasks)//10) == 0:
                    print(f"  {100*n/len(tasks):5.1f}%")
    else:
        for iy, ix, qxi, qyi in tasks:
            val = float(power(c, qxi, qyi)["P_total"])
            P[iy, ix] = val
            if use_sym:
                P[NQ-1-iy, NQ-1-ix] = val

    vals = P[mask]
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("Non-finite values found in q-map.")

    print(f"Finished in {time.perf_counter()-t0:.1f} s")
    print(f"P: min={vals.min():.4e}, max={vals.max():.4e}, mean={vals.mean():.4e}")
    return QX, QY, mask, P


# ============================================================
# 7. DISPLAY REAL SPACE + q SPACE
# ============================================================

def show_results(QX, QY, mask, P):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # ---------- real-space lattice ----------
    fig1, ax = plt.subplots(figsize=(7, 6))

    for m in range(-2, 3):
        for n in range(-2, 3):
            ctr = m*a1 + n*a2
            ax.add_patch(plt.Circle(ctr, r, fill=False))

    # Primitive cell
    cell = np.array([[0, 0], a1, a1+a2, a2, [0, 0]])
    ax.plot(cell[:, 0], cell[:, 1], linewidth=1.5)

    # A, B, C
    ax.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], s=[45, 100, 45])
    ax.text(A[0], A[1]+0.08, "A", ha="center")
    ax.text(B[0], B[1]+0.08, "B", ha="center")
    ax.text(C[0], C[1]+0.08, "C", ha="center")

    # x-oriented dipole at B
    ax.annotate(
        "", xy=(B[0]+0.22, B[1]), xytext=(B[0]-0.22, B[1]),
        arrowprops=dict(arrowstyle="->", linewidth=2)
    )
    ax.text(B[0], B[1]-0.14, "x-dipole", ha="center")

    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        title="Real-space inverse triangular PhC: B-point dipole"
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 2.1)
    ax.set_ylim(-1.3, 1.3)
    fig1.tight_layout()

    # ---------- q-space map ----------
    vals = P[mask]
    vmax = vals[vals > 0].max()
    z = np.log10(np.maximum(vals, vmax*1e-6)/vmax)

    tri = mtri.Triangulation(
        QX[mask]/(2*np.pi),
        QY[mask]/(2*np.pi)
    )

    fig2, ax = plt.subplots(figsize=(7, 6))
    im = ax.tricontourf(
        tri, z,
        levels=np.linspace(-6, 0, 121),
        cmap="hot",
        extend="min"
    )

    fig2.colorbar(
        im, ax=ax,
        label=r"$\log_{10}[P(\mathbf q)/P_{\max}]$"
    )

    ax.plot(BZC[:, 0]/(2*np.pi), BZC[:, 1]/(2*np.pi), linewidth=1.3)
    ax.scatter(
        [Gamma[0], M[0], K[0]],
        [Gamma[1], M[1], K[1]],
        s=25
    )

    ax.text(0, 0, r"$\Gamma$")
    ax.text(M[0]/(2*np.pi), M[1]/(2*np.pi), " M")
    ax.text(K[0]/(2*np.pi), K[1]/(2*np.pi), " K")

    ax.set(
        xlabel=r"$q_xa/(2\pi)$",
        ylabel=r"$q_ya/(2\pi)$",
        title="800 meV near-field power: B-point x-dipole"
    )
    ax.set_aspect("equal")
    fig2.tight_layout()

    # Display only — no image files are written.
    plt.show()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("PRB inverse triangular PhC | 800 meV | B-point x-dipole")
    print(f"pw={PW}, Nq={NQ}, workers={WORKERS}")
    print(f"A={A}, B={B}, C={C}")
    print(f"dipole (x,y,z)={dipole_pos}")

    c = build_crystal()

    # Source-jump sanity at Gamma.
    g = power(c, 0, 0, details=True)
    print(f"Gamma: P={g['P_total']:.6e}, jump={g['jump_relative_error']:.2e}")
    if g["jump_relative_error"] > 1e-8:
        raise RuntimeError("Dipole source-jump check failed.")

    QX, QY, mask, P = qmap(c)
    show_results(QX, QY, mask, P)


# Required for ProcessPoolExecutor on Windows.
if __name__ == "__main__":
    main()
