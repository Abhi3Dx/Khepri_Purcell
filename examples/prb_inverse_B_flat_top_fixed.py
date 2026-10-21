"""
PRB 108, 155416 (2023) - inverse triangular PhC
800 meV, B-point, x-oriented dipole

This version fixes the geometry/plotting issues:
  - only the B dipole is shown (A and C removed)
  - conventional triangular lattice: a1=(1,0), a2=(1/2,sqrt(3)/2)
  - reciprocal-space hexagon has a HORIZONTAL top/bottom side
  - Gamma, M, K are placed correctly
  - q-map mask and plotted FBZ use the SAME reciprocal lattice
  - hot colormap, linear normalization P(q)/Pmax
  - pw=(7,7), Nq=101, dense backend
  - multiprocessing + validated P(q)=P(-q) speedup
  - images are displayed only, never saved
  - numerical q-map is cached in a NEW cache file
"""

# One BLAS thread per worker avoids CPU oversubscription.
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
# 1. SETTINGS / PAPER PARAMETERS
# ============================================================

PW = (7, 7)
NQ = 101
PIX = (384, 384)
WORKERS = max(1, min(8, (os.cpu_count() or 2)//2))

# New name prevents reuse of the old, wrongly oriented map.
CACHE = "qmap_cache_flat_top_Bx.npz"
USE_CACHE = True

a_nm, h_nm, r = 445.0, 230.0, 0.405
energy_eV = 0.800
wavelength = (1239.841984/energy_eV)/a_nm   # normalized lambda/a
h = h_nm/a_nm

eps_air = 1.0
eps_si = 3.48**2
eps_sub = 1.45**2 - 0.001j
current = (1.0, 0.0, 0.0)                   # x dipole


# ============================================================
# 2. REAL-SPACE TRIANGULAR LATTICE + B DIPOLE
# ============================================================

# Standard triangular lattice.  This gives a flat-top reciprocal hexagon.
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])
lattice = np.array([a1, a2])

# The paper labels B graphically rather than giving Cartesian coordinates.
# We keep the same B-point assumption used in the previous reproduction:
# midpoint between the neighboring hole centers at a1 and a2.
B = (a1 + a2)/2
dipole_pos = (float(B[0]), float(B[1]), h/2)

# B must lie in silicon.
dB = min(np.linalg.norm(B - (m*a1+n*a2))
         for m in range(-2, 3) for n in range(-2, 3))
if dB <= r:
    raise RuntimeError("B lies inside an air hole.")


# ============================================================
# 3. RECIPROCAL LATTICE + FIRST BRILLOUIN ZONE
# ============================================================

def reciprocal(u, v):
    d = u[0]*v[1] - u[1]*v[0]
    b1 = (2*np.pi/d)*np.array([v[1], -v[0]])
    b2 = (2*np.pi/d)*np.array([-u[1], u[0]])
    return b1, b2

b1, b2 = reciprocal(a1, a2)

# Six nearest reciprocal-lattice vectors define the Wigner-Seitz FBZ.
Gs = np.array([b1, -b1, b2, -b2, b1+b2, -(b1+b2)])

# Flat-top hexagon: top edge is parallel to x.
BZ = np.array([
    [+4*np.pi/3, 0],
    [+2*np.pi/3, +2*np.pi/np.sqrt(3)],
    [-2*np.pi/3, +2*np.pi/np.sqrt(3)],
    [-4*np.pi/3, 0],
    [-2*np.pi/3, -2*np.pi/np.sqrt(3)],
    [+2*np.pi/3, -2*np.pi/np.sqrt(3)],
])
BZC = np.vstack([BZ, BZ[0]])

# Paper-style Gamma -> M -> K choice:
# M = midpoint of the upper-right BZ edge, K = right-hand corner.
Gamma = np.array([0.0, 0.0])
M = np.array([np.pi, np.pi/np.sqrt(3)])
K = np.array([4*np.pi/3, 0.0])

def in_bz(qx, qy):
    q = np.array([qx, qy])
    return np.all(Gs @ q <= 0.5*np.sum(Gs*Gs, axis=1) + 1e-12)


# ============================================================
# 4. KHEPRI MODEL
# ============================================================

def build_crystal():
    c = Crystal(
        pw=PW, lattice=lattice, lattice_pitch=1.0,
        epsi=eps_air, epse=eps_sub
    )

    d = Drawing(PIX, eps_si, c.lattice)
    d.disc((0.0, 0.0), r, eps_air)
    pattern = d.canvas()

    # Split the PhC slab at the dipole plane z=h/2.
    c.add_layer_pixmap("L", pattern, h/2)
    c.add_layer_pixmap("R", pattern, h/2)
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


# ============================================================
# 5. MULTIPROCESSING
# ============================================================

_worker_crystal = None

def init_worker():
    global _worker_crystal
    _worker_crystal = build_crystal()

def worker(task):
    iy, ix, qx, qy = task
    return iy, ix, float(power(_worker_crystal, qx, qy)["P_total"])


# ============================================================
# 6. VERIFY q <-> -q SYMMETRY BEFORE USING THE SPEEDUP
# ============================================================

def reciprocity_ok(c):
    tests = [(0.7, 0.5), (-1.0, 1.0), (1.4, -0.7)]
    ok = True

    print("\nReciprocity check:")
    for qx, qy in tests:
        if not in_bz(qx, qy):
            continue
        p1 = power(c, qx, qy)["P_total"]
        p2 = power(c, -qx, -qy)["P_total"]
        rel = abs(p1-p2)/max(abs(p1), abs(p2), 1e-14)
        good = rel < 2e-5
        print(f"  q=({qx:+.2f},{qy:+.2f})  rel.diff={rel:.2e}  "
              f"{'PASS' if good else 'FAIL'}")
        ok &= good

    return ok


# ============================================================
# 7. FAST q-MAP
# ============================================================

def calculate_qmap(c):
    # Bounding rectangle of the flat-top FBZ.
    qx = np.linspace(-4*np.pi/3, +4*np.pi/3, NQ)
    qy = np.linspace(-2*np.pi/np.sqrt(3), +2*np.pi/np.sqrt(3), NQ)
    QX, QY = np.meshgrid(qx, qy, indexing="xy")

    # Vectorized Wigner-Seitz mask.
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

    print(f"\nFBZ points={mask.sum()}, RCWA solves={len(tasks)}, workers={WORKERS}")
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
        raise RuntimeError("Non-finite values in q-map.")

    print(f"Finished in {time.perf_counter()-t0:.1f} s")
    print(f"P: min={vals.min():.4e}, max={vals.max():.4e}, mean={vals.mean():.4e}")
    return QX, QY, mask, P


# ============================================================
# 8. CACHE THE EXPENSIVE NUMERICAL MAP
# ============================================================

def load_cache():
    if not (USE_CACHE and os.path.exists(CACHE)):
        return None

    d = np.load(CACHE)
    valid = (
        int(d["NQ"]) == NQ
        and np.array_equal(d["PW"], np.array(PW))
        and np.allclose(d["a1"], a1)
        and np.allclose(d["a2"], a2)
        and np.allclose(d["B"], B)
    )

    if valid:
        print(f"\nLoaded corrected q-map from {CACHE}")
        return d["QX"], d["QY"], d["mask"], d["P"]

    print("\nCache geometry does not match; recalculating.")
    return None

def save_cache(QX, QY, mask, P):
    np.savez(
        CACHE, QX=QX, QY=QY, mask=mask, P=P,
        NQ=NQ, PW=np.array(PW), a1=a1, a2=a2, B=B
    )
    print(f"Cached numerical q-map -> {CACHE}")


# ============================================================
# 9. REAL-SPACE VIEWER: ONLY B DIPOLE
# ============================================================

def show_real_space():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # Air-hole lattice.
    for m in range(-2, 3):
        for n in range(-2, 3):
            p = m*a1 + n*a2
            ax.add_patch(plt.Circle(p, r, fill=False, linewidth=1.1))

    # Primitive cell containing the chosen B position.
    cell = np.array([[0, 0], a1, a1+a2, a2, [0, 0]])
    ax.plot(cell[:, 0], cell[:, 1], "--", linewidth=1.2)

    # B point and x-oriented dipole.
    ax.scatter(B[0], B[1], s=80, zorder=5)
    ax.text(B[0], B[1]+0.10, "B", ha="center", fontsize=13)
    ax.annotate(
        "",
        xy=(B[0]+0.23, B[1]),
        xytext=(B[0]-0.23, B[1]),
        arrowprops=dict(arrowstyle="->", linewidth=2),
    )
    ax.text(B[0], B[1]-0.15, "x-dipole", ha="center")

    ax.set(
        xlabel=r"$x/a$",
        ylabel=r"$y/a$",
        title="Real-space inverse triangular PhC: B-point x-dipole",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(-1.25, 1.55)
    fig.tight_layout()

    # Display only.
    plt.show()
    plt.close(fig)


# ============================================================
# 10. RECIPROCAL-SPACE VIEWER
# ============================================================

def show_qmap(QX, QY, mask, P):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    vals = P[mask]
    z = vals/np.max(vals)   # linear normalization, closer to paper Fig. 3

    # Plot in units of 2*pi/a.
    x = QX[mask]/(2*np.pi)
    y = QY[mask]/(2*np.pi)
    tri = mtri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.tricontourf(
        tri, z,
        levels=np.linspace(0, 1, 121),
        cmap="hot",
    )

    fig.colorbar(im, ax=ax, label=r"$P(\mathbf{q})/P_{\max}$")

    # FBZ boundary.
    ax.plot(
        BZC[:, 0]/(2*np.pi),
        BZC[:, 1]/(2*np.pi),
        "w-",
        linewidth=1.5,
    )

    # IMPORTANT: points are scaled by 2*pi exactly like the map.
    hs = np.array([Gamma, M, K])/(2*np.pi)
    ax.scatter(hs[:, 0], hs[:, 1], s=35, c="cyan", zorder=5)

    ax.text(hs[0,0]+0.015, hs[0,1]+0.015, r"$\Gamma$", color="cyan")
    ax.text(hs[1,0]+0.015, hs[1,1]+0.015, "M", color="cyan")
    ax.text(hs[2,0]+0.015, hs[2,1]+0.015, "K", color="cyan")

    ax.set(
        xlabel=r"$q_xa/(2\pi)$",
        ylabel=r"$q_ya/(2\pi)$",
        title="800 meV near-field emissivity: B-point x-dipole",
    )
    ax.set_aspect("equal")

    # Exact flat-top FBZ bounds.
    ax.set_xlim(-0.70, 0.70)
    ax.set_ylim(-0.62, 0.62)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print("PRB inverse triangular PhC | 800 meV | B-point x-dipole")
    print(f"pw={PW}, Nq={NQ}, workers={WORKERS}")
    print(f"a1={a1}")
    print(f"a2={a2}")
    print(f"B={B}, dipole z/a={h/2:.6f}")
    print(f"Gamma={Gamma}, M={M}, K={K}")

    # Show the actual real-space B position before the long calculation.
    show_real_space()

    cached = load_cache()

    if cached is None:
        c = build_crystal()

        # Dipole-source sanity check.
        g = power(c, 0, 0, details=True)
        print(f"\nGamma source check: P={g['P_total']:.6e}, "
              f"jump={g['jump_relative_error']:.2e}")

        if g["jump_relative_error"] > 1e-8:
            raise RuntimeError("Dipole source-jump check failed.")

        QX, QY, mask, P = calculate_qmap(c)
        save_cache(QX, QY, mask, P)
    else:
        QX, QY, mask, P = cached

    show_qmap(QX, QY, mask, P)


# Required for multiprocessing on Windows.
if __name__ == "__main__":
    main()
