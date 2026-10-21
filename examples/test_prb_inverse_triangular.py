"""
First triangular-lattice Purcell benchmark based on
Dyakov et al., Phys. Rev. B 108, 155416 (2023).

Purpose
-------
This script is deliberately a FIRST controlled test, not yet an exact
reproduction of the paper's B-point result.

It does four things:

1. Builds the paper's inverse triangular photonic-crystal geometry:
       air holes in a silicon slab on a silica substrate.

2. Uses a 7 x 7 RCWA plane-wave truncation.

3. Performs one single-q sanity check at Gamma:
       q = (0, 0).

4. Computes the dipole power over ONE COMPLETE reciprocal primitive
   cell and integrates it to obtain a total Purcell factor.

Important distinction
---------------------
A Purcell-factor calculation is NOT a calculation at only one k point.
The isolated point dipole is recovered by integrating the phased-array
RCWA result over a complete reciprocal primitive cell:

    integral P(q) d^2q / (2*pi)^2.

Therefore Gamma is used only as a sanity check. The final P_structure
and F_purcell printed by this script come from the full q-space map.

Dipole position used here
-------------------------
The paper states that point A is the most symmetric point between pores,
but does not print Cartesian A/B/C coordinates in the main text.

For this first benchmark we therefore use the natural high-symmetry
triangular interstitial point

    rho_A = (a1 + a2) / 3,

which is equidistant from the three nearest hole centers at 0, a1, a2.
The dipole is placed at the slab midplane z = h/2 and is x-oriented.

This is an A-like symmetry benchmark. It is NOT yet the exact B-point
benchmark used in Fig. 3(a) of the paper.

Units
-----
Khepri is run in normalized units with lattice constant a = 1.
The physical geometry is converted to normalized units before solving.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from khepri.crystal import Crystal
from khepri.draw import Drawing

from khepri.dipole import (
    dipole_power_q,
    dipole_power_map_unitcell,
    purcell_factor_from_unitcell_map,
    reciprocal_vectors_from_unitcell,
)


# ============================================================================
# 1. PAPER PARAMETERS
# ============================================================================

# Physical geometry from PRB 108, 155416.
a_nm = 445.0
h_nm = 230.0
r_over_a = 0.405

# Target photon energy used prominently in the paper.
energy_eV = 0.800
hc_eV_nm = 1239.841984
wavelength_nm = hc_eV_nm / energy_eV

# Normalize all lengths to a = 1.
a = 1.0
h = h_nm / a_nm
r = r_over_a
wavelength = wavelength_nm / a_nm
half_h = h / 2.0

# Approximate refractive indices near 1550 nm.
# The main PRB text gives the geometry but does not state the exact numerical
# Si / SiO2 dispersion values used by the authors, so these are starting values.
n_si = 3.48
n_sio2 = 1.45

eps_air = 1.0
eps_si = n_si ** 2
eps_sio2 = n_sio2 ** 2

# The paper adds +0.001i to the substrate permittivity when calculating the
# TOTAL Purcell factor to regularize extremely narrow guided-mode poles.
eps_substrate = eps_sio2 + 0.001j

# Purcell reference medium: silicon, because our dipole is embedded in silicon.
reference_index = n_si


# ============================================================================
# 2. NUMERICAL SETTINGS
# ============================================================================

# Requested plane-wave truncation.
pw = (7, 7)

# Real-space pixmap used to represent the circular air hole.
pixmap_resolution = (512, 512)

# Reciprocal primitive-cell sampling.
# 11 x 11 = 121 RCWA solves for the full map.
Nu = 11
Nv = 11

# A generic shifted grid avoids accidentally landing exactly on a
# Rayleigh/grazing singularity. Gamma is tested separately below.
q_shift = (0.23, 0.17)

# x-oriented electric dipole.
current = (1.0, 0.0, 0.0)

# Dense backend only, matching the dipole implementation we validated.
backend = "dense"


# ============================================================================
# 3. TRIANGULAR REAL-SPACE LATTICE
# ============================================================================

# Primitive vectors of a triangular Bravais lattice with a = 1.
a1 = np.array([
    1.0,
    0.0,
])

a2 = np.array([
    0.5,
    np.sqrt(3.0) / 2.0,
])

# Reciprocal vectors are used only for diagnostics / plotting here.
b1, b2, A_real, A_recip = reciprocal_vectors_from_unitcell(
    a1=a1,
    a2=a2,
)


# ============================================================================
# 4. DIPOLE POSITION
# ============================================================================

# Natural high-symmetry triangular interstitial point.
rho_A = (a1 + a2) / 3.0

position = (
    float(rho_A[0]),
    float(rho_A[1]),
    half_h,
)

# Nearest-hole-center distance. For this point it should be 1/sqrt(3).
nearest_hole_distance = np.linalg.norm(rho_A)

if nearest_hole_distance <= r:
    raise RuntimeError(
        "Chosen dipole position lies inside the air hole. "
        f"distance={nearest_hole_distance}, r={r}"
    )


# ============================================================================
# 5. PRINT SETUP
# ============================================================================

print("=" * 78)
print("PRB INVERSE TRIANGULAR PhC — FIRST PURCELL BENCHMARK")
print("=" * 78)

print("\nPHYSICAL GEOMETRY")
print(f"a                 = {a_nm:.6f} nm")
print(f"h                 = {h_nm:.6f} nm")
print(f"r                 = {r_over_a * a_nm:.6f} nm = {r_over_a:.6f} a")
print(f"target energy     = {energy_eV * 1000.0:.3f} meV")
print(f"vacuum wavelength = {wavelength_nm:.6f} nm")

print("\nNORMALIZED GEOMETRY")
print(f"a                 = {a:.12f}")
print(f"h/a               = {h:.12f}")
print(f"r/a               = {r:.12f}")
print(f"lambda/a          = {wavelength:.12f}")
print(f"a/lambda          = {1.0 / wavelength:.12f}")

print("\nMATERIALS")
print(f"n_Si              = {n_si:.6f}")
print(f"epsilon_Si        = {eps_si:.12f}")
print(f"n_SiO2            = {n_sio2:.6f}")
print(f"epsilon_substrate = {eps_substrate}")

print("\nRCWA SETTINGS")
print(f"plane waves       = {pw}")
print(f"q grid            = {Nu} x {Nv} = {Nu * Nv} points")
print(f"q-grid shift      = {q_shift}")
print(f"backend           = {backend}")

print("\nREAL-SPACE PRIMITIVE VECTORS")
print("a1 =", a1)
print("a2 =", a2)

print("\nRECIPROCAL PRIMITIVE VECTORS")
print("b1 =", b1)
print("b2 =", b2)
print(f"real-cell area       = {A_real:.12f}")
print(f"reciprocal-cell area = {A_recip:.12f}")

print("\nDIPOLE")
print("orientation       = x")
print(f"rho/a             = ({position[0]:.12f}, {position[1]:.12f})")
print(f"z/a               = {position[2]:.12f} = h/2")
print(
    "rho physical      = "
    f"({position[0] * a_nm:.6f}, {position[1] * a_nm:.6f}) nm"
)
print(f"z physical        = {position[2] * a_nm:.6f} nm")
print(f"distance to nearest hole center = {nearest_hole_distance:.12f} a")
print(f"hole radius                    = {r:.12f} a")


# ============================================================================
# 6. BUILD THE KHEPRI CRYSTAL
# ============================================================================

# epsi = incidence-side exterior = silica substrate
# epse = emergence-side exterior = air
#
# The structure runs from substrate -> patterned slab -> air along +z.
cl = Crystal(
    pw=pw,
    lattice="hexagonal",
    lattice_pitch=a,
    epsi=eps_air,
    epse=eps_substrate,
)

# Silicon background with one circular air pore at the triangular-lattice site.
drawing = Drawing(
    pixmap_resolution,
    eps_si,
    cl.lattice,
)

drawing.disc(
    (0.0, 0.0),
    r,
    eps_air,
)

epsilon_pattern = drawing.canvas()

# Split the physical PhC slab at z = h/2 so that the dipole source plane lies
# exactly between two identical patterned half-layers.
cl.add_layer_pixmap(
    "phc_lower",
    epsilon_pattern,
    half_h,
)

cl.add_layer_pixmap(
    "phc_upper",
    epsilon_pattern,
    half_h,
)

# fields_mask keeps the partial scattering matrices needed at the internal
# source plane after phc_lower.
cl.set_device(
    [
        "phc_lower",
        "phc_upper",
    ],
    fields_mask=[
        True,
        False,
    ],
)


# ============================================================================
# 7. SINGLE-k SANITY CHECK AT GAMMA
# ============================================================================

# Gamma means Bloch wavevector q = 0.
# This is NOT the final Purcell factor; it is only one point of the q integral.
print("\n" + "=" * 78)
print("STEP 1 — SINGLE-q SANITY CHECK AT GAMMA")
print("=" * 78)

Gamma = (0.0, 0.0)

Gamma_result = dipole_power_q(
    crystal=cl,
    source_after="phc_lower",
    wavelength=wavelength,
    kp=Gamma,
    position=position,
    current=current,
    backend=backend,
    return_details=True,
)

print(f"Gamma q = {Gamma}")
print(f"P_left  = {Gamma_result['P_left']:.12e}")
print(f"P_right = {Gamma_result['P_right']:.12e}")
print(f"P_total = {Gamma_result['P_total']:.12e}")
print(
    "source jump relative error = "
    f"{Gamma_result['jump_relative_error']:.3e}"
)

if not np.isfinite(Gamma_result["P_total"]):
    raise RuntimeError("Gamma-point dipole power is not finite.")

if Gamma_result["jump_relative_error"] > 1e-8:
    raise RuntimeError("Gamma-point source jump check failed.")

print("[PASS] Gamma-point source solve")


# ============================================================================
# 8. FULL RECIPROCAL-PRIMITIVE-CELL POWER MAP
# ============================================================================

print("\n" + "=" * 78)
print("STEP 2 — FULL RECIPROCAL-CELL DIPOLE POWER MAP")
print("=" * 78)
print(
    "This is the quantity required for the isolated-dipole Purcell factor.\n"
    "Each q point corresponds to one phased periodic dipole array."
)

map_result = dipole_power_map_unitcell(
    crystal=cl,
    source_after="phc_lower",
    wavelength=wavelength,
    a1=a1,
    a2=a2,
    Nu=Nu,
    Nv=Nv,
    position=position,
    current=current,
    shift=q_shift,
    backend=backend,
    verbose=True,
    return_point_details=False,
)

P_map = map_result["P_total"]
QX = map_result["qx"]
QY = map_result["qy"]

print("\nMAP DIAGNOSTICS")
print(f"min(P_total) = {np.min(P_map):.12e}")
print(f"max(P_total) = {np.max(P_map):.12e}")
print(f"mean(P_total)= {np.mean(P_map):.12e}")
print(f"finite map   = {np.all(np.isfinite(P_map))}")

if not np.all(np.isfinite(P_map)):
    raise RuntimeError("q-space power map contains NaN or infinity.")


# ============================================================================
# 9. INTEGRATE MAP -> TOTAL PURCELL FACTOR
# ============================================================================

print("\n" + "=" * 78)
print("STEP 3 — RECIPROCAL-CELL INTEGRATION")
print("=" * 78)

purcell = purcell_factor_from_unitcell_map(
    map_result=map_result,
    wavelength=wavelength,
    reference_index=reference_index,
    current=current,
    component="P_total",
)

print(f"P_structure = {purcell['P_structure']:.12e}")
print(f"P_reference = {purcell['P_reference']:.12e}")
print(f"F_purcell   = {purcell['F_purcell']:.12e}")

if not np.isfinite(purcell["F_purcell"]):
    raise RuntimeError("Purcell factor is not finite.")





# ============================================================================
# 11. PLOT q-SPACE POWER MAP
# ============================================================================

# Plot q in dimensionless units q*a/(2*pi). Since a=1 here this is q/(2*pi).
QX_plot = QX / (2.0 * np.pi)
QY_plot = QY / (2.0 * np.pi)

fig, ax = plt.subplots(
    figsize=(6.5, 6.0),
)

pcm = ax.pcolormesh(
    QX_plot,
    QY_plot,
    P_map,
    shading="auto",
)

cbar = fig.colorbar(
    pcm,
    ax=ax,
)

cbar.set_label(
    r"$P_{\rm total}(\mathbf{q})$  [Khepri normalized]"
)

# Reciprocal primitive-cell corners corresponding to u,v = +/-1/2.
cell_corners = np.array([
    -0.5 * b1 - 0.5 * b2,
    +0.5 * b1 - 0.5 * b2,
    +0.5 * b1 + 0.5 * b2,
    -0.5 * b1 + 0.5 * b2,
    -0.5 * b1 - 0.5 * b2,
]) / (2.0 * np.pi)

ax.plot(
    cell_corners[:, 0],
    cell_corners[:, 1],
    "k--",
    linewidth=1.2,
    label="reciprocal primitive cell",
)

# Mark Gamma even though the shifted integration grid does not necessarily
# sample it directly.
ax.plot(
    0.0,
    0.0,
    "ko",
    markersize=5,
    label=r"$\Gamma$",
)

ax.set_xlabel(
    r"$q_x a/(2\pi)$"
)

ax.set_ylabel(
    r"$q_y a/(2\pi)$"
)

ax.set_title(
    "Inverse triangular PhC, 800 meV\n"
    "x-dipole at A-like interstitial point, pw=(7,7)"
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.legend(
    loc="best",
    fontsize=8,
)

fig.tight_layout()



plt.show()


# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 78)
print("RUN COMPLETE")
print("=" * 78)
print("Single-q check : Gamma = (0,0)")
print("Final Purcell  : integrated over complete reciprocal primitive cell")
print("Dipole         : x-oriented")
print("Position       : rho=(a1+a2)/3, z=h/2")
print("Plane waves    : (7,7)")
print(f"q points       : {Nu} x {Nv} = {Nu * Nv}")
print(f"F_purcell      : {purcell['F_purcell']:.12e}")
print("=" * 78)
