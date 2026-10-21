"""
Direct radial-q planar-slab benchmark for the Khepri dipole/Purcell solver.

Why this test exists
--------------------
The previous 2D Brillouin-zone test used a very small substrate loss
(Im eps = -0.001 in Khepri's time convention) and coarse 11x11--23x23
q grids.  The resulting Purcell factor was strongly non-converged because
a planar high-index slab supports very narrow guided-mode features in q-space.

This test removes two complications:

1. The planar structure has no physical in-plane periodicity, so we use pw=(1,1)
   and integrate directly over the continuous radial in-plane wave number q.
2. We deliberately use a larger passive substrate loss, Im eps = -0.1 in
   Khepri's convention, to broaden the guided-mode poles.  The same physical
   substrate is +0.1j in the Chance et al. convention.

For an x-oriented dipole in an isotropic planar structure, the azimuthal
average at fixed q is exactly

    P_avg(q) = [P_x(q along x) + P_x(q along y)] / 2.

Then

    P = integral_0^infinity P_avg(q) q dq / (2*pi).

The script checks:
  A. homogeneous silicon -> Fp ~ 1
  B. air | Si | dipole | Si | lossy SiO2
     against the exact Chance/Miller/Prock/Silbey two-interface formula.

Run from the repository root:
    python test_purcell_planar_radial.py
"""

import numpy as np
from scipy.integrate import quad

from khepri.crystal import Crystal
from khepri.dipole import dipole_power_q, homogeneous_reference_power_xy


# ============================================================================
# 1. PARAMETERS
# ============================================================================

a_nm = 445.0
h_nm = 230.0
energy_eV = 0.800
hc_eV_nm = 1239.841984

wavelength_nm = hc_eV_nm / energy_eV

# Keep the same normalized length unit as the PRB test: a = 1.
h = h_nm / a_nm
half_h = h / 2.0
wavelength = wavelength_nm / a_nm

n_si = 3.48
n_sio2 = 1.45

eps_air = 1.0
eps_si = n_si**2
eps_sio2 = n_sio2**2

# Deliberately broaden the planar guided-mode poles for this validation.
loss_imag = 0.1

# Khepri uses the opposite harmonic-time sign from the Chance convention.
eps_substrate_khepri = eps_sio2 - 1j * loss_imag
eps_substrate_chance = eps_sio2 + 1j * loss_imag

current = (1.0, 0.0, 0.0)
reference_index = n_si
backend = "dense"

# The planar problem does not need reciprocal harmonics when q is integrated
# directly over the full continuous k_parallel plane.
pw = (1, 1)

k0 = 2.0 * np.pi / wavelength
k1 = n_si * k0

# Direct radial integration settings.
# q_max = 4*k1 is already deep in the exponentially suppressed high-q tail
# for a source 115 nm from each interface.
q_max_factor = 4.0
q_max = q_max_factor * k1

# Midpoint radial grids.  Start with the smaller values if runtime is an issue.
radial_sizes = (2001, 4001, 8001)


# ============================================================================
# 2. CHANCE ET AL. TWO-INTERFACE PARALLEL-DIPOLE FORMULA
# ============================================================================

def _ell(eps_j, eps_1, u):
    """
    Chance et al. Eq. (9):
        l_j = -i sqrt(eps_j/eps_1 - u^2)

    Choose the branch with Re(l_j) >= 0, and on the imaginary axis Im(l_j) <= 0.
    """
    value = -1j * np.sqrt(complex(eps_j / eps_1 - u * u))
    atol = 1e-13

    if (value.real < -atol) or (
        abs(value.real) <= atol and value.imag > 0.0
    ):
        value = -value

    return value


def _chance_parallel_bracket(u, eps_1, eps_2, eps_3, dbar, sbar):
    """
    Integrand bracket from Chance et al., JCP 63, 1589 (1975), Eq. (4).
    """
    l1 = _ell(eps_1, eps_1, u)
    l2 = _ell(eps_2, eps_1, u)
    l3 = _ell(eps_3, eps_1, u)

    # Eq. (5), perpendicular/s polarization.
    r_perp_12 = (l1 - l2) / (l1 + l2)
    r_perp_13 = (l1 - l3) / (l1 + l3)

    # Eq. (6)/(8), parallel/p polarization in Chance notation.
    r_para_12 = (eps_1 * l2 - eps_2 * l1) / (
        eps_1 * l2 + eps_2 * l1
    )
    r_para_13 = (eps_1 * l3 - eps_3 * l1) / (
        eps_1 * l3 + eps_3 * l1
    )

    def F(xbar, reflection):
        return 1.0 + reflection * np.exp(-2.0 * l1 * xbar)

    term_s = (
        F(dbar, r_perp_12)
        * F(sbar, r_perp_13)
        / F(dbar + sbar, -r_perp_12 * r_perp_13)
    )

    term_p = (
        F(dbar, r_para_12)
        * F(sbar, r_para_13)
        / F(dbar + sbar, -r_para_12 * r_para_13)
    )

    return term_s + (1.0 - u * u) * term_p


def chance_parallel_purcell(
    eps_1,
    eps_2,
    eps_3,
    dbar,
    sbar,
    epsabs=1e-10,
    epsrel=1e-9,
):
    """
    Normalized decay rate for a dipole parallel to two planar interfaces.

    For unit intrinsic quantum yield, Chance et al. Eq. (2)+(4) gives

        F_parallel = (3/4) Im integral_0^inf [...] u du / l1.

    The substitutions below remove the square-root point at u=1.
    """

    def propagating_integrand(w):
        u = np.sqrt(max(0.0, 1.0 - w * w))
        bracket = _chance_parallel_bracket(
            u, eps_1, eps_2, eps_3, dbar, sbar
        )
        return float(np.real(bracket))

    def evanescent_integrand(v):
        u = np.sqrt(1.0 + v * v)
        bracket = _chance_parallel_bracket(
            u, eps_1, eps_2, eps_3, dbar, sbar
        )
        return float(np.imag(bracket))

    I_prop, err_prop = quad(
        propagating_integrand,
        0.0,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=1000,
    )

    I_evan, err_evan = quad(
        evanescent_integrand,
        0.0,
        np.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=1000,
    )

    return {
        "F_purcell": 0.75 * (I_prop + I_evan),
        "I_propagating": I_prop,
        "I_evanescent": I_evan,
        "quad_error": err_prop + err_evan,
    }


# ============================================================================
# 3. KHEPRI BUILDERS
# ============================================================================

def build_planar(epse, epsi=eps_air):
    cl = Crystal(
        pw=pw,
        lattice="square",
        lattice_pitch=1.0,
        epsi=epsi,
        epse=epse,
    )

    cl.add_layer_uniform("si_left", eps_si, half_h)
    cl.add_layer_uniform("si_right", eps_si, half_h)

    cl.set_device(
        ["si_left", "si_right"],
        fields_mask=[True, False],
    )

    return cl


def build_homogeneous_si():
    return build_planar(epse=eps_si, epsi=eps_si)


def p_q(crystal, qx, qy):
    result = dipole_power_q(
        crystal=crystal,
        source_after="si_left",
        wavelength=wavelength,
        kp=(float(qx), float(qy)),
        position=(0.0, 0.0, half_h),
        current=current,
        backend=backend,
        return_details=False,
    )

    value = float(result["P_total"])

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite P_total at q=({qx:.12e},{qy:.12e})"
        )

    return value


def radial_integral(crystal, N):
    """
    Midpoint radial integration of

        integral P_avg(q) q dq / (2*pi).

    The x-dipole azimuthal average follows from planar rotational symmetry:
        P_avg = [P(q,0) + P(0,q)]/2.
    """
    dq = q_max / N
    q = (np.arange(N, dtype=float) + 0.5) * dq

    P_parallel_q = np.empty(N, dtype=float)
    P_perp_q = np.empty(N, dtype=float)

    for i, qi in enumerate(q):
        # q parallel to dipole axis x
        P_parallel_q[i] = p_q(crystal, qi, 0.0)

        # q perpendicular to dipole axis
        P_perp_q[i] = p_q(crystal, 0.0, qi)

    P_avg = 0.5 * (P_parallel_q + P_perp_q)

    integrand = P_avg * q / (2.0 * np.pi)
    P_integrated = float(np.sum(integrand) * dq)

    return {
        "N": N,
        "dq": dq,
        "q": q,
        "P_parallel_q": P_parallel_q,
        "P_perp_q": P_perp_q,
        "P_avg": P_avg,
        "P_min": float(np.min(P_avg)),
        "P_max": float(np.max(P_avg)),
        "P_integrated": P_integrated,
    }


# ============================================================================
# 4. RUN
# ============================================================================

print("=" * 78)
print("KHEPRI PLANAR-SLAB DIRECT RADIAL-q BENCHMARK")
print("=" * 78)

print("\nSETUP")
print(f"lambda/a                  = {wavelength:.12f}")
print(f"h/a                       = {h:.12f}")
print(f"k0                        = {k0:.12f}")
print(f"k_Si                      = {k1:.12f}")
print(f"eps_Si                    = {eps_si}")
print(f"eps_substrate Khepri      = {eps_substrate_khepri}")
print(f"eps_substrate Chance      = {eps_substrate_chance}")
print(f"pw                         = {pw}")
print(f"q_max                     = {q_max:.12f} = {q_max_factor:.1f} k_Si")
print(f"radial sizes               = {radial_sizes}")

P_reference = homogeneous_reference_power_xy(
    wavelength=wavelength,
    refractive_index=reference_index,
    current=current,
)

print(f"P_reference exact          = {P_reference:.12e}")


# ----------------------------------------------------------------------------
# A. Analytical planar result
# ----------------------------------------------------------------------------

dbar = k1 * half_h
sbar = k1 * half_h

analytic = chance_parallel_purcell(
    eps_1=eps_si,
    eps_2=eps_air,
    eps_3=eps_substrate_chance,
    dbar=dbar,
    sbar=sbar,
)

print("\nANALYTICAL CHANCE RESULT")
print(f"k1*d = k1*s               = {dbar:.12f}")
print(f"I_propagating              = {analytic['I_propagating']:.12e}")
print(f"I_evanescent               = {analytic['I_evanescent']:.12e}")
print(f"F_purcell analytic         = {analytic['F_purcell']:.12e}")
print(f"quadrature error estimate  = {analytic['quad_error']:.3e}")


# ----------------------------------------------------------------------------
# B. Homogeneous silicon direct radial check
# ----------------------------------------------------------------------------

print("\nHOMOGENEOUS SILICON RADIAL CHECK")
homogeneous = build_homogeneous_si()

# A modest grid is sufficient because there are no interfaces/guided resonances.
hom_N = 2001
hom = radial_integral(homogeneous, hom_N)
hom_F = hom["P_integrated"] / P_reference

print(f"N                          = {hom_N}")
print(f"P_integrated               = {hom['P_integrated']:.12e}")
print(f"F_purcell                  = {hom_F:.12e}")
print(f"|Fp - 1|                   = {abs(hom_F - 1.0):.3e}")

if abs(hom_F - 1.0) > 5e-3:
    print(
        "[WARNING] Homogeneous radial integral is not within 0.5% of unity.\n"
        "Increase q_max or N before interpreting the planar comparison."
    )
else:
    print("[PASS] homogeneous silicon radial normalization")


# ----------------------------------------------------------------------------
# C. Planar slab radial convergence
# ----------------------------------------------------------------------------

print("\nPLANAR SLAB RADIAL CONVERGENCE")
planar = build_planar(epse=eps_substrate_khepri, epsi=eps_air)

results = []

for N in radial_sizes:
    out = radial_integral(planar, N)
    Fp = out["P_integrated"] / P_reference
    err = (
        abs(Fp - analytic["F_purcell"])
        / abs(analytic["F_purcell"])
        * 100.0
    )

    results.append((N, Fp, err, out))

    print(
        f"N={N:5d}  "
        f"dq={out['dq']:.4e}  "
        f"minP={out['P_min']:+.4e}  "
        f"maxP={out['P_max']:+.4e}  "
        f"Fp={Fp:.10f}  "
        f"error={err:.3f}%"
    )


print("\n" + "=" * 78)
print("FINAL")
print("=" * 78)

N, Fp, err, out = results[-1]

print(f"Khepri radial Fp (N={N})   = {Fp:.12e}")
print(f"Analytical Fp              = {analytic['F_purcell']:.12e}")
print(f"relative difference        = {err:.6f} %")

if err <= 5.0:
    print(
        "\n[PASS] The dipole source + asymmetric complex planar interfaces "
        "agree with the independent analytical benchmark."
    )
    print(
        "The earlier 2D 11x11--23x23 BZ discrepancy was therefore a q-sampling "
        "problem caused by narrow guided-mode structure."
    )
else:
    print(
        "\n[WARNING] The direct radial benchmark is still outside 5%.\n"
        "Because this test removes reciprocal-cell folding and strongly broadens "
        "the guided poles, a persistent large discrepancy would point to the "
        "dipole/source-plane formulation rather than the old coarse q grid."
    )
