"""
Khepri planar Purcell-factor loss-convergence benchmark.

Purpose
-------
Validate the dipole/Purcell implementation in the weak-loss regime used for
our PRB reproduction by comparing Khepri against the independent two-interface
Sommerfeld result for an x-oriented dipole in

    air | Si(h/2) | x dipole | Si(h/2) | weakly lossy SiO2.

The loss sweep is

    |Im eps_sub| = 0.1, 0.03, 0.01, 0.003, 0.001.

Khepri uses eps_sub = eps_sio2 - i*loss with the complex-kz fix currently in
our fork.  The analytical Chance convention uses eps_sio2 + i*loss for the
same passive material.

Why this file is resonance-aware
--------------------------------
At small loss the guided-mode contribution becomes extremely narrow in q.
A coarse 2D Brillouin-zone grid can miss it completely.  The planar system is
rotationally invariant, so we integrate the continuous radial in-plane wave
number directly:

    P_avg(q) = 1/2 [P_x(q,0) + P_x(0,q)]

    P_structure = integral_0^infinity P_avg(q) q dq / (2*pi).

The script first locates the broad guided-mode peaks at loss=0.1, estimates
an HWHM for each one, then tracks each peak as loss is reduced.  Each narrow
resonance is integrated in a scaled coordinate while the smooth background is
integrated separately.

Run from the repository root:

    python test_purcell_planar_loss_convergence.py

Dense backend only.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from khepri.crystal import Crystal
from khepri.dipole import dipole_power_q, homogeneous_reference_power_xy


# ============================================================================
# 1. PHYSICAL PARAMETERS
# ============================================================================

a_nm = 445.0
h_nm = 230.0
energy_eV = 0.800
hc_eV_nm = 1239.841984
wavelength_nm = hc_eV_nm / energy_eV

# Normalize lengths by a = 445 nm.
a = 1.0
h = h_nm / a_nm
half_h = 0.5 * h
wavelength = wavelength_nm / a_nm

n_si = 3.48
n_sio2 = 1.45

eps_air = 1.0
eps_si = n_si**2
eps_sio2 = n_sio2**2

current = (1.0, 0.0, 0.0)
reference_index = n_si
backend = "dense"

# No reciprocal diffraction orders are required for the direct continuous-q
# planar benchmark.
pw = (1, 1)

k0 = 2.0 * np.pi / wavelength
k1 = n_si * k0

loss_values = (0.1, 0.03, 0.01, 0.003, 0.001)

# Conservative high-q cutoff.  The evanescent contribution is exponentially
# suppressed because the dipole is 115 nm from either interface.
q_max_factor = 6.0
q_max = q_max_factor * k1

# Broad-loss scan used only to identify guided-mode resonances.
initial_scan_points = 5001
q_guided_min = 0.98 * max(k0, n_sio2 * k0)
q_guided_max = 1.002 * k1

# Background integration tolerances.
quad_epsabs = 1e-8
quad_epsrel = 2e-6
quad_limit = 1000

# Final acceptance threshold.
comparison_tolerance_percent = 2.0

# Number of broad-loss HWHM on either side used for the narrow-mode integral.
# 60 is already far into the tails of a Lorentzian-like peak.
resonance_half_window_in_widths = 60.0


# ============================================================================
# 2. INDEPENDENT CHANCE / SOMMERFELD REFERENCE
# ============================================================================

def _ell(eps_j, eps_1, u):
    """Chance vertical-wave parameter l_j = -i sqrt(eps_j/eps_1-u^2)."""
    value = -1j * np.sqrt(complex(eps_j / eps_1 - u * u))
    atol = 1e-13
    if (value.real < -atol) or (abs(value.real) <= atol and value.imag > 0.0):
        value = -value
    return value


def _chance_parallel_bracket(u, eps_1, eps_2, eps_3, dbar, sbar):
    l1 = _ell(eps_1, eps_1, u)
    l2 = _ell(eps_2, eps_1, u)
    l3 = _ell(eps_3, eps_1, u)

    rs12 = (l1 - l2) / (l1 + l2)
    rs13 = (l1 - l3) / (l1 + l3)

    rp12 = (eps_1 * l2 - eps_2 * l1) / (eps_1 * l2 + eps_2 * l1)
    rp13 = (eps_1 * l3 - eps_3 * l1) / (eps_1 * l3 + eps_3 * l1)

    def F(xbar, reflection):
        return 1.0 + reflection * np.exp(-2.0 * l1 * xbar)

    term_s = F(dbar, rs12) * F(sbar, rs13) / F(
        dbar + sbar, -rs12 * rs13
    )
    term_p = F(dbar, rp12) * F(sbar, rp13) / F(
        dbar + sbar, -rp12 * rp13
    )

    return term_s + (1.0 - u * u) * term_p


def chance_parallel_purcell(eps_1, eps_2, eps_3, dbar, sbar):
    """Total normalized decay rate for a dipole parallel to two interfaces."""

    def prop_integrand(w):
        u = np.sqrt(max(0.0, 1.0 - w * w))
        bracket = _chance_parallel_bracket(u, eps_1, eps_2, eps_3, dbar, sbar)
        return float(np.real(bracket))

    def evan_integrand(v):
        u = np.sqrt(1.0 + v * v)
        bracket = _chance_parallel_bracket(u, eps_1, eps_2, eps_3, dbar, sbar)
        return float(np.imag(bracket))

    I_prop, err_prop = quad(
        prop_integrand, 0.0, 1.0,
        epsabs=1e-10, epsrel=1e-9, limit=1000,
    )
    I_evan, err_evan = quad(
        evan_integrand, 0.0, np.inf,
        epsabs=1e-10, epsrel=1e-9, limit=1000,
    )

    return {
        "F_purcell": float(0.75 * (I_prop + I_evan)),
        "I_propagating": float(I_prop),
        "I_evanescent": float(I_evan),
        "quad_error": float(err_prop + err_evan),
    }


# ============================================================================
# 3. KHEPRI PLANAR BUILDERS
# ============================================================================

def build_planar(eps_left, eps_right):
    crystal = Crystal(
        pw=pw,
        lattice="square",
        lattice_pitch=a,
        epsi=eps_left,
        epse=eps_right,
    )
    crystal.add_layer_uniform("si_left", eps_si, half_h)
    crystal.add_layer_uniform("si_right", eps_si, half_h)
    crystal.set_device(["si_left", "si_right"], fields_mask=[True, False])
    return crystal


def build_lossy_planar(loss_imag):
    return build_planar(
        eps_left=eps_air,
        eps_right=eps_sio2 - 1j * loss_imag,
    )


def build_homogeneous_si():
    return build_planar(eps_left=eps_si, eps_right=eps_si)


# ============================================================================
# 4. CACHED q-RESOLVED POWER
# ============================================================================

class RadialPower:
    def __init__(self, crystal):
        self.crystal = crystal
        self.cache_parallel = {}
        self.cache_perpendicular = {}

    @staticmethod
    def _key(q):
        return round(float(q), 13)

    def _solve(self, qx, qy):
        result = dipole_power_q(
            crystal=self.crystal,
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
            raise RuntimeError(f"Non-finite power at q=({qx},{qy}).")
        if value < -1e-8:
            raise RuntimeError(
                f"Significant negative power at q=({qx},{qy}): {value}"
            )
        return value

    def parallel(self, q):
        key = self._key(q)
        if key not in self.cache_parallel:
            self.cache_parallel[key] = self._solve(q, 0.0)
        return self.cache_parallel[key]

    def perpendicular(self, q):
        key = self._key(q)
        if key not in self.cache_perpendicular:
            self.cache_perpendicular[key] = self._solve(0.0, q)
        return self.cache_perpendicular[key]

    def average(self, q):
        return 0.5 * (self.parallel(q) + self.perpendicular(q))

    def integrand(self, q):
        return self.average(q) * q / (2.0 * np.pi)

    @property
    def evaluation_count(self):
        return len(self.cache_parallel) + len(self.cache_perpendicular)


# ============================================================================
# 5. BASIC PIECEWISE QUADRATURE
# ============================================================================

def integrate_piecewise(function, lower, upper, split_points=()):
    points = [float(lower)]
    for value in sorted(float(x) for x in split_points):
        if lower < value < upper and abs(value - points[-1]) > 1e-12:
            points.append(value)
    points.append(float(upper))

    total = 0.0
    error = 0.0
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        val, err = quad(
            function, left, right,
            epsabs=quad_epsabs,
            epsrel=quad_epsrel,
            limit=quad_limit,
        )
        total += val
        error += err
    return float(total), float(error)


# ============================================================================
# 6. FIND BROAD-LOSS GUIDED RESONANCES AND THEIR HWHM
# ============================================================================

def broad_loss_modes(radial_power):
    q = np.linspace(q_guided_min, q_guided_max, initial_scan_points)
    P = np.array([radial_power.average(qi) for qi in q], dtype=float)

    positive = P[P > 0]
    floor = max(np.min(positive) * 1e-3, 1e-30)
    logP = np.log10(np.maximum(P, floor))

    peaks, _ = find_peaks(logP, prominence=0.25)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(P))])

    modes = []
    for ip in peaks:
        peak = P[ip]

        # Local background estimate from a moderate region around the peak.
        i0 = max(0, ip - 150)
        i1 = min(len(P) - 1, ip + 150)
        background = min(P[i0], P[i1])
        half_level = background + 0.5 * (peak - background)

        left = ip
        while left > 0 and P[left] > half_level:
            left -= 1

        right = ip
        while right < len(P) - 1 and P[right] > half_level:
            right += 1

        if left == ip or right == ip:
            hwhm = 0.005 * k1
        else:
            left_width = q[ip] - q[left]
            right_width = q[right] - q[ip]
            hwhm = max(0.5 * (left_width + right_width), 1e-6 * k1)

        modes.append({
            "center": float(q[ip]),
            "broad_hwhm": float(hwhm),
            "broad_peak": float(peak),
        })

    # Sort and de-duplicate very close detections.
    modes.sort(key=lambda item: item["center"])
    dedup = []
    for mode in modes:
        if not dedup or mode["center"] - dedup[-1]["center"] > 3.0 * (
            q[1] - q[0]
        ):
            dedup.append(mode)
        elif mode["broad_peak"] > dedup[-1]["broad_peak"]:
            dedup[-1] = mode

    return dedup


# ============================================================================
# 7. TRACK AND INTEGRATE NARROW RESONANCES
# ============================================================================

def tracking_bounds(modes, index):
    center = modes[index]["center"]
    left_neighbor = modes[index - 1]["center"] if index > 0 else q_guided_min
    right_neighbor = (
        modes[index + 1]["center"] if index + 1 < len(modes) else q_guided_max
    )

    neighbor_limit = 0.35 * min(
        center - left_neighbor,
        right_neighbor - center,
    )
    half_window = min(
        max(6.0 * modes[index]["broad_hwhm"], 0.01 * k1),
        max(neighbor_limit, 0.005 * k1),
        0.08 * k1,
    )

    return (
        max(q_guided_min, center - half_window),
        min(q_guided_max, center + half_window),
    )


def track_modes(radial_power, previous_modes, loss_imag):
    tracked = []

    for index, old in enumerate(previous_modes):
        left, right = tracking_bounds(previous_modes, index)

        result = minimize_scalar(
            lambda q: -radial_power.average(q),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-10 * k1, "maxiter": 500},
        )
        if not result.success:
            raise RuntimeError(f"Peak tracking failed near q={old['center']}")

        center = float(result.x)
        peak = float(radial_power.average(center))

        # For loss-broadened guided modes the linewidth is approximately linear
        # in the small material loss.  Use the broad-loss measured HWHM as the
        # integration scale.  A generous +/-60 HWHM window makes this robust to
        # moderate deviations from exact linear scaling.
        scale = max(
            old["broad_hwhm"] * (loss_imag / loss_values[0]),
            2e-7 * k1,
        )

        half_window = resonance_half_window_in_widths * scale
        window_left = max(left, center - half_window)
        window_right = min(right, center + half_window)

        tracked.append({
            "center": center,
            "broad_hwhm": old["broad_hwhm"],
            "scale": float(scale),
            "peak": peak,
            "window_left": float(window_left),
            "window_right": float(window_right),
        })

    return tracked


def merge_windows(tracked):
    intervals = sorted(
        (m["window_left"], m["window_right"]) for m in tracked
    )
    if not intervals:
        return []

    merged = [list(intervals[0])]
    for left, right in intervals[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [tuple(x) for x in merged]


def integrate_single_scaled_mode(radial_power, mode):
    center = mode["center"]
    scale = mode["scale"]
    left = mode["window_left"]
    right = mode["window_right"]

    x_left = (left - center) / scale
    x_right = (right - center) / scale

    def f(x):
        q = center + scale * x
        return radial_power.integrand(q) * scale

    value, error = quad(
        f,
        x_left,
        x_right,
        points=[0.0] if x_left < 0.0 < x_right else None,
        epsabs=quad_epsabs,
        epsrel=quad_epsrel,
        limit=quad_limit,
    )
    return float(value), float(error)


def resonance_aware_integral(radial_power, previous_modes, loss_imag):
    tracked = track_modes(radial_power, previous_modes, loss_imag)
    windows = merge_windows(tracked)

    # Background: integrate everything except narrow resonance windows.
    q_air = k0
    q_sio2 = n_sio2 * k0
    q_si = k1

    background = 0.0
    background_error = 0.0
    cursor = 0.0

    for left, right in windows:
        if left > cursor:
            val, err = integrate_piecewise(
                radial_power.integrand,
                cursor,
                left,
                split_points=(q_air, q_sio2, q_si),
            )
            background += val
            background_error += err
        cursor = max(cursor, right)

    if cursor < q_max:
        val, err = integrate_piecewise(
            radial_power.integrand,
            cursor,
            q_max,
            split_points=(q_air, q_sio2, q_si),
        )
        background += val
        background_error += err

    # Resonance windows.  Isolated modes get the width-scaled integral.  If two
    # windows overlap, integrate the merged interval directly with both centers
    # as explicit split points.
    resonances = 0.0
    resonance_error = 0.0

    for left, right in windows:
        inside = [m for m in tracked if left <= m["center"] <= right]

        if len(inside) == 1:
            val, err = integrate_single_scaled_mode(radial_power, inside[0])
        else:
            val, err = integrate_piecewise(
                radial_power.integrand,
                left,
                right,
                split_points=[m["center"] for m in inside],
            )

        resonances += val
        resonance_error += err

    return {
        "P_total": float(background + resonances),
        "P_background": float(background),
        "P_resonances": float(resonances),
        "quad_error": float(background_error + resonance_error),
        "tracked": tracked,
    }


# ============================================================================
# 8. HOMOGENEOUS ADAPTIVE CHECK
# ============================================================================

def homogeneous_check():
    radial = RadialPower(build_homogeneous_si())
    P, error = integrate_piecewise(
        radial.integrand,
        0.0,
        q_max,
        split_points=(k1,),
    )
    P_ref = homogeneous_reference_power_xy(
        wavelength=wavelength,
        refractive_index=reference_index,
        current=current,
    )
    return P / P_ref, P, P_ref, error, radial.evaluation_count


# ============================================================================
# 9. RUN
# ============================================================================

print("=" * 88)
print("KHEPRI PLANAR PURCELL LOSS-CONVERGENCE BENCHMARK")
print("=" * 88)

print("\nSETUP")
print(f"lambda/a                         = {wavelength:.12f}")
print(f"h/a                              = {h:.12f}")
print(f"k0                               = {k0:.12f}")
print(f"k_Si                             = {k1:.12f}")
print(f"eps_Si                           = {eps_si}")
print(f"eps_SiO2                         = {eps_sio2}")
print(f"pw                               = {pw}")
print(f"backend                          = {backend}")
print(f"q_max                            = {q_max:.12f} = {q_max_factor:.1f} k_Si")
print(f"loss sweep                       = {loss_values}")

P_reference = homogeneous_reference_power_xy(
    wavelength=wavelength,
    refractive_index=reference_index,
    current=current,
)
print(f"P_reference                      = {P_reference:.12e}")


print("\n" + "=" * 88)
print("STEP 1 — HOMOGENEOUS SILICON")
print("=" * 88)

Fp_hom, P_hom, P_ref_hom, err_hom, eval_hom = homogeneous_check()
print(f"P_integrated                     = {P_hom:.12e}")
print(f"P_reference                      = {P_ref_hom:.12e}")
print(f"F_purcell                        = {Fp_hom:.12e}")
print(f"|Fp-1|                           = {abs(Fp_hom - 1.0):.3e}")
print(f"quadrature error estimate        = {err_hom:.3e}")
print(f"directional q evaluations        = {eval_hom}")

if abs(Fp_hom - 1.0) <= 5e-3:
    print("[PASS] homogeneous normalization")
else:
    print("[WARNING] homogeneous result is not within 0.5% of unity")


print("\n" + "=" * 88)
print("STEP 2 — FIND GUIDED MODES AT LOSS=0.1")
print("=" * 88)

broad_radial = RadialPower(build_lossy_planar(loss_values[0]))
modes = broad_loss_modes(broad_radial)

print(f"Detected {len(modes)} guided-mode candidate(s):")
for i, mode in enumerate(modes, start=1):
    print(
        f"  mode {i}: q={mode['center']:.12f}  "
        f"q/kSi={mode['center']/k1:.9f}  "
        f"broad HWHM={mode['broad_hwhm']:.6e}  "
        f"Ppeak={mode['broad_peak']:.6e}"
    )


print("\n" + "=" * 88)
print("STEP 3 — LOSS SWEEP")
print("=" * 88)

# Distances used by the analytical two-interface formula.
dbar = k1 * half_h
sbar = dbar

results = []
previous_modes = modes

for loss_imag in loss_values:
    print("\n" + "-" * 88)
    print(f"|Im eps_sub| = {loss_imag:g}")
    print("-" * 88)

    eps_sub_khepri = eps_sio2 - 1j * loss_imag
    eps_sub_analytic = eps_sio2 + 1j * loss_imag

    analytic = chance_parallel_purcell(
        eps_1=eps_si,
        eps_2=eps_air,
        eps_3=eps_sub_analytic,
        dbar=dbar,
        sbar=sbar,
    )

    radial = RadialPower(build_lossy_planar(loss_imag))
    numerical = resonance_aware_integral(radial, previous_modes, loss_imag)

    Fp_khepri = numerical["P_total"] / P_reference
    Fp_analytic = analytic["F_purcell"]
    rel_percent = (
        abs(Fp_khepri - Fp_analytic) / abs(Fp_analytic) * 100.0
    )

    print(f"eps substrate Khepri             = {eps_sub_khepri}")
    print(f"eps substrate analytical         = {eps_sub_analytic}")
    print(f"F_analytic                       = {Fp_analytic:.12e}")
    print(f"P_background                     = {numerical['P_background']:.12e}")
    print(f"P_resonances                     = {numerical['P_resonances']:.12e}")
    print(f"P_total                          = {numerical['P_total']:.12e}")
    print(f"F_Khepri                         = {Fp_khepri:.12e}")
    print(f"relative difference              = {rel_percent:.6f} %")
    print(f"quadrature error estimate        = {numerical['quad_error']:.3e}")
    print(f"directional q evaluations        = {radial.evaluation_count}")

    print("TRACKED MODES")
    new_modes = []
    for i, mode in enumerate(numerical["tracked"], start=1):
        print(
            f"  mode {i}: q={mode['center']:.12f}  "
            f"q/kSi={mode['center']/k1:.9f}  "
            f"Ppeak={mode['peak']:.6e}  "
            f"integration scale={mode['scale']:.6e}"
        )
        # Keep the original broad HWHM as the linewidth reference, but update
        # the center for the next smaller-loss search.
        new_modes.append({
            "center": mode["center"],
            "broad_hwhm": mode["broad_hwhm"],
            "broad_peak": mode["peak"],
        })

    previous_modes = new_modes

    results.append({
        "loss": float(loss_imag),
        "Fp_khepri": float(Fp_khepri),
        "Fp_analytic": float(Fp_analytic),
        "difference_percent": float(rel_percent),
    })


print("\n" + "=" * 88)
print("STEP 4 — SUMMARY")
print("=" * 88)

print(
    f"{'loss':>10s}  {'Fp Khepri':>16s}  "
    f"{'Fp analytic':>16s}  {'difference':>14s}"
)
for row in results:
    print(
        f"{row['loss']:10.4g}  "
        f"{row['Fp_khepri']:16.9f}  "
        f"{row['Fp_analytic']:16.9f}  "
        f"{row['difference_percent']:13.5f}%"
    )

final = results[-1]

print("\nFINAL PAPER-REGIME CHECK")
print(f"loss                             = {final['loss']}")
print(f"F_Khepri                         = {final['Fp_khepri']:.12e}")
print(f"F_analytic                       = {final['Fp_analytic']:.12e}")
print(f"relative difference              = {final['difference_percent']:.6f} %")
print(f"target tolerance                 = {comparison_tolerance_percent:.3f} %")

if final["difference_percent"] <= comparison_tolerance_percent:
    print(
        "\n[PASS] Khepri agrees with the independent planar benchmark "
        "at |Im eps_sub| = 0.001."
    )
    print(
        "The dipole source, complex-kz treatment, and weak-loss planar "
        "integration are validated for the PRB-style total Purcell calculation."
    )
else:
    print(
        "\n[WARNING] The final weak-loss result is outside the requested tolerance."
    )
    print(
        "Inspect the tracked mode centers and integration scales before changing "
        "the dipole equations."
    )
