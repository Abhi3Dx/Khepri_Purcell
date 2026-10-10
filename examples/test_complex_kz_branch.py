import numpy as np
from khepri.tools import outgoing_kz


def check_square(z2, label):
    kz = outgoing_kz(z2)
    err = abs(kz * kz - z2)
    print(f'{label:28s} z2={z2!r:>20}  kz={kz!r:>28}  square error={err:.3e}')
    assert err < 1e-11 * max(1.0, abs(z2))
    return kz


print('KHEPRI COMPLEX-kz BRANCH TEST')
print('=' * 72)

kz = check_square(2.0 + 0j, 'propagating real')
assert kz.real > 0 and abs(kz.imag) < 1e-12

kz = check_square(-2.0 + 0j, 'evanescent real')
assert abs(kz.real) < 1e-12 and kz.imag < 0

# Under Khepri's exp(-i*kz*z) forward-wave convention, passive loss has
# Im(epsilon) < 0.  At normal incidence the selected kz must therefore have
# positive real part and negative imaginary part.
eps_passive = 2.1025 - 0.001j
kz = check_square(eps_passive, 'passive complex epsilon')
assert kz.real > 0
assert kz.imag < 0

# Opposite sign corresponds to the opposite material-loss convention.
eps_opposite = 2.1025 + 0.001j
kz = check_square(eps_opposite, 'opposite-sign epsilon')
assert kz.real > 0
assert kz.imag > 0

# Vector input should work too.
z2 = np.array([1.0 + 0j, -1.0 + 0j, 2.1025 - 0.001j])
kz = outgoing_kz(z2)
assert np.allclose(kz * kz, z2, rtol=1e-12, atol=1e-12)

print('\n[PASS] outgoing_kz preserves kz^2 and selects the Khepri branch.')
