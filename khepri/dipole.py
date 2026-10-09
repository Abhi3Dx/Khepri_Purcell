import numpy as np

from .alternative import free_space_eigenmodes
from .fields import layer_eigenbasis_matrix


# ============================================================
# INTERNAL VALIDATION HELPERS
# ============================================================

def _as_complex_1d(array, name):
    """
    Convert input to a one-dimensional complex NumPy array.
    """

    array = np.asarray(
        array,
        dtype=np.complex128,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array."
        )

    return array


def _check_same_shape(a, b, name_a, name_b):
    """
    Ensure two arrays have identical shapes.
    """

    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape. "
            f"Got {a.shape} and {b.shape}."
        )


# ============================================================
# 1. FOURIER REPRESENTATION OF THE DIPOLE CURRENT
# ============================================================

def dipole_fourier_current(
    expansion,
    kp,
    position,
    current,
):
    """
    Construct the Fourier components of a periodic array
    of electric dipoles.

    For each reciprocal-lattice harmonic G,

        J_alpha(G)
            = j_alpha
              exp[-i (q + G) . rho_d]

    where

        q     = (qx, qy)
        G     = (Gx, Gy)
        rho_d = (xd, yd)

    Parameters
    ----------
    expansion : khepri.expansion.Expansion
        Khepri reciprocal-space expansion.

    kp : tuple of float
        Bloch in-plane wavevector

            (qx, qy)

        in the same reciprocal-space units as
        ``expansion.g_vectors``.

    position : tuple of float
        Dipole position

            (xd, yd, zd)

        Only xd and yd enter the Fourier phase.
        zd determines the source plane in the layer stack.

    current : tuple of complex
        Electric-current amplitudes

            (jx, jy, jz)

    Returns
    -------
    Jx, Jy, Jz : ndarray
        Fourier current components for every retained
        reciprocal-lattice harmonic.

    Notes
    -----
    This function generates Jz as well, but the source-jump
    implementation below currently supports only x/y dipoles.
    """

    if len(kp) != 2:
        raise ValueError(
            "kp must contain exactly (qx, qy)."
        )

    if len(position) != 3:
        raise ValueError(
            "position must contain exactly (xd, yd, zd)."
        )

    if len(current) != 3:
        raise ValueError(
            "current must contain exactly (jx, jy, jz)."
        )

    # --------------------------------------------------------
    # Bloch vector
    # --------------------------------------------------------

    qx, qy = kp


    # --------------------------------------------------------
    # Dipole position
    # --------------------------------------------------------

    xd, yd, _zd = position


    # --------------------------------------------------------
    # Dipole current
    # --------------------------------------------------------

    jx, jy, jz = current


    # --------------------------------------------------------
    # Reciprocal-lattice vectors
    #
    # shape:
    #
    #     (2, N_G)
    #
    # row 0 -> Gx
    # row 1 -> Gy
    # --------------------------------------------------------

    g = np.asarray(
        expansion.g_vectors
    )

    if g.ndim != 2 or g.shape[0] != 2:
        raise ValueError(
            "expansion.g_vectors must have shape (2, N_G)."
        )

    Gx = g[0]
    Gy = g[1]


    # --------------------------------------------------------
    # Fourier phase
    # --------------------------------------------------------

    phase = np.exp(
        -1j
        * (
            (qx + Gx) * xd
            +
            (qy + Gy) * yd
        )
    )


    # --------------------------------------------------------
    # Fourier current
    # --------------------------------------------------------

    Jx = np.asarray(
        jx * phase,
        dtype=np.complex128,
    )

    Jy = np.asarray(
        jy * phase,
        dtype=np.complex128,
    )

    Jz = np.asarray(
        jz * phase,
        dtype=np.complex128,
    )


    return Jx, Jy, Jz


# ============================================================
# 2. ELECTROMAGNETIC SOURCE JUMP
# ============================================================

def dipole_source_jump_xy(
    Jx,
    Jy,
):
    """
    Construct the source jump for an in-plane electric dipole
    using Khepri's transverse-field convention.

    Khepri's transverse state is

        psi = [Ex, Ey, Ux, Uy]

    where numerical verification gives

        U = -i H.

    For an electric surface current,

        Delta Hx = +Jy
        Delta Hy = -Jx

    therefore

        Delta Ux = -i Jy
        Delta Uy = +i Jx.

    Hence

        b = [
            Delta Ex,
            Delta Ey,
            Delta Ux,
            Delta Uy
        ]

          = [
            0,
            0,
            -i Jy,
            +i Jx
        ].

    Parameters
    ----------
    Jx, Jy : ndarray
        Fourier components of the in-plane electric current.

    Returns
    -------
    b : ndarray
        Khepri source-jump vector with shape

            (4 * N_G,)

    Notes
    -----
    This function intentionally supports only x/y dipoles.

    A z-oriented dipole requires additional terms involving
    Kx, Ky and the appropriate Fourier permittivity operator.
    """

    Jx = _as_complex_1d(
        Jx,
        "Jx",
    )

    Jy = _as_complex_1d(
        Jy,
        "Jy",
    )

    _check_same_shape(
        Jx,
        Jy,
        "Jx",
        "Jy",
    )


    zeros = np.zeros_like(
        Jx,
        dtype=np.complex128,
    )


    b = np.concatenate(
        [
            zeros,        # Delta Ex
            zeros,        # Delta Ey
            -1j * Jy,     # Delta Ux
            +1j * Jx,     # Delta Uy
        ]
    )


    return b


# ============================================================
# 3. SOLVE THE DIPOLE SOURCE PLANE
# ============================================================

def solve_dipole_source_plane(
    R0,
    Sl,
    Sr,
    b,
):
    """
    Solve the self-consistent dipole radiation problem at the
    source plane.

    Parameters
    ----------
    R0 : ndarray
        Khepri reference eigenbasis matrix

            R0 = [[W0,  W0],
                  [-V0, V0]]

        mapping reference modal amplitudes to

            [Ex, Ey, Ux, Uy].

    Sl : ndarray
        Partial scattering matrix containing everything
        to the LEFT of the dipole source plane.

    Sr : ndarray
        Partial scattering matrix containing everything
        to the RIGHT of the dipole source plane.

    b : ndarray
        Electromagnetic source jump in Khepri's convention

            [Delta Ex,
             Delta Ey,
             Delta Ux,
             Delta Uy].

    Returns
    -------
    c_plus : ndarray
        Dipole-generated modal amplitudes travelling
        toward +z / right / emergence.

    c_minus : ndarray
        Dipole-generated modal amplitudes travelling
        toward -z / left / incidence.

    Notes
    -----
    The source-plane fields satisfy

        psi_right - psi_left = b.

    Multiple reflections between the left and right parts of
    the structure are included self-consistently.
    """

    R0 = np.asarray(
        R0,
        dtype=np.complex128,
    )

    Sl = np.asarray(
        Sl,
        dtype=np.complex128,
    )

    Sr = np.asarray(
        Sr,
        dtype=np.complex128,
    )

    b = _as_complex_1d(
        b,
        "b",
    )


    # --------------------------------------------------------
    # Basic dimensional checks
    # --------------------------------------------------------

    if R0.ndim != 2:
        raise ValueError(
            "R0 must be a two-dimensional matrix."
        )

    if R0.shape[0] != R0.shape[1]:
        raise ValueError(
            "R0 must be square."
        )

    if R0.shape[0] != len(b):
        raise ValueError(
            "R0 dimension must match the source vector b."
        )

    if Sl.ndim != 4 or Sl.shape[:2] != (2, 2):
        raise ValueError(
            "Sl must have shape (2, 2, N, N)."
        )

    if Sr.ndim != 4 or Sr.shape[:2] != (2, 2):
        raise ValueError(
            "Sr must have shape (2, 2, N, N)."
        )


    # --------------------------------------------------------
    # Convert physical source jump to Khepri's common
    # reference modal basis.
    # --------------------------------------------------------

    source_modal = np.linalg.solve(
        R0,
        b,
    )


    if len(source_modal) % 2 != 0:
        raise ValueError(
            "Reference modal source must contain an even "
            "number of components."
        )


    s_plus, s_minus = np.split(
        source_modal,
        2,
    )


    n = len(
        s_plus
    )


    if Sl.shape[2:] != (n, n):
        raise ValueError(
            "Sl modal dimension is inconsistent with R0."
        )

    if Sr.shape[2:] != (n, n):
        raise ValueError(
            "Sr modal dimension is inconsistent with R0."
        )


    # --------------------------------------------------------
    # Reflection operators seen from source plane
    # --------------------------------------------------------

    # Wave travelling left sees this reflection operator.
    R_left = Sl[
        1,
        1,
    ]

    # Wave travelling right sees this reflection operator.
    R_right = Sr[
        0,
        0,
    ]


    # --------------------------------------------------------
    # Source-plane equations
    #
    # psi_right - psi_left = b
    #
    # gives:
    #
    # c_plus - R_left c_minus = s_plus
    #
    # R_right c_plus - c_minus = s_minus
    # --------------------------------------------------------

    I = np.eye(
        n,
        dtype=np.complex128,
    )


    A = np.block(
        [
            [
                I,
                -R_left,
            ],
            [
                R_right,
                -I,
            ],
        ]
    )


    rhs = np.concatenate(
        [
            s_plus,
            s_minus,
        ]
    )


    solution = np.linalg.solve(
        A,
        rhs,
    )


    c_plus = solution[
        :n
    ]

    c_minus = solution[
        n:
    ]


    return c_plus, c_minus


# ============================================================
# 4. RECONSTRUCT FIELDS IMMEDIATELY AROUND THE DIPOLE
# ============================================================

def dipole_source_plane_fields(
    R0,
    Sl,
    Sr,
    c_plus,
    c_minus,
):
    """
    Reconstruct transverse Fourier fields immediately on the
    two sides of the dipole plane.

    Parameters
    ----------
    R0 : ndarray
        Khepri reference eigenbasis matrix.

    Sl, Sr : ndarray
        Left and right partial scattering matrices.

    c_plus : ndarray
        Dipole-generated wave travelling toward +z.

    c_minus : ndarray
        Dipole-generated wave travelling toward -z.

    Returns
    -------
    psi_left : ndarray
        Fourier field state immediately to the left of source:

            [Ex, Ey, Ux, Uy].

    psi_right : ndarray
        Fourier field state immediately to the right of source:

            [Ex, Ey, Ux, Uy].

    Notes
    -----
    These fields should satisfy

        psi_right - psi_left = b

    where b is the source jump used in
    ``solve_dipole_source_plane``.
    """

    R0 = np.asarray(
        R0,
        dtype=np.complex128,
    )

    Sl = np.asarray(
        Sl,
        dtype=np.complex128,
    )

    Sr = np.asarray(
        Sr,
        dtype=np.complex128,
    )

    c_plus = _as_complex_1d(
        c_plus,
        "c_plus",
    )

    c_minus = _as_complex_1d(
        c_minus,
        "c_minus",
    )


    _check_same_shape(
        c_plus,
        c_minus,
        "c_plus",
        "c_minus",
    )


    R_left = Sl[
        1,
        1,
    ]

    R_right = Sr[
        0,
        0,
    ]


    # --------------------------------------------------------
    # LEFT side of source
    #
    # +z component:
    #     reflection of c_minus from left stack
    #
    # -z component:
    #     directly emitted c_minus
    # --------------------------------------------------------

    left_modal = np.concatenate(
        [
            R_left @ c_minus,
            c_minus,
        ]
    )


    psi_left = (
        R0
        @
        left_modal
    )


    # --------------------------------------------------------
    # RIGHT side of source
    #
    # +z component:
    #     directly emitted c_plus
    #
    # -z component:
    #     reflection of c_plus from right stack
    # --------------------------------------------------------

    right_modal = np.concatenate(
        [
            c_plus,
            R_right @ c_plus,
        ]
    )


    psi_right = (
        R0
        @
        right_modal
    )


    return psi_left, psi_right


# ============================================================
# 5. PROPAGATE THE DIPOLE WAVES TO THE EXTERIORS
# ============================================================

def dipole_exterior_amplitudes(
    Sl,
    Sr,
    c_plus,
    c_minus,
):
    """
    Calculate the dipole-generated modal amplitudes reaching
    the two exterior half-spaces.

    Parameters
    ----------
    Sl : ndarray
        Left partial scattering matrix.

    Sr : ndarray
        Right partial scattering matrix.

    c_plus : ndarray
        Source wave travelling toward +z.

    c_minus : ndarray
        Source wave travelling toward -z.

    Returns
    -------
    c_left : ndarray
        Outgoing amplitudes entering the incidence exterior.

    c_right : ndarray
        Outgoing amplitudes entering the transmission exterior.
    """

    c_plus = _as_complex_1d(
        c_plus,
        "c_plus",
    )

    c_minus = _as_complex_1d(
        c_minus,
        "c_minus",
    )


    _check_same_shape(
        c_plus,
        c_minus,
        "c_plus",
        "c_minus",
    )


    # --------------------------------------------------------
    # Standard scattering convention:
    #
    # [b_left ]   [S11 S12] [a_left ]
    # [b_right] = [S21 S22] [a_right]
    #
    # Source enters LEFT stack from its right port.
    # --------------------------------------------------------

    c_left = (
        Sl[0, 1]
        @
        c_minus
    )


    # Source enters RIGHT stack from its left port.

    c_right = (
        Sr[1, 0]
        @
        c_plus
    )


    return c_left, c_right


# ============================================================
# 6. EXTERIOR FAR-FIELD POWER FLUX
# ============================================================

def dipole_outgoing_flux(
    expansion,
    electric_field,
    kp,
    wavelength,
    epsilon=1.0,
    return_orders=False,
    grazing_tolerance=1e-12,
):
    """
    Calculate outward Khepri-normalized power flux carried by
    Fourier diffraction orders in a homogeneous, lossless
    exterior medium.

    This follows the same normalization convention as the
    numerator of Khepri's ``poynting_fluxes()`` but does NOT
    normalize by an incident plane-wave flux.

    Parameters
    ----------
    expansion : Expansion
        Khepri reciprocal-space expansion.

    electric_field : ndarray
        Transverse Fourier electric field

            [Ex_1 ... Ex_N,
             Ey_1 ... Ey_N]

    kp : tuple
        Bloch wavevector (qx, qy).

    wavelength : float
        Vacuum wavelength.

    epsilon : float
        Real relative permittivity of the homogeneous exterior.

    return_orders : bool
        If True, return the individual diffraction-order
        contributions as well.

    grazing_tolerance : float
        Threshold below which |kz| is treated as a grazing
        diffraction condition.

    Returns
    -------
    total_flux : float
        Total outward Khepri-normalized flux.

    order_flux : ndarray, optional
        Flux contribution from every Fourier order.

    Notes
    -----
    This is not an absolute power in watts.

    Use it consistently for comparisons or normalize against
    an identically implemented reference system.

    Complex/absorbing exterior media are intentionally not
    supported by this helper.
    """

    electric_field = _as_complex_1d(
        electric_field,
        "electric_field",
    )


    epsilon_complex = complex(
        epsilon
    )

    if abs(
        epsilon_complex.imag
    ) > 1e-14:
        raise NotImplementedError(
            "dipole_outgoing_flux currently supports only "
            "lossless exterior media with real epsilon."
        )


    if len(
        electric_field
    ) % 2 != 0:
        raise ValueError(
            "electric_field must contain equal Ex and Ey blocks."
        )


    Ex, Ey = np.split(
        electric_field,
        2,
    )


    # --------------------------------------------------------
    # Khepri returns normalized wavevectors:
    #
    #     kx/k0, ky/k0, kz/k0
    # --------------------------------------------------------

    kx, ky, kz = expansion.k_vectors(
        kp,
        wavelength,
        float(
            epsilon_complex.real
        ),
    )


    if Ex.shape != kx.shape:
        raise ValueError(
            "Electric-field harmonic count does not match "
            "the reciprocal-space expansion."
        )


    # --------------------------------------------------------
    # Detect exact/near grazing diffraction orders.
    #
    # Ez = -(kx Ex + ky Ey)/kz becomes singular exactly at
    # a Rayleigh threshold, so silently evaluating there is
    # dangerous.
    # --------------------------------------------------------

    grazing = (
        np.abs(kz)
        <
        grazing_tolerance
    )


    if np.any(
        grazing
    ):
        indices = np.where(
            grazing
        )[0]

        raise ValueError(
            "A grazing diffraction order was encountered at "
            f"indices {indices.tolist()}. "
            "Shift wavelength or kp slightly away from the "
            "Rayleigh threshold."
        )


    # --------------------------------------------------------
    # Longitudinal E component from k . E = 0
    # --------------------------------------------------------

    Ez = -(
        kx * Ex
        +
        ky * Ey
    ) / kz


    # --------------------------------------------------------
    # Khepri-normalized outward flux per order.
    #
    # Lossless evanescent orders have Re(kz)=0 and therefore
    # contribute zero exterior power.
    # --------------------------------------------------------

    order_flux = (
        np.real(kz)
        *
        (
            np.abs(Ex) ** 2
            +
            np.abs(Ey) ** 2
            +
            np.abs(Ez) ** 2
        )
    )


    # Remove tiny numerical negative values.

    order_flux = np.real_if_close(
        order_flux
    ).real


    tiny_negative = (
        (order_flux < 0)
        &
        (
            np.abs(order_flux)
            <
            1e-13
        )
    )

    order_flux[
        tiny_negative
    ] = 0.0


    total_flux = float(
        np.sum(
            order_flux
        ).real
    )


    if return_orders:
        return (
            total_flux,
            order_flux,
        )


    return total_flux


# ============================================================
# 7. NEAR-FIELD POYNTING FLUX AT THE SOURCE PLANE
# ============================================================

def dipole_nearfield_flux(
    psi,
):
    """
    Calculate the z-directed Khepri-normalized Poynting flux
    from the transverse Fourier field state.

    Khepri state ordering:

        psi = [Ex, Ey, Ux, Uy]

    with

        U = -i H.

    Since

        H = i U,

    the usual Poynting expression

        Re(Ex Hy* - Ey Hx*)

    becomes

        Im(Ex Uy* - Ey Ux*).

    Parameters
    ----------
    psi : ndarray
        Transverse Fourier field state

            [Ex, Ey, Ux, Uy].

    Returns
    -------
    Sz : float
        Net +z-directed Khepri-normalized flux summed over all
        retained Fourier harmonics.

    Notes
    -----
    Positive Sz means net power flow toward +z.

    Negative Sz means net power flow toward -z.

    This is a normalized solver quantity, not watts.
    """

    psi = _as_complex_1d(
        psi,
        "psi",
    )


    if len(
        psi
    ) % 4 != 0:
        raise ValueError(
            "psi length must be divisible by four."
        )


    N = (
        len(psi)
        //
        4
    )


    Ex = psi[
        0:N
    ]

    Ey = psi[
        N:2 * N
    ]

    Ux = psi[
        2 * N:3 * N
    ]

    Uy = psi[
        3 * N:4 * N
    ]


    Sz = np.imag(
        np.sum(
            Ex * np.conj(Uy)
            -
            Ey * np.conj(Ux)
        )
    )


    return float(
        np.real(
            Sz
        )
    )


# ============================================================
# 8. TOTAL POWER LEAVING THE DIPOLE PLANE
# ============================================================

def dipole_total_nearfield_power(
    psi_left,
    psi_right,
):
    """
    Calculate the total Khepri-normalized power leaving the
    dipole plane.

    Coordinate convention:

        +z -> right / emergence
        -z -> left  / incidence

    Therefore:

        P_left  = -Sz_left
        P_right = +Sz_right

    Parameters
    ----------
    psi_left : ndarray
        Field state immediately to the left of the source.

    psi_right : ndarray
        Field state immediately to the right of the source.

    Returns
    -------
    P_left : float
        Outward power emitted toward -z.

    P_right : float
        Outward power emitted toward +z.

    P_total : float
        Sum of the two outward fluxes.
    """

    Sz_left = dipole_nearfield_flux(
        psi_left
    )

    Sz_right = dipole_nearfield_flux(
        psi_right
    )


    P_left = float(
        -Sz_left
    )

    P_right = float(
        Sz_right
    )

    P_total = (
        P_left
        +
        P_right
    )


    return (
        P_left,
        P_right,
        P_total,
    )

# ============================================================
# 9. SINGLE-q DIPOLE POWER SOLVER
# ============================================================

def dipole_power_q(
    crystal,
    source_after,
    wavelength,
    kp,
    position,
    current,
    backend="dense",
    return_details=False,
):
    """
    Calculate dipole-emitted power for one Bloch wavevector q.

    This is the high-level interface connecting the validated
    dipole source implementation to a Khepri Crystal.

    Parameters
    ----------
    crystal : khepri.crystal.Crystal
        A fully constructed Khepri Crystal.

        The device must already have been created with
        set_device(...), and the partial scattering matrices
        must be retained at the dipole plane.

    source_after : str
        Name of the layer immediately BEFORE the dipole plane.

        Example:

            air
             |
          left_half
        ----------------  <- dipole plane
          right_half
             |
            air

        then:

            source_after = "left_half"

    wavelength : float
        Vacuum wavelength.

    kp : tuple
        Bloch wavevector

            (qx, qy)

    position : tuple
        Dipole position

            (xd, yd, zd)

        xd and yd determine the Fourier phase.

        zd is currently informational: the actual source
        z-plane is determined by ``source_after``.

    current : tuple
        Electric-current amplitudes

            (jx, jy, jz)

        Only x/y-oriented dipoles are currently supported.

    backend : str
        Khepri solver backend.

        Currently this dipole implementation is intended for

            backend="dense"

    return_details : bool
        If False, return only the three emitted powers.

        If True, also return fields, modal amplitudes and
        diagnostic quantities useful for debugging.

    Returns
    -------
    result : dict

        Always contains:

            result["P_left"]
            result["P_right"]
            result["P_total"]

        where these are Khepri-normalized near-source power
        fluxes.

        If return_details=True, additional internal quantities
        are included.

    Notes
    -----
    P_total is the quantity we will eventually integrate over
    the first Brillouin zone.

    This is NOT an absolute power in watts.

    z-oriented electric dipoles are not implemented yet.
    """

    # --------------------------------------------------------
    # Validate source orientation
    # --------------------------------------------------------

    if len(current) != 3:
        raise ValueError(
            "current must contain exactly (jx, jy, jz)."
        )

    if abs(current[2]) > 1e-14:
        raise NotImplementedError(
            "z-oriented electric dipoles are not implemented "
            "yet. Use current=(jx, jy, 0)."
        )


    # --------------------------------------------------------
    # Validate kp
    # --------------------------------------------------------

    if len(kp) != 2:
        raise ValueError(
            "kp must contain exactly (qx, qy)."
        )


    # --------------------------------------------------------
    # Set wavelength and Bloch vector
    #
    # te/tm are only dummy values required by Crystal's
    # ordinary source interface. They do NOT represent the
    # dipole excitation.
    # --------------------------------------------------------

    crystal.set_source(
        wavelength=wavelength,
        te=1.0,
        tm=0.0,
        kp=kp,
    )


    # --------------------------------------------------------
    # Solve ordinary passive RCWA structure
    # --------------------------------------------------------

    crystal.solve(
        backend=backend
    )


    # --------------------------------------------------------
    # Find source plane
    # --------------------------------------------------------

    if source_after not in crystal.global_stacking:
        raise ValueError(
            f"Layer '{source_after}' was not found in "
            "crystal.global_stacking."
        )


    source_index = crystal.global_stacking.index(
        source_after
    )


    # --------------------------------------------------------
    # Retrieve partial scattering matrices
    # --------------------------------------------------------

    Sl = crystal.stacking_matrices[
        source_index
    ]

    Sr = crystal.stacking_reverse_matrices[
        source_index
    ]


    if Sl is None:
        raise RuntimeError(
            "The left partial scattering matrix was not "
            "retained at the dipole plane. "
            "Check fields_mask in set_device()."
        )


    if Sr is None:
        raise RuntimeError(
            "The right partial scattering matrix was not "
            "retained at the dipole plane. "
            "Check fields_mask in set_device()."
        )


    # --------------------------------------------------------
    # Build Khepri reference basis
    # --------------------------------------------------------

    Kx, Ky, _ = crystal.expansion.k_vectors(
        kp,
        wavelength,
    )


    W0, V0 = free_space_eigenmodes(
        Kx,
        Ky,
    )


    R0 = layer_eigenbasis_matrix(
        W0,
        V0,
    )


    # --------------------------------------------------------
    # Fourier representation of dipole
    # --------------------------------------------------------

    Jx, Jy, Jz = dipole_fourier_current(
        expansion=crystal.expansion,
        kp=kp,
        position=position,
        current=current,
    )


    # --------------------------------------------------------
    # Build Maxwell source jump
    # --------------------------------------------------------

    b = dipole_source_jump_xy(
        Jx,
        Jy,
    )


    # --------------------------------------------------------
    # Solve self-consistent source-plane radiation
    # --------------------------------------------------------

    c_plus, c_minus = solve_dipole_source_plane(
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        b=b,
    )


    # --------------------------------------------------------
    # Reconstruct fields immediately around dipole
    # --------------------------------------------------------

    psi_left, psi_right = dipole_source_plane_fields(
        R0=R0,
        Sl=Sl,
        Sr=Sr,
        c_plus=c_plus,
        c_minus=c_minus,
    )


    # --------------------------------------------------------
    # Near-source emitted power
    #
    # This is the quantity required for total Purcell-factor
    # integration.
    # --------------------------------------------------------

    (
        P_left,
        P_right,
        P_total,
    ) = dipole_total_nearfield_power(
        psi_left,
        psi_right,
    )


    # --------------------------------------------------------
    # Basic source-plane consistency check
    # --------------------------------------------------------

    jump_difference = (
        psi_right
        -
        psi_left
        -
        b
    )


    source_norm = np.linalg.norm(
        b
    )


    if source_norm > 0:
        jump_relative_error = (
            np.linalg.norm(
                jump_difference
            )
            /
            source_norm
        )

    else:
        jump_relative_error = 0.0


    if jump_relative_error > 1e-8:
        raise RuntimeError(
            "Dipole source-plane Maxwell jump failed. "
            f"Relative error = {jump_relative_error:.3e}"
        )


    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------

    result = {
        "P_left": float(P_left),
        "P_right": float(P_right),
        "P_total": float(P_total),
    }


    # --------------------------------------------------------
    # Optional diagnostics
    # --------------------------------------------------------

    if return_details:

        result.update(
            {
                "kp": tuple(kp),
                "wavelength": wavelength,

                "source_index": source_index,

                "Jx": Jx,
                "Jy": Jy,
                "Jz": Jz,

                "b": b,

                "Sl": Sl,
                "Sr": Sr,

                "R0": R0,

                "c_plus": c_plus,
                "c_minus": c_minus,

                "psi_left": psi_left,
                "psi_right": psi_right,

                "jump_relative_error":
                    float(jump_relative_error),
            }
        )


    return result



def dipole_power_map(
    crystal,
    source_after,
    wavelength,
    qx_values,
    qy_values,
    position,
    current,
    backend="dense",
    verbose=False,
):
    """
    Calculate near-source dipole power over a 2D Bloch-vector grid.

    Parameters
    ----------
    crystal : Crystal
        Configured Khepri crystal.

    source_after : str
        Layer immediately before the dipole plane.

    wavelength : float
        Vacuum wavelength.

    qx_values, qy_values : array_like
        Bloch-vector coordinates.

    position : tuple
        Dipole position (xd, yd, zd).

    current : tuple
        Dipole current (jx, jy, jz).

        Currently only in-plane dipoles are supported.

    backend : str
        Khepri backend. Intended for "dense".

    verbose : bool
        Print progress if True.

    Returns
    -------
    result : dict
        Contains

            qx
            qy
            P_left
            P_right
            P_total

        P arrays have shape

            (len(qy_values), len(qx_values))
    """

    qx_values = np.asarray(
        qx_values,
        dtype=float,
    )

    qy_values = np.asarray(
        qy_values,
        dtype=float,
    )

    if qx_values.ndim != 1:
        raise ValueError(
            "qx_values must be one-dimensional."
        )

    if qy_values.ndim != 1:
        raise ValueError(
            "qy_values must be one-dimensional."
        )

    shape = (
        len(qy_values),
        len(qx_values),
    )

    P_left = np.zeros(
        shape,
        dtype=float,
    )

    P_right = np.zeros(
        shape,
        dtype=float,
    )

    P_total = np.zeros(
        shape,
        dtype=float,
    )

    total_points = (
        len(qx_values)
        *
        len(qy_values)
    )

    counter = 0

    for iy, qy in enumerate(
        qy_values
    ):

        for ix, qx in enumerate(
            qx_values
        ):

            counter += 1

            if verbose:
                print(
                    f"[{counter}/{total_points}] "
                    f"qx={qx:.6f}, qy={qy:.6f}"
                )

            result_q = dipole_power_q(
                crystal=crystal,
                source_after=source_after,
                wavelength=wavelength,
                kp=(
                    qx,
                    qy,
                ),
                position=position,
                current=current,
                backend=backend,
                return_details=False,
            )

            P_left[
                iy,
                ix
            ] = result_q[
                "P_left"
            ]

            P_right[
                iy,
                ix
            ] = result_q[
                "P_right"
            ]

            P_total[
                iy,
                ix
            ] = result_q[
                "P_total"
            ]

    return {
        "qx": qx_values,
        "qy": qy_values,
        "P_left": P_left,
        "P_right": P_right,
        "P_total": P_total,
    }



# ============================================================
# 11. HOMOGENEOUS REFERENCE POWER FOR IN-PLANE DIPOLE
# ============================================================

def homogeneous_reference_power_xy(
    wavelength,
    refractive_index,
    current,
):
    """
    Exact homogeneous-medium reference power for an in-plane
    electric dipole in the normalization used by the Khepri
    dipole implementation.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength.

    refractive_index : float
        Refractive index of the homogeneous reference medium.

    current : tuple
        Electric-current amplitude

            (jx, jy, jz)

        Only jz = 0 is currently supported.

    Returns
    -------
    P_ref : float
        Exact homogeneous reference power.

    Notes
    -----
    For an in-plane dipole,

        P_ref =
            k0^2 n / (6 pi)
            * (|Jx|^2 + |Jy|^2)

    where

        k0 = 2 pi / wavelength.

    This is in the same normalized convention used by the
    current Khepri dipole implementation.

    It is not an absolute power in watts.
    """

    if len(current) != 3:
        raise ValueError(
            "current must contain exactly (jx, jy, jz)."
        )

    jx, jy, jz = current

    if abs(jz) > 1e-14:
        raise NotImplementedError(
            "The homogeneous reference currently supports "
            "only in-plane x/y dipoles."
        )

    if wavelength <= 0:
        raise ValueError(
            "wavelength must be positive."
        )

    if refractive_index <= 0:
        raise ValueError(
            "refractive_index must be positive."
        )

    k0 = (
        2.0
        *
        np.pi
        /
        wavelength
    )

    current_squared = (
        abs(jx)**2
        +
        abs(jy)**2
    )

    if current_squared == 0:
        raise ValueError(
            "The in-plane dipole current cannot be zero."
        )

    P_ref = (
        k0**2
        *
        refractive_index
        /
        (
            6.0
            *
            np.pi
        )
        *
        current_squared
    )

    return float(
        np.real(
            P_ref
        )
    )


# ============================================================
# 12. INTEGRATE A q-SPACE POWER MAP
# ============================================================

def integrate_dipole_power_map(
    power_map,
    qx_values,
    qy_values,
):
    """
    Integrate a dipole-power map over a rectangular Bloch
    wavevector grid using midpoint quadrature.

    Calculates

        integral P(q) dqx dqy / (2 pi)^2.

    Parameters
    ----------
    power_map : ndarray
        q-resolved power with shape

            (len(qy_values), len(qx_values)).

    qx_values : ndarray
        Uniform midpoint-sampled qx coordinates.

    qy_values : ndarray
        Uniform midpoint-sampled qy coordinates.

    Returns
    -------
    integrated_power : float
        Brillouin-zone integrated normalized power.

    Notes
    -----
    The q-grid must be uniformly spaced.

    For high-Q photonic resonances, convergence with q-grid
    density must always be checked.
    """

    power_map = np.asarray(
        power_map,
        dtype=float,
    )

    qx_values = np.asarray(
        qx_values,
        dtype=float,
    )

    qy_values = np.asarray(
        qy_values,
        dtype=float,
    )


    if qx_values.ndim != 1:
        raise ValueError(
            "qx_values must be one-dimensional."
        )

    if qy_values.ndim != 1:
        raise ValueError(
            "qy_values must be one-dimensional."
        )


    expected_shape = (
        len(qy_values),
        len(qx_values),
    )


    if power_map.shape != expected_shape:
        raise ValueError(
            "power_map shape does not match q-grid. "
            f"Expected {expected_shape}, "
            f"got {power_map.shape}."
        )


    if len(qx_values) < 2:
        raise ValueError(
            "At least two qx points are required."
        )

    if len(qy_values) < 2:
        raise ValueError(
            "At least two qy points are required."
        )


    dqx_values = np.diff(
        qx_values
    )

    dqy_values = np.diff(
        qy_values
    )


    dqx = dqx_values[0]

    dqy = dqy_values[0]


    if not np.allclose(
        dqx_values,
        dqx,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "qx_values must be uniformly spaced."
        )


    if not np.allclose(
        dqy_values,
        dqy,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "qy_values must be uniformly spaced."
        )


    integrated_power = (
        np.sum(
            power_map
        )
        *
        dqx
        *
        dqy
        /
        (
            2.0
            *
            np.pi
        )**2
    )


    return float(
        integrated_power
    )


# ============================================================
# 13. PURCELL FACTOR FROM A PRECOMPUTED q MAP
# ============================================================

def purcell_factor_from_map(
    power_map,
    qx_values,
    qy_values,
    wavelength,
    reference_index,
    current,
):
    """
    Calculate the total Purcell factor from a q-resolved
    near-source power map.

    Parameters
    ----------
    power_map : ndarray
        Total near-source dipole power P(qx,qy).

    qx_values, qy_values : ndarray
        Uniform first-Brillouin-zone q grid.

    wavelength : float
        Vacuum wavelength.

    reference_index : float
        Refractive index of the homogeneous material relative
        to which the Purcell factor is defined.

    current : tuple
        Dipole current (jx, jy, jz).

        Currently only x/y dipoles are supported.

    Returns
    -------
    result : dict

        result["P_structure"]
            BZ-integrated structured-medium power.

        result["P_reference"]
            Exact homogeneous-medium reference power.

        result["F_purcell"]
            Purcell factor.
    """

    P_structure = integrate_dipole_power_map(
        power_map=power_map,
        qx_values=qx_values,
        qy_values=qy_values,
    )


    P_reference = homogeneous_reference_power_xy(
        wavelength=wavelength,
        refractive_index=reference_index,
        current=current,
    )


    F_purcell = (
        P_structure
        /
        P_reference
    )


    return {
        "P_structure": float(
            P_structure
        ),

        "P_reference": float(
            P_reference
        ),

        "F_purcell": float(
            F_purcell
        ),
    }

# ============================================================
# 14. RECIPROCAL LATTICE FROM REAL-SPACE UNIT CELL
# ============================================================

def reciprocal_vectors_from_unitcell(
    a1,
    a2,
):
    """
    Construct reciprocal primitive vectors from a two-dimensional
    real-space primitive cell.

    Parameters
    ----------
    a1, a2 : array_like, shape (2,)
        Real-space primitive lattice vectors. Their units must be
        consistent with the geometry and wavelength used by Khepri.

    Returns
    -------
    b1, b2 : ndarray, shape (2,)
        Reciprocal primitive vectors satisfying

            a_i . b_j = 2*pi*delta_ij.

    real_cell_area : float
        Area |a1 x a2| of the real-space primitive cell.

    reciprocal_cell_area : float
        Area |b1 x b2| of the reciprocal primitive cell.

    Notes
    -----
    The supplied a1 and a2 must describe the SAME Bravais lattice,
    orientation, and length scale used to construct ``crystal``.
    """

    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    if a1.shape != (2,):
        raise ValueError("a1 must be a two-component vector.")

    if a2.shape != (2,):
        raise ValueError("a2 must be a two-component vector.")

    if not np.all(np.isfinite(a1)) or not np.all(np.isfinite(a2)):
        raise ValueError("a1 and a2 must contain only finite values.")

    det = (
        a1[0] * a2[1]
        -
        a1[1] * a2[0]
    )

    scale = max(
        np.linalg.norm(a1) * np.linalg.norm(a2),
        1.0,
    )

    if abs(det) <= 1e-14 * scale:
        raise ValueError(
            "a1 and a2 are linearly dependent or nearly degenerate."
        )

    real_cell_area = abs(det)

    b1 = (
        2.0
        *
        np.pi
        /
        det
        *
        np.array(
            [
                a2[1],
                -a2[0],
            ],
            dtype=float,
        )
    )

    b2 = (
        2.0
        *
        np.pi
        /
        det
        *
        np.array(
            [
                -a1[1],
                a1[0],
            ],
            dtype=float,
        )
    )

    reciprocal_cell_area = abs(
        b1[0] * b2[1]
        -
        b1[1] * b2[0]
    )

    duality = np.array(
        [
            [
                np.dot(a1, b1),
                np.dot(a1, b2),
            ],
            [
                np.dot(a2, b1),
                np.dot(a2, b2),
            ],
        ],
        dtype=float,
    )

    if not np.allclose(
        duality,
        2.0 * np.pi * np.eye(2),
        rtol=1e-11,
        atol=1e-11,
    ):
        raise RuntimeError(
            "Reciprocal-lattice construction failed the "
            "a_i . b_j = 2*pi*delta_ij check."
        )

    expected_reciprocal_area = (
        (2.0 * np.pi) ** 2
        /
        real_cell_area
    )

    if not np.isclose(
        reciprocal_cell_area,
        expected_reciprocal_area,
        rtol=1e-11,
        atol=1e-13,
    ):
        raise RuntimeError(
            "Reciprocal-cell area failed the "
            "(2*pi)^2 / A_real consistency check."
        )

    return (
        b1,
        b2,
        float(real_cell_area),
        float(reciprocal_cell_area),
    )


# ============================================================
# 15. GENERAL RECIPROCAL PRIMITIVE-CELL GRID
# ============================================================

def reciprocal_unitcell_grid(
    a1,
    a2,
    Nu,
    Nv=None,
    shift=(0.0, 0.0),
):
    """
    Build a midpoint grid over one reciprocal primitive cell.

    Bloch vectors are parameterized as

        q = u*b1 + v*b2,

    with

        -1/2 <= u < 1/2,
        -1/2 <= v < 1/2.

    The resulting parallelogram is a reciprocal primitive cell.
    For any reciprocal-lattice-periodic integrand, integrating over
    this cell is equivalent to integrating over the Wigner-Seitz
    first Brillouin zone.

    Parameters
    ----------
    a1, a2 : array_like, shape (2,)
        Real-space primitive vectors.

    Nu, Nv : int
        Numbers of midpoint samples along b1 and b2. If Nv is
        omitted, Nv = Nu.

    shift : tuple(float, float)
        Shift in units of ONE grid cell. For example,

            shift=(0.23, 0.17)

        displaces the u and v midpoint grids by 0.23 and 0.17 of a
        grid spacing, respectively, before periodic wrapping.

    Returns
    -------
    result : dict
        Contains u, v, U, V, qx, qy, b1, b2, real_cell_area,
        reciprocal_cell_area, dA_q, and integration_weight.
    """

    if Nv is None:
        Nv = Nu

    try:
        Nu_int = int(Nu)
        Nv_int = int(Nv)
    except (TypeError, ValueError):
        raise ValueError("Nu and Nv must be positive integers.")

    if Nu_int != Nu or Nv_int != Nv:
        raise ValueError("Nu and Nv must be integers.")

    Nu = Nu_int
    Nv = Nv_int

    if Nu < 1 or Nv < 1:
        raise ValueError("Nu and Nv must both be >= 1.")

    if len(shift) != 2:
        raise ValueError("shift must contain exactly (shift_u, shift_v).")

    shift_u = float(shift[0])
    shift_v = float(shift[1])

    if not np.isfinite(shift_u) or not np.isfinite(shift_v):
        raise ValueError("shift components must be finite.")

    (
        b1,
        b2,
        real_cell_area,
        reciprocal_cell_area,
    ) = reciprocal_vectors_from_unitcell(
        a1=a1,
        a2=a2,
    )

    # Midpoint coordinates, periodically wrapped to [-1/2, +1/2).
    #
    # IMPORTANT:
    #   shift_u and shift_v are measured in FRACTIONS OF ONE GRID
    #   CELL.  Therefore the unshifted midpoint grid is
    #
    #       u_i = -1/2 + (i + 1/2) / Nu
    #
    #   and similarly for v.  Writing the periodic form as
    #
    #       ((i + 1/2 + shift_u) / Nu) % 1 - 1/2
    #
    #   preserves that midpoint grid exactly when shift=(0, 0).
    #
    # The previous implementation added +1/2 BEFORE the modulo,
    # which shifted the whole grid by half a reciprocal cell and
    # incorrectly put one point on u=-1/2 for odd Nu.
    u = (
        (
            (
                np.arange(Nu, dtype=float)
                +
                0.5
                +
                shift_u
            )
            /
            Nu
        )
        %
        1.0
        -
        0.5
    )

    v = (
        (
            (
                np.arange(Nv, dtype=float)
                +
                0.5
                +
                shift_v
            )
            /
            Nv
        )
        %
        1.0
        -
        0.5
    )

    # Sorting only changes presentation/order, not quadrature weight.
    u = np.sort(u)
    v = np.sort(v)

    U, V = np.meshgrid(
        u,
        v,
        indexing="xy",
    )

    qx = (
        U * b1[0]
        +
        V * b2[0]
    )

    qy = (
        U * b1[1]
        +
        V * b2[1]
    )

    dA_q = (
        reciprocal_cell_area
        /
        (Nu * Nv)
    )

    integration_weight = (
        dA_q
        /
        (2.0 * np.pi) ** 2
    )

    # Equivalent identity:
    #   integration_weight = 1 / (A_real * Nu * Nv)
    expected_weight = (
        1.0
        /
        (
            real_cell_area
            *
            Nu
            *
            Nv
        )
    )

    if not np.isclose(
        integration_weight,
        expected_weight,
        rtol=1e-11,
        atol=1e-15,
    ):
        raise RuntimeError(
            "Reciprocal-grid integration weight failed the "
            "1/(A_real*Nu*Nv) consistency check."
        )

    return {
        "u": u,
        "v": v,
        "U": U,
        "V": V,
        "qx": qx,
        "qy": qy,
        "b1": b1,
        "b2": b2,
        "real_cell_area": float(real_cell_area),
        "reciprocal_cell_area": float(reciprocal_cell_area),
        "dA_q": float(dA_q),
        "integration_weight": float(integration_weight),
        "Nu": Nu,
        "Nv": Nv,
        "shift": (shift_u, shift_v),
    }


# ============================================================
# 16. GENERAL DIPOLE POWER MAP ON A RECIPROCAL PRIMITIVE CELL
# ============================================================

def dipole_power_map_unitcell(
    crystal,
    source_after,
    wavelength,
    a1,
    a2,
    Nu,
    Nv=None,
    position=(0.0, 0.0, 0.0),
    current=(1.0, 0.0, 0.0),
    shift=(0.0, 0.0),
    backend="dense",
    verbose=False,
    return_point_details=False,
):
    """
    Calculate near-source dipole power over one complete reciprocal
    primitive cell of an arbitrary 2D Bravais lattice.

    Unlike ``dipole_power_map()``, this function does not assume
    separate Cartesian qx and qy axes. Instead,

        q(u,v) = u*b1 + v*b2,

    where b1 and b2 are constructed from the supplied real-space
    primitive vectors a1 and a2.

    Parameters
    ----------
    crystal : khepri.crystal.Crystal
        Fully configured Khepri crystal.

    source_after : str
        Layer immediately before the dipole source plane.

    wavelength : float
        Vacuum wavelength.

    a1, a2 : array_like, shape (2,)
        Real-space primitive vectors of the SAME lattice used by
        ``crystal``.

    Nu, Nv : int
        Reciprocal-space sample counts along b1 and b2.

    position : tuple
        Dipole position (xd, yd, zd).

    current : tuple
        Dipole current (jx, jy, jz). The present source jump supports
        only jz = 0.

    shift : tuple
        Grid shift in fractions of one u/v grid spacing.

    backend : str
        Khepri solver backend. The current implementation is intended
        for ``backend="dense"``.

    verbose : bool
        Print progress when True.

    return_point_details : bool
        If True, retain the complete ``dipole_power_q`` result at every
        q point. This can consume substantial memory.

    Returns
    -------
    result : dict
        P_left, P_right, P_total have shape (Nv, Nu). qx and qy are
        2D Cartesian-coordinate arrays with the same shape. The result
        also contains fractional coordinates and the exact quadrature
        weight for integration of d^2q/(2*pi)^2.
    """

    grid = reciprocal_unitcell_grid(
        a1=a1,
        a2=a2,
        Nu=Nu,
        Nv=Nv,
        shift=shift,
    )

    qx = grid["qx"]
    qy = grid["qy"]

    Nv_eff, Nu_eff = qx.shape

    P_left = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )

    P_right = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )

    P_total = np.empty(
        (Nv_eff, Nu_eff),
        dtype=float,
    )

    if return_point_details:
        point_details = np.empty(
            (Nv_eff, Nu_eff),
            dtype=object,
        )
    else:
        point_details = None

    total_points = Nu_eff * Nv_eff
    counter = 0

    for iv in range(Nv_eff):

        for iu in range(Nu_eff):

            counter += 1

            qx_ij = float(qx[iv, iu])
            qy_ij = float(qy[iv, iu])

            if verbose:
                print(
                    f"[{counter}/{total_points}] "
                    f"u={grid['U'][iv, iu]:+.8f}, "
                    f"v={grid['V'][iv, iu]:+.8f}, "
                    f"qx={qx_ij:+.8e}, "
                    f"qy={qy_ij:+.8e}"
                )

            try:
                point_result = dipole_power_q(
                    crystal=crystal,
                    source_after=source_after,
                    wavelength=wavelength,
                    kp=(qx_ij, qy_ij),
                    position=position,
                    current=current,
                    backend=backend,
                    return_details=return_point_details,
                )
            except Exception as exc:
                raise RuntimeError(
                    "dipole_power_q failed at reciprocal-grid point "
                    f"(iv={iv}, iu={iu}), "
                    f"u={grid['U'][iv, iu]:+.12g}, "
                    f"v={grid['V'][iv, iu]:+.12g}, "
                    f"qx={qx_ij:+.12g}, qy={qy_ij:+.12g}."
                ) from exc

            p_left = float(np.real(point_result["P_left"]))
            p_right = float(np.real(point_result["P_right"]))
            p_total = float(np.real(point_result["P_total"]))

            if not np.all(
                np.isfinite(
                    [
                        p_left,
                        p_right,
                        p_total,
                    ]
                )
            ):
                raise RuntimeError(
                    "Non-finite dipole power encountered at "
                    f"(iv={iv}, iu={iu}), "
                    f"u={grid['U'][iv, iu]:+.12g}, "
                    f"v={grid['V'][iv, iu]:+.12g}, "
                    f"qx={qx_ij:+.12g}, qy={qy_ij:+.12g}. "
                    "The grid may be sampling an exact grazing/Rayleigh "
                    "condition or an ill-conditioned pole."
                )

            P_left[iv, iu] = p_left
            P_right[iv, iu] = p_right
            P_total[iv, iu] = p_total

            if return_point_details:
                point_details[iv, iu] = point_result

    result = dict(grid)

    result.update(
        {
            "P_left": P_left,
            "P_right": P_right,
            "P_total": P_total,
        }
    )

    if return_point_details:
        result["point_details"] = point_details

    return result


# ============================================================
# 17. INTEGRATE GENERAL RECIPROCAL-CELL POWER MAP
# ============================================================

def integrate_dipole_power_unitcell(
    map_result,
    component="P_total",
):
    """
    Integrate a map returned by ``dipole_power_map_unitcell``.

    Computes

        integral P(q) d^2q / (2*pi)^2

    over one complete reciprocal primitive cell.
    """

    if component not in (
        "P_total",
        "P_left",
        "P_right",
    ):
        raise ValueError(
            "component must be 'P_total', 'P_left', or 'P_right'."
        )

    if "integration_weight" not in map_result:
        raise ValueError(
            "map_result does not contain 'integration_weight'. "
            "Use dipole_power_map_unitcell() to generate the map."
        )

    P = np.asarray(
        map_result[component],
        dtype=float,
    )

    if P.ndim != 2:
        raise ValueError(
            f"{component} must be a two-dimensional power map."
        )

    if not np.all(np.isfinite(P)):
        raise ValueError(
            f"{component} contains NaN or infinite values."
        )

    if "qx" in map_result:
        qx = np.asarray(map_result["qx"])
        if qx.shape != P.shape:
            raise ValueError(
                f"{component} shape does not match qx shape."
            )

    if "qy" in map_result:
        qy = np.asarray(map_result["qy"])
        if qy.shape != P.shape:
            raise ValueError(
                f"{component} shape does not match qy shape."
            )

    weight = float(
        map_result["integration_weight"]
    )

    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(
            "integration_weight must be positive and finite."
        )

    return float(
        np.sum(P)
        *
        weight
    )


# ============================================================
# 18. PURCELL FACTOR FROM A GENERAL RECIPROCAL-CELL MAP
# ============================================================

def purcell_factor_from_unitcell_map(
    map_result,
    wavelength,
    reference_index,
    current,
    component="P_total",
):
    """
    Calculate a Purcell factor from a reciprocal-primitive-cell map.

    For the total Purcell factor, use component="P_total".
    """

    P_structure = integrate_dipole_power_unitcell(
        map_result=map_result,
        component=component,
    )

    P_reference = homogeneous_reference_power_xy(
        wavelength=wavelength,
        refractive_index=reference_index,
        current=current,
    )

    F_purcell = (
        P_structure
        /
        P_reference
    )

    return {
        "P_structure": float(P_structure),
        "P_reference": float(P_reference),
        "F_purcell": float(F_purcell),
    }


# ============================================================
# 19. SHIFT-AVERAGED GENERAL RECIPROCAL-CELL PURCELL FACTOR
# ============================================================

def purcell_factor_shift_averaged_unitcell(
    crystal,
    source_after,
    wavelength,
    a1,
    a2,
    Nu,
    Nv=None,
    position=(0.0, 0.0, 0.0),
    current=(1.0, 0.0, 0.0),
    reference_index=1.0,
    shifts=None,
    backend="dense",
    verbose=False,
):
    """
    Calculate the total Purcell factor using several shifted grids
    over a general reciprocal primitive cell.

    This is the general-lattice analogue of
    ``purcell_factor_shift_averaged``.

    Parameters
    ----------
    a1, a2 : array_like
        Real-space primitive vectors.

    Nu, Nv : int
        Reciprocal-grid sizes.

    shifts : sequence of (shift_u, shift_v), optional
        Shifts in fractions of one reciprocal-grid spacing.

        The defaults deliberately avoid simple half-cell shifts,
        which can accidentally land exactly on Rayleigh/grazing
        conditions for some wavelengths.

    Returns
    -------
    result : dict
        Contains mean structured power, reference power, mean Purcell
        factor, standard deviations, full spread, individual values,
        and the shifts used.
    """

    if Nv is None:
        Nv = Nu

    if shifts is None:
        shifts = [
            (0.11, 0.07),
            (0.29, 0.43),
            (0.53, 0.19),
            (0.73, 0.61),
        ]

    shifts = list(shifts)

    if len(shifts) < 1:
        raise ValueError(
            "At least one reciprocal-grid shift is required."
        )

    P_reference = homogeneous_reference_power_xy(
        wavelength=wavelength,
        refractive_index=reference_index,
        current=current,
    )

    individual_P = []
    individual_F = []

    for ishift, shift in enumerate(
        shifts,
        start=1,
    ):

        map_result = dipole_power_map_unitcell(
            crystal=crystal,
            source_after=source_after,
            wavelength=wavelength,
            a1=a1,
            a2=a2,
            Nu=Nu,
            Nv=Nv,
            position=position,
            current=current,
            shift=shift,
            backend=backend,
            verbose=False,
            return_point_details=False,
        )

        P_integrated = integrate_dipole_power_unitcell(
            map_result=map_result,
            component="P_total",
        )

        Fp = (
            P_integrated
            /
            P_reference
        )

        individual_P.append(
            float(P_integrated)
        )

        individual_F.append(
            float(Fp)
        )

        if verbose:
            print(
                f"Shift {ishift:2d}/{len(shifts):2d} "
                f"(su,sv)=({shift[0]:.4f},{shift[1]:.4f}) "
                f"P={P_integrated:.12g} "
                f"Fp={Fp:.12g}"
            )

    individual_P = np.asarray(
        individual_P,
        dtype=float,
    )

    individual_F = np.asarray(
        individual_F,
        dtype=float,
    )

    P_mean = float(
        np.mean(individual_P)
    )

    F_mean = float(
        np.mean(individual_F)
    )

    if len(individual_P) > 1:
        P_std = float(
            np.std(
                individual_P,
                ddof=1,
            )
        )

        F_std = float(
            np.std(
                individual_F,
                ddof=1,
            )
        )
    else:
        P_std = 0.0
        F_std = 0.0

    F_min = float(
        np.min(individual_F)
    )

    F_max = float(
        np.max(individual_F)
    )

    F_spread = float(
        F_max
        -
        F_min
    )

    if F_mean != 0.0:
        F_spread_percent = float(
            F_spread
            /
            abs(F_mean)
            *
            100.0
        )

        F_std_percent = float(
            F_std
            /
            abs(F_mean)
            *
            100.0
        )
    else:
        F_spread_percent = np.nan
        F_std_percent = np.nan

    return {
        "P_structure": P_mean,
        "P_structure_std": P_std,
        "P_reference": float(P_reference),
        "F_purcell": F_mean,
        "F_std": F_std,
        "F_std_percent": F_std_percent,
        "F_spread": F_spread,
        "F_spread_percent": F_spread_percent,
        "individual_P": individual_P,
        "individual_F": individual_F,
        "shifts": shifts,
        "Nu": int(Nu),
        "Nv": int(Nv),
    }


# ============================================================
# 20. BACKWARD-COMPATIBLE SQUARE-LATTICE SHIFT AVERAGING
# ============================================================

def purcell_factor_shift_averaged(
    crystal,
    source_after,
    wavelength,
    a,
    Nq,
    position,
    current,
    reference_index,
    shifts=None,
    backend="dense",
    verbose=False,
):
    """
    Backward-compatible square-lattice wrapper around
    ``purcell_factor_shift_averaged_unitcell``.

    The square real-space primitive vectors are

        a1 = (a, 0)
        a2 = (0, a).
    """

    a = float(a)

    if not np.isfinite(a) or a <= 0:
        raise ValueError(
            "a must be a positive finite lattice period."
        )

    return purcell_factor_shift_averaged_unitcell(
        crystal=crystal,
        source_after=source_after,
        wavelength=wavelength,
        a1=(a, 0.0),
        a2=(0.0, a),
        Nu=Nq,
        Nv=Nq,
        position=position,
        current=current,
        reference_index=reference_index,
        shifts=shifts,
        backend=backend,
        verbose=verbose,
    )
