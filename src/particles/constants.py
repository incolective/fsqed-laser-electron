"""Physical constants (SI) and strong-field QED scaling relations."""

from __future__ import annotations

import math

# Exact / CODATA SI fundamental constants
C_LIGHT: float = 299_792_458.0  # m / s
E_CHARGE: float = 1.602176634e-19  # C
M_ELECTRON: float = 9.1093837015e-31  # kg
HBAR: float = 1.054571817e-34  # J * s
EPSILON_0: float = 8.8541878128e-12  # F / m
ALPHA_FS: float = 7.2973525693e-3  # Fine-structure constant ~ 1/137.036

# Derived QED & relativistic scales
MC_KGMS: float = M_ELECTRON * C_LIGHT  # Electron Compton momentum scale (kg * m / s)
MC2_J: float = M_ELECTRON * (C_LIGHT**2)  # Electron rest energy (J)
EV_J: float = E_CHARGE  # 1 eV in Joules
GEV_J: float = 1.0e9 * E_CHARGE  # 1 GeV in Joules (1.602176634e-10 J)
MC2_GEV: float = MC2_J / GEV_J  # ~ 0.00051099895 GeV

# Schwinger critical field E_cr = m^2 c^3 / (e * hbar) ~ 1.32328548e18 V/m
E_CRIT: float = (M_ELECTRON**2) * (C_LIGHT**3) / (E_CHARGE * HBAR)
B_CRIT: float = E_CRIT / C_LIGHT

# Intensity FWHM to Gaussian field-envelope sigma_t conversion: sqrt(4 * ln(2))
FWHM_TO_FIELD_SIGMA: float = math.sqrt(4.0 * math.log(2.0))  # ~ 1.6651092223153954

# Paper reference constants (Olofsson & Gonoskov, PRA 106, 063512, 2022)
A_DIPOLE_1PW_1UM: float = 780.0  # Bassett dipole limit a_d at P=1 PW, lambda=1 um
A_BIDIPOLE_1PW_1UM: float = A_DIPOLE_1PW_1UM / math.sqrt(2.0)  # ~ 551.543
KAPPA_BIDIPOLE_MAX: float = 5.25  # Eq. (1) optimal focusing constant


def omega_from_lambda(lambda_m: float) -> float:
    """Convert wavelength in meters to angular frequency omega_0 in rad/s."""
    if not math.isfinite(lambda_m) or lambda_m <= 0.0:
        raise ValueError(f"lambda_m must be finite and positive, got {lambda_m}")
    return 2.0 * math.pi * C_LIGHT / lambda_m


def period_from_lambda(lambda_m: float) -> float:
    """Return optical cycle period T_0 = lambda / c in seconds."""
    if not math.isfinite(lambda_m) or lambda_m <= 0.0:
        raise ValueError(f"lambda_m must be finite and positive, got {lambda_m}")
    return lambda_m / C_LIGHT


def energy_gev_to_gamma(energy_gev: float) -> float:
    """Convert electron total energy in GeV to Lorentz factor gamma."""
    if not math.isfinite(energy_gev) or energy_gev <= MC2_GEV:
        raise ValueError(
            f"energy_gev ({energy_gev}) must be finite and exceed electron rest energy ({MC2_GEV:.6e} GeV)"
        )
    return energy_gev / MC2_GEV


def relativistic_field_scale(omega0_rad_s: float) -> float:
    """Return relativistic electric field unit E_rel = m_e * c * omega_0 / e (V/m)."""
    if not math.isfinite(omega0_rad_s) or omega0_rad_s <= 0.0:
        raise ValueError(f"omega0_rad_s must be finite and positive, got {omega0_rad_s}")
    return M_ELECTRON * C_LIGHT * omega0_rad_s / E_CHARGE


def e_field_to_a0(e_field_vm: float, omega0_rad_s: float) -> float:
    """Convert electric field amplitude (V/m) to dimensionless a_0 = e E / (m c omega_0)."""
    if not math.isfinite(e_field_vm):
        raise ValueError(f"e_field_vm must be finite, got {e_field_vm}")
    return e_field_vm / relativistic_field_scale(omega0_rad_s)


def a0_to_e_field(a0: float, omega0_rad_s: float) -> float:
    """Convert dimensionless amplitude a_0 to electric field (V/m)."""
    if not math.isfinite(a0):
        raise ValueError(f"a0 must be finite, got {a0}")
    return a0 * relativistic_field_scale(omega0_rad_s)


def tau_fwhm_to_sigma_t(tau_fwhm_s: float) -> float:
    """Convert intensity FWHM duration (s) to Gaussian field-envelope sigma_t (s)."""
    if not math.isfinite(tau_fwhm_s) or tau_fwhm_s <= 0.0:
        raise ValueError(f"tau_fwhm_s must be finite and positive, got {tau_fwhm_s}")
    return tau_fwhm_s / FWHM_TO_FIELD_SIGMA


def theoretical_chi_max(
    energy_gev: float,
    power_w: float,
    lambda_m: float,
    kappa: float = KAPPA_BIDIPOLE_MAX,
) -> float:
    """Evaluate Eq. (1): chi_max = kappa * (eps / 1 GeV) * sqrt(P / 1 PW) * (1 um / lambda)."""
    if (
        not math.isfinite(energy_gev)
        or not math.isfinite(power_w)
        or not math.isfinite(lambda_m)
        or not math.isfinite(kappa)
        or energy_gev <= 0.0
        or power_w < 0.0
        or lambda_m <= 0.0
        or kappa < 0.0
    ):
        raise ValueError("Invalid parameters for theoretical_chi_max")
    return kappa * energy_gev * math.sqrt(power_w / 1.0e15) * (1.0e-6 / lambda_m)
