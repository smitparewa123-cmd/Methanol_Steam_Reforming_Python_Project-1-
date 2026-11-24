import numpy as np
from scipy.optimize import minimize

R_GAS = 8.314462618  # J/(mol*K)

MOLECULAR_WEIGHTS = {
    "CH3OH": 0.03204,
    "H2O": 0.018015,
    "H2": 0.002016,
    "CO": 0.02801,
    "CO2": 0.04401,
    "CH4": 0.01604,
    "O2": 0.031998,
    "N2": 0.0280134
}

# Elemental composition matrix: [C, H, O, N]
ELEMENT_MATRIX = {
    "CH3OH": np.array([1, 4, 1, 0], dtype=float),
    "H2O":   np.array([0, 2, 1, 0], dtype=float),
    "H2":    np.array([0, 2, 0, 0], dtype=float),
    "CO":    np.array([1, 0, 1, 0], dtype=float),
    "CO2":   np.array([1, 0, 2, 0], dtype=float),
    "CH4":   np.array([1, 4, 0, 0], dtype=float),
    "O2":    np.array([0, 0, 2, 0], dtype=float),
    "N2":    np.array([0, 0, 0, 2], dtype=float),
}

def parse_nasa_therm(filepath="therm.dat"):
    """Parses 7-coefficient NASA polynomials from CHEMKIN therm.dat."""
    with open(filepath, "r", errors="replace") as f:
        lines = [l.rstrip("\r\n") for l in f.readlines()]
    
    thermo_db = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if len(line) >= 80 and line[79] == "1" and not line.startswith("!"):
            sp_name = line[:18].strip().split()[0].upper()
            try:
                t_low = float(line[45:55].strip())
                t_high = float(line[55:65].strip())
                t_mid = float(line[65:75].strip())
                l2, l3, l4 = lines[i+1], lines[i+2], lines[i+3]
                high_coeffs = [
                    float(l2[0:15]), float(l2[15:30]), float(l2[30:45]),
                    float(l2[45:60]), float(l2[60:75]), float(l3[0:15]), float(l3[15:30])
                ]
                low_coeffs = [
                    float(l3[30:45]), float(l3[45:60]), float(l3[60:75]),
                    float(l4[0:15]), float(l4[15:30]), float(l4[30:45]), float(l4[45:60])
                ]
                thermo_db[sp_name] = {
                    "T_low": t_low, "T_mid": t_mid, "T_high": t_high,
                    "high": np.array(high_coeffs), "low": np.array(low_coeffs)
                }
                i += 4
                continue
            except Exception:
                pass
        i += 1
    return thermo_db

def calc_standard_gibbs_RT(sp, T, thermo_db):
    """Calculates dimensionless standard Gibbs free energy G0(T) / (R*T)."""
    data = thermo_db[sp.upper()]
    a = data["low"] if T <= data["T_mid"] else data["high"]
    H_RT = a[0] + a[1]*T/2.0 + a[2]*(T**2)/3.0 + a[3]*(T**3)/4.0 + a[4]*(T**4)/5.0 + a[5]/T
    S_R  = a[0]*np.log(T) + a[1]*T + a[2]*(T**2)/2.0 + a[3]*(T**3)/3.0 + a[4]*(T**4)/4.0 + a[6]
    return H_RT - S_R

def solve_equilibrium(T, P_pa, gas_species, init_mole_fracs, thermo_db):
    """Minimizes total Gibbs free energy using reaction-extent parameterization."""
    P_bar = P_pa / 1e5
    n_sp = len(gas_species)
    n0 = np.array([init_mole_fracs.get(sp, 0.0) for sp in gas_species], dtype=float)
    A_elem = np.array([ELEMENT_MATRIX[sp] for sp in gas_species]).T
    b_elem = A_elem @ n0

    active_rows = np.where(b_elem > 1e-12)[0]
    A_eq = A_elem[active_rows]
    b_eq = b_elem[active_rows]

    g0_RT = np.array([calc_standard_gibbs_RT(sp, T, thermo_db) for sp in gas_species])

    # Interior feasible starting point: 50% converted to CO2 + 3 H2
    x0 = n0.copy()
    if "CH3OH" in gas_species and "H2O" in gas_species and "CO2" in gas_species and "H2" in gas_species:
        i_m, i_w = gas_species.index("CH3OH"), gas_species.index("H2O")
        i_c2, i_h2 = gas_species.index("CO2"), gas_species.index("H2")
        xi = 0.4 * min(x0[i_m], x0[i_w])
        x0[i_m] -= xi
        x0[i_w] -= xi
        x0[i_c2] += xi
        x0[i_h2] += 3.0 * xi
    x0 = np.maximum(x0, 1e-5)

    def objective(n):
        n_tot = np.sum(n)
        y = n / n_tot
        return np.sum(n * (g0_RT + np.log(y * P_bar)))

    def grad(n):
        n_tot = np.sum(n)
        y = n / n_tot
        return g0_RT + np.log(y * P_bar)

    bounds = [(1e-12, None) for _ in range(n_sp)]
    constraints = {"type": "eq", "fun": lambda n: A_eq @ n - b_eq, "jac": lambda n: A_eq}

    res = minimize(objective, x0, method="SLSQP", jac=grad, bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-14, "maxiter": 2000})
    n_eq = np.maximum(res.x, 0.0)
    y_eq = n_eq / np.sum(n_eq)
    return dict(zip(gas_species, y_eq))
