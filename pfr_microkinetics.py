import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from thermo_equil import R_GAS, MOLECULAR_WEIGHTS

def parse_surface_mechanism(xml_path="2steamreform.xml"):
    """Parses species, site density, sticking reactions, Arrhenius steps, and coverage dependencies."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    surf_species = root.find("species").text.strip().split()
    site_el = root.find("site")
    site_density = float(site_el.find("density").text.strip())  # mol/cm2
    site_density_m2 = site_density * 1e4  # convert mol/cm2 -> mol/m2

    init_cov = {sp: 0.0 for sp in surf_species}
    for pair in site_el.find("initial").text.strip().split(","):
        k, v = pair.split("=")
        init_cov[k.strip()] = float(v.strip())

    cov_deps = {}
    for cov_el in root.findall("coverage"):
        rxn_ids = [int(x) for x in cov_el.attrib["id"].split()]
        sp_dep, val = cov_el.text.strip().split("=")
        for rid in rxn_ids:
            cov_deps.setdefault(rid, {})[sp_dep.strip()] = float(val.strip())

    reactions = []

    def parse_side(side_str):
        counts = {}
        for token in side_str.split("+"):
            tok = token.strip()
            if tok:
                counts[tok] = counts.get(tok, 0) + 1
        return counts

    for rxn in root.find("stick").findall("rxn"):
        rid = int(rxn.attrib["id"])
        eq_part, param_part = rxn.text.split("@")
        lhs, rhs = eq_part.split("=>")
        s0 = float(param_part.strip())
        reactions.append({
            "id": rid, "type": "stick",
            "reactants": parse_side(lhs), "products": parse_side(rhs),
            "s0": s0
        })

    for rxn in root.find("arrhenius").findall("rxn"):
        rid = int(rxn.attrib["id"])
        eq_part, param_part = rxn.text.split("@")
        lhs, rhs = eq_part.split("=>")
        A, beta, Ea = [float(x) for x in param_part.strip().split()]
        reactions.append({
            "id": rid, "type": "arrhenius",
            "reactants": parse_side(lhs), "products": parse_side(rhs),
            "A": A, "beta": beta, "Ea_kJ": Ea
        })

    return {
        "surf_species": surf_species,
        "site_density_m2": site_density_m2,
        "init_cov": init_cov,
        "reactions": reactions,
        "cov_deps": cov_deps
    }

def compute_rates(T, P_pa, gas_y, gas_species, theta_vec, mech):
    """Computes net surface reaction rates (mol / m2 / s) for gas and surface species."""
    surf_species = mech["surf_species"]
    sp_to_idx = {sp: i for i, sp in enumerate(surf_species)}
    theta = {sp: max(theta_vec[i], 0.0) for i, sp in enumerate(surf_species)}
    gamma = mech["site_density_m2"]  # mol/m2

    P_i = {sp: max(gas_y[i], 0.0) * P_pa for i, sp in enumerate(gas_species)}

    dtheta_dt = np.zeros(len(surf_species))
    gas_net_rate = {sp: 0.0 for sp in gas_species}

    for rxn in mech["reactions"]:
        rid = rxn["id"]
        if rxn["type"] == "stick":
            gas_sp = [s for s in rxn["reactants"] if s in P_i][0]
            MW = MOLECULAR_WEIGHTS[gas_sp]
            flux_coeff = rxn["s0"] / np.sqrt(2.0 * np.pi * MW * R_GAS * T)
            rate = flux_coeff * P_i[gas_sp]
            for sp, nu in rxn["reactants"].items():
                if sp in sp_to_idx:
                    rate *= (theta[sp] ** nu)
        else:
            Ea_eff_kJ = rxn["Ea_kJ"]
            if rid in mech["cov_deps"]:
                for dep_sp, coeff in mech["cov_deps"][rid].items():
                    Ea_eff_kJ += coeff * theta.get(dep_sp, 0.0)
            Ea_eff_J = Ea_eff_kJ * 1000.0
            k_f = rxn["A"] * (T ** rxn["beta"]) * np.exp(-Ea_eff_J / (R_GAS * T))
            # Turnover frequency k_f (1/s) * site density gamma (mol/m2) = mol/(m2*s)
            rate = k_f * gamma
            for sp, nu in rxn["reactants"].items():
                if sp in sp_to_idx:
                    rate *= (theta[sp] ** nu)

        for sp, nu in rxn["reactants"].items():
            if sp in sp_to_idx:
                dtheta_dt[sp_to_idx[sp]] -= nu * (rate / gamma)
            elif sp in gas_net_rate:
                gas_net_rate[sp] -= nu * rate

        for sp, nu in rxn["products"].items():
            if sp in sp_to_idx:
                dtheta_dt[sp_to_idx[sp]] += nu * (rate / gamma)
            elif sp in gas_net_rate:
                gas_net_rate[sp] += nu * rate

    return dtheta_dt, gas_net_rate

def solve_steady_coverage(T, P_pa, gas_y, gas_species, mech, theta_guess=None):
    """Integrates surface coverage ODEs to pseudo-steady state (dtheta/dt = 0)."""
    surf_species = mech["surf_species"]
    if theta_guess is None:
        y0 = np.array([mech["init_cov"][sp] for sp in surf_species], dtype=float)
    else:
        y0 = theta_guess.copy()

    def rhs(t, th):
        dth, _ = compute_rates(T, P_pa, gas_y, gas_species, th, mech)
        return dth

    sol = solve_ivp(rhs, [0.0, 500.0], y0, method="Radau", rtol=1e-9, atol=1e-12)
    th_ss = np.maximum(sol.y[:, -1], 0.0)
    th_ss /= np.sum(th_ss)
    return th_ss

def simulate_pfr(T, P_pa, gas_species, feed_mole_fracs, mech,
                 length=0.3, dia=0.005, velocity=0.02, cat_geom_factor=5000.0):
    """Solves 1D isothermal catalytic Plug Flow Reactor coupled with surface microkinetics."""
    A_cross = np.pi * (dia / 2.0) ** 2
    P_perim = np.pi * dia
    a_cat = (P_perim / A_cross) * cat_geom_factor

    C_tot = P_pa / (R_GAS * T)
    Q_vol = velocity * A_cross
    F0 = np.array([feed_mole_fracs.get(sp, 0.0) * C_tot * Q_vol for sp in gas_species])

    theta_cache = [solve_steady_coverage(T, P_pa, np.array([feed_mole_fracs.get(s, 0.0) for s in gas_species]), gas_species, mech)]

    def pfr_ode(z, F):
        F_pos = np.maximum(F, 0.0)
        F_tot = np.sum(F_pos)
        y_gas = F_pos / F_tot
        theta_ss = solve_steady_coverage(T, P_pa, y_gas, gas_species, mech, theta_cache[0])
        theta_cache[0] = theta_ss
        _, r_gas = compute_rates(T, P_pa, y_gas, gas_species, theta_ss, mech)
        dF_dz = np.array([a_cat * A_cross * r_gas[sp] for sp in gas_species])
        return dF_dz

    z_eval = np.linspace(0.0, length, 50)
    sol = solve_ivp(pfr_ode, [0.0, length], F0, t_eval=z_eval, method="BDF", rtol=1e-7, atol=1e-10)

    F_out = np.maximum(sol.y[:, -1], 0.0)
    y_out = F_out / np.sum(F_out)

    profile_rows = []
    for idx, z_val in enumerate(sol.t):
        F_z = np.maximum(sol.y[:, idx], 0.0)
        y_z = F_z / np.sum(F_z)
        row = {"z_m": z_val}
        row.update({sp: y_z[i] for i, sp in enumerate(gas_species)})
        profile_rows.append(row)

    return dict(zip(gas_species, y_out)), theta_cache[0], pd.DataFrame(profile_rows)
