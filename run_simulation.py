import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thermo_equil import parse_nasa_therm, solve_equilibrium
from pfr_microkinetics import parse_surface_mechanism, simulate_pfr

def main():
    gas_species = ["CH3OH", "H2O", "H2", "CO", "CO2", "CH4", "O2", "N2"]
    # Stoichiometric Steam Reforming of Methanol (1:1 molar CH3OH:H2O)
    feed_mole_fracs = {
        "CH3OH": 0.50,
        "H2O":   0.50,
        "H2": 0.0, "CO": 0.0, "CO2": 0.0, "CH4": 0.0, "O2": 0.0, "N2": 0.0
    }

    P_pa = 1.0e5  # 1 bar
    # Temperature sweep covering ACS Catal. 2022, 12, 512-526 range (450 K to 650 K)
    temperatures_K = np.linspace(450.0, 650.0, 9)

    thermo_db = parse_nasa_therm("therm.dat")
    mech = parse_surface_mechanism("2steamreform.xml")

    pfr_results = []
    equil_results = []
    sample_axial_df = None

    print("Running Methanol Steam Reforming PFR vs. Equilibrium Simulation...")
    for T in temperatures_K:
        y_pfr, theta_out, axial_df = simulate_pfr(
            T=T, P_pa=P_pa, gas_species=gas_species,
            feed_mole_fracs=feed_mole_fracs, mech=mech,
            length=0.3, dia=0.005, velocity=0.02, cat_geom_factor=5000.0
        )
        y_eq = solve_equilibrium(T, P_pa, gas_species, feed_mole_fracs, thermo_db)

        pfr_row = {"T_K": T, "T_C": T - 273.15, **y_pfr}
        eq_row = {"T_K": T, "T_C": T - 273.15, **y_eq}
        pfr_results.append(pfr_row)
        equil_results.append(eq_row)

        if np.isclose(T, 550.0):
            sample_axial_df = axial_df

        print(f"T = {T:5.1f} K ({T-273.15:5.1f} C) | PFR H2={y_pfr['H2']:.3f}, CO2={y_pfr['CO2']:.3f}, CO={y_pfr['CO']:.4f} | "
              f"Equil H2={y_eq['H2']:.3f}, CO2={y_eq['CO2']:.3f}, CO={y_eq['CO']:.4f}")

    df_pfr = pd.DataFrame(pfr_results)
    df_eq = pd.DataFrame(equil_results)

    df_pfr.to_csv("pfr_outlet_vs_T.csv", index=False)
    df_eq.to_csv("equilibrium_vs_T.csv", index=False)
    if sample_axial_df is not None:
        sample_axial_df.to_csv("pfr_axial_profile_550K.csv", index=False)

    # Generate Comparison Plot: PFR Outlet vs Equilibrium across Temperature
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Main Reactants & Products (H2, CH3OH, H2O, CO2)
    for sp, color in [("H2", "#1f77b4"), ("CO2", "#2ca02c"), ("CH3OH", "#d62728"), ("H2O", "#ff7f0e")]:
        axes[0].plot(df_pfr["T_K"], df_pfr[sp], marker="o", linestyle="-", color=color, label=f"{sp} (PFR)")
        axes[0].plot(df_eq["T_K"], df_eq[sp], linestyle="--", color=color, alpha=0.75, label=f"{sp} (Equil)")
    axes[0].set_title("Methanol Steam Reforming: PFR vs. Equilibrium", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Outlet Mole Fraction")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(ncol=2, fontsize=8)

    # Right: Axial Concentration Profile along PFR at 550 K
    if sample_axial_df is not None:
        for sp, color in [("CH3OH", "#d62728"), ("H2O", "#ff7f0e"), ("H2", "#1f77b4"), ("CO2", "#2ca02c")]:
            axes[1].plot(sample_axial_df["z_m"], sample_axial_df[sp], linewidth=2, color=color, label=sp)
        axes[1].set_title("PFR Axial Mole Fraction Profile at T = 550 K", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Reactor Axial Position z (m)")
        axes[1].set_ylabel("Gas Mole Fraction")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("pfr_vs_equilibrium_comparison.png", dpi=200)
    print("\nSaved CSV outputs and comparison plot: pfr_vs_equilibrium_comparison.png")

if __name__ == "__main__":
    main()
