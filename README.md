# Catalytic Steam Reforming of Methanol (CH3OH): PFR Microkinetics vs. Equilibrium (Python)

A complete Python implementation of the catalytic **Plug Flow Reactor (PFR)** microkinetic model and **Gibbs Free Energy Equilibrium** solver for **Methanol Steam Reforming ($	ext{CH}_3	ext{OH} + 	ext{H}_2	ext{O} ightleftharpoons 	ext{CO}_2 + 3	ext{H}_2$)** over $	ext{Ni@Cu(110)}$, based on *ACS Catal.* 2022, 12, 512–526.

---

## Project Structure
```text
├── 2steamreform.xml                  # Surface reaction mechanism (AT^beta Arrhenius & sticking parameters)
├── therm.dat                         # NASA 7-coefficient thermodynamic polynomial database
├── thermo_equil.py                   # NASA polynomial parser & Gibbs free energy equilibrium solver
├── pfr_microkinetics.py              # Coupled surface coverage & 1D catalytic PFR ODE solver
├── run_simulation.py                 # Main script sweeping temperature & comparing PFR vs. Equilibrium
├── pfr_outlet_vs_T.csv               # Generated PFR outlet mole fractions across temperature
├── equilibrium_vs_T.csv              # Generated equilibrium mole fractions across temperature
├── pfr_axial_profile_550K.csv        # Generated axial concentration profile along the PFR at 550 K
└── pfr_vs_equilibrium_comparison.png # Comparison plot of PFR outlet vs. equilibrium & axial profile
```

---

## How to Run
```bash
pip install numpy pandas scipy matplotlib
python run_simulation.py
```
