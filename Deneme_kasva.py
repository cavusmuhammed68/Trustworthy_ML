# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:26:06 2025

@author: cavus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import R
import math

# === Create Results Directory ===
results_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\Results"
os.makedirs(results_path, exist_ok=True)

# === Load Dataset ===
data = pd.read_csv(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\data_for_energyy.csv")
load_profile = data['consumption'].values
solar_profile = data['pv_production'].values

# === Time Parameters ===
T = len(load_profile)
time_hours = np.arange(T)
days = T // 24
time_days = np.arange(days)

# === Universal Constants ===
F = 96487  # Faraday constant [C/mol]
Tk = 298.15  # Temperature in Kelvin (25°C)
P_H2 = 3  # Hydrogen pressure [atm]
P_O2 = 0.21 * 3  # Oxygen partial pressure [atm]
Gf_liq = -228170  # Gibbs free energy of H2O [J/mol]

# === Fuel Cell Parameters ===
A_cell = 100           # Active area [cm^2]
N_cells = 90           # Number of fuel cells
i0_an, i0_cat = 1e-4, 2e-4  # Exchange current densities
Alpha, Alpha1 = 0.5, 0.085
R_ohm = 0.02           # Ohmic resistance [Ω]
k_conc = 1.1
il = 1.4               # Limiting current density

# === Fuel Cell Degradation Parameters ===
EOC = 0.7              # Open circuit voltage at t=0
a, m, n0 = 0.015, 1e-5, 1
k_vd = 3.736e-6        # Voltage degradation coefficient
krfc = 1.0             # Resistance degradation factor

# === Electrolyzer Parameters ===
V_rev = 1.48             # Reversible voltage [V]
N_el = 80                # Number of electrolyzer cells
kvi = 3e-5               # Efficiency degradation rate
krele = 0.02             # Resistance degradation factor
kel0 = 0.02              # Initial hydrogen production efficiency
P_ini_max_el = 8000      # Initial max electrolyzer power [W]
kelem = 0.004933         # Electrolyzer power degradation rate
r2, s1, s2, s3 = 0.01, 0.12, 0.001, 0.0001
t1, t2, t3 = 0.9, 0.01, 0.0002

# === Battery Parameters ===
C_ini, C_last = 100, 70      # Battery capacity range [kWh]
cycles_max = 4000
eta_ch, eta_dis = 0.95, 0.9  # Charging/discharging efficiency
delta_t = 1                  # Time step [h]

# === Hydrogen Tank ===
U_H2 = 0.95                  # Hydrogen utilization
LOH_init = 5000             # Initial hydrogen level

# === Thermal Storage & Cooling ===
eta_hs_ch, eta_hs_dis = 0.9, 0.85
eta_ac, eta_ahc = 3.5, 0.7   # COPs for air conditioner and absorption chiller

# === Cost Factors ===
cost_factors = {
    "PV": 1000, "EL": 1200, "FC": 1000, "BAT": 300,
    "HTANK": 50, "HSYS": 200, "AC": 150, "AHC": 180
}

# === Lifetime & Replacement ===
life = {
    "batt": 4000, "fc": 8000, "el": 8000,
    "hb": 10000, "ac": 10000, "ahc": 10000, "hs": 10000
}

# === Operational Cost Coefficients ===
C_start_fc, C_start_el = 5, 5
C_op_fc, C_op_el = 0.03, 0.02

# === Penalty Coefficients for Optimization (SOC, Hydrogen, etc.) ===
penalty_lambda = {
    "SOC": 2000,
    "LOH": 1500,
    "HS": 1000,
    "P_balance": 3000
}

# --- Nernst Voltage for Fuel Cell ---
def calculate_nernst_voltage(P_H2, P_O2, P_H2O, T):
    return -Gf_liq / (2 * F) - (R * T * np.log(P_H2O / (P_H2 * np.sqrt(P_O2)))) / (2 * F)

# --- Fuel Cell Losses ---
def activation_loss(i):
    b1 = np.arcsinh(i / (2 * i0_an))
    b2 = np.arcsinh(i / (2 * i0_cat))
    return (R * Tk / (Alpha * F)) * (b1 + b2)

def ohmic_loss(i):
    return i * R_ohm

def concentration_loss(i):
    return Alpha1 * (i ** k_conc) * np.log(1 - i / il) if i < il else 0

# --- Fuel Cell Degraded Voltage Model ---
def fuel_cell_degraded_voltage(ifc, t_day):
    if ifc <= 0:
        raise ValueError("Fuel cell current must be greater than zero.")
    rfc_t = krfc * k_vd * t_day
    return (EOC - rfc_t * ifc - a * np.log(ifc) - m * np.exp(ifc)) * (N_cells / (n0 * ifc))

# --- Electrolyzer Voltage Model (Degradation Aware) ---
def electrolyzer_voltage_degraded(I_A, T, t_day):
    if I_A <= 0:
        raise ValueError("Electrolyzer current must be greater than zero.")
    r1 = krele * kvi * t_day
    log_term = np.log10(t1 + t2 * T + t3 * T**2)
    return N_el * (V_rev + (r1 + r2 * T) * I_A +
                   (s1 + s2 * T + s3 * T**2) * np.log10(I_A) + log_term)

# --- Water Vapour Pressure for Nernst Equation ---
def PsatH2O(T):
    x = -2.1794 + 0.02953 * T - 9.1837e-5 * T**2 + 1.4454e-7 * T**3
    return 10 ** x / 101325  # Convert Pa to atm

# --- Hydrogen Production & Consumption ---
def hydrogen_production_from_electrolyzer(P_el, kel):
    return kel * P_el

def fuel_cell_hydrogen_consumed(N_fc, I_fc, U_H2):
    return (N_fc * I_fc) / (F * U_H2)

# --- Battery State-of-Charge Update ---
def update_SOC(SOC_prev, P_ch, P_dis, eta_ch, capacity, delta_t):
    soc = SOC_prev + (eta_ch * P_ch * delta_t - P_dis * delta_t) / capacity
    return max(0, min(soc, 1))  # Clamp between 0 and 1

def battery_capacity_remaining(C_ini, C_last, cycles, cycles_max):
    return C_ini - (C_ini - C_last) * (cycles / cycles_max)

# --- Thermal System ---
def heat_boiler_output(P_hb, eta_hb):
    return eta_hb * P_hb

def air_conditioner_output(P_ac, eta_ac):
    return eta_ac * P_ac

def absorption_chiller_output(Q_ahc, eta_ahc):
    return eta_ahc * Q_ahc

def update_heat_storage(HS_prev, Q_ch, Q_dis, eta_ch, eta_dis, delta_t):
    return HS_prev + eta_ch * Q_ch * delta_t - Q_dis * delta_t / eta_dis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Setup ===
data_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\data_for_energyy.csv"
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\Results"
os.makedirs(save_dir, exist_ok=True)

# === Load dataset ===
data = pd.read_csv(data_path)

# === Parameters ===
start_hour = 100
window = 24
t_range = np.arange(start_hour, start_hour + window)

# === Extract 24-hour window data ===
load = data['consumption'].values[t_range]
solar = data['pv_production'].values[t_range]
wind = data['wind_production'].values[t_range]
renewables = solar + wind

# === Thermal demand models ===
days_in_year = 365
day_of_year = (t_range // 24) % days_in_year

Q_heat_demand = 50 + 80 * np.cos((2 * np.pi * day_of_year) / 365)
Q_cool_demand = 60 + 100 * np.sin((2 * np.pi * (day_of_year - 30)) / 365)

# === Dispatch logic ===
renewable_used = np.minimum(load, renewables)
residual_demand = load - renewable_used

P_discharge = np.clip(residual_demand, 0, 100)
residual_after_batt = residual_demand - P_discharge

P_fc = np.clip(residual_after_batt, 0, 300)

renewable_surplus = renewables - renewable_used
P_charge = np.clip(renewable_surplus, 0, 100)
P_el = np.clip(renewable_surplus - P_charge, 0, 300)

P_boiler = Q_heat_demand
P_ac = Q_cool_demand

# === Plotting ===
plt.figure(figsize=(10, 6))

plt.step(t_range, load, label='Load', color='red', linewidth=2)
plt.step(t_range, renewables, label='Renewable Supply', color='orange', linestyle='--', linewidth=1.5)
plt.step(t_range, residual_demand, label='Residual Load', color='gray', linestyle=':')

plt.step(t_range, P_el, label='Electrolyzer', color='magenta')
plt.step(t_range, P_fc, label='Fuel Cell', color='green')
plt.step(t_range, P_charge, label='Battery Charge', color='blue')
plt.step(t_range, P_discharge, label='Battery Discharge', color='black')
plt.step(t_range, P_boiler, label='Heat Boiler', color='purple', linestyle='--')
plt.step(t_range, P_ac, label='Air Conditioner', color='skyblue', linestyle='--')

plt.title("Strategy S2: Electric Power Dispatch (24h)", fontsize=14)
plt.xlabel("Time [h]", fontsize=13)
plt.ylabel("Power [kW]", fontsize=13)
plt.xticks(t_range, labels=t_range % 24)  # show local hour
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()

fig_path = os.path.join(save_dir, "Fig_8_Electric_Dispatch_24h_Updated.png")
plt.savefig(fig_path, dpi=600)
plt.show()

print(f"✅ Saved: {fig_path}")

# === PART 14: Heating Power Scheduling (Updated) ===

# Heat demand remains as defined earlier
Qload_heat = 50 + 80 * np.cos((2 * np.pi * day_of_year) / 365)  # [kW]

# FC heat recovery (30% of electric output, capped at 80 kW)
Qfc_heat = np.clip(P_fc * 0.3, 0, 80)

# Thermal charging from renewable surplus
Qcharge = np.clip((solar + wind - load) * 0.3, 0, 100)

# Discharge from thermal storage (priority after FC + charge)
Qdischarge = np.clip(Qload_heat - Qfc_heat - Qcharge, 0, 100)

# Boiler fills remaining unmet heating demand
Qboiler = np.clip(Qload_heat - Qfc_heat - Qcharge - Qdischarge, 0, 150)

# === Plot Heating ===
plt.figure(figsize=(9, 5))
plt.step(t_range, Qload_heat, label='Heat Demand', color='red')
plt.step(t_range, Qboiler, label='Heat Boiler', color='blue')
plt.step(t_range, Qcharge, label='Thermal Charge', color='green')
plt.step(t_range, Qdischarge, label='Thermal Discharge', color='black')
plt.step(t_range, Qfc_heat, label='Fuel Cell Heating', color='purple')

plt.title("Fig. 9 (Updated). Strategy S2: Heating Power Dispatch (24h)", fontsize=14)
plt.xlabel("Time [h]", fontsize=13)
plt.ylabel("Heat [kW]", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Fig_9_Heating_Dispatch_24h_Updated.png"), dpi=600)
plt.show()


# Cooling demand (same model as before)
Qload_cool = 60 + 100 * np.sin((2 * np.pi * (day_of_year - 30)) / 365)

# Air Conditioner active 6am to 6pm
AC_supply = np.where((t_range % 24 >= 6) & (t_range % 24 < 18), 50, 0)

# Absorption Chiller handles the rest
Qahc = np.clip(Qload_cool - AC_supply, 0, 80)

# === Plot Cooling ===
plt.figure(figsize=(9, 4))
plt.step(t_range, Qload_cool, label='Cooling Load', color='red')
plt.step(t_range, AC_supply, label='Air Conditioner', color='blue')
plt.step(t_range, Qahc, label='AHC (Absorption Chiller)', color='green')

plt.title("Fig. 10 (Updated). Strategy S2: Cooling Power Dispatch (24h)", fontsize=14)
plt.xlabel("Time [h]", fontsize=13)
plt.ylabel("Cooling [kW]", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Fig_10_Cooling_Dispatch_24h_Updated.png"), dpi=600)
plt.show()


# --- Fuel Cell State ---
V_out = np.zeros(T)       # Instantaneous FC voltage
P_out = np.zeros(T)       # Instantaneous FC power
P_deg = np.zeros(T)       # Power output after degradation

# --- Hydrogen Tank State ---
nH2_prod = np.zeros(T)    # Hydrogen production by EL
nH2_cons = np.zeros(T)    # Hydrogen consumption by FC
LOH = np.zeros(T)         # Level of Hydrogen in tank
LOH[0] = LOH_init

# --- Battery State ---
SOC = np.zeros(T)
SOC[0] = 0.5              # Initial state of charge
P_ch = np.zeros(T)        # Battery charging power
P_dis = np.zeros(T)       # Battery discharging power
cycle_count = 0
battery_capacity = C_ini  # Initial battery capacity (updates over time)

# --- Electrolyzer ---
P_el_actual = np.zeros(T)  # Actual electrolyzer power used

# --- Thermal Storage ---
Q_ch = np.zeros(T)        # Heat charging
Q_dis = np.zeros(T)       # Heat discharging
HS = np.zeros(T)          # Heat storage level
HS[0] = 0

LOH_range = LOH[t_range]  # Hydrogen storage in mol

min_LOH = LOH_range.min()
max_LOH = LOH_range.max()

plt.figure(figsize=(10, 4))
ax1 = plt.gca()

# Plot LOH time series
ax1.plot(t_range, LOH_range, label='LOH', color='steelblue', linewidth=2)

# Plot min and max horizontal lines
ax1.axhline(min_LOH, linestyle=':', color='black', linewidth=2, label=f'Min LOH')
ax1.axhline(max_LOH, linestyle=':', color='red', linewidth=2, label=f'Max LOH')

# Styling
ax1.set_ylabel("Hydrogen Storage [mol]", fontsize=13)
ax1.set_xlabel("Time [h]", fontsize=13)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.legend(loc='upper right', fontsize=10)

plt.title("Fig. 12 (Updated). Hydrogen Storage with Min/Max (24h)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Fig_12_LOH_with_MinMax.png"), dpi=600)
plt.show()


import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os

# === Setup ===
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\Results"
os.makedirs(save_dir, exist_ok=True)

# === LE-SMPC: Reference Tracking ===
ref_voltage = 0.65 + 0.02 * np.sin(np.linspace(0, 2 * np.pi, 24))
horizon = 24
A_mpc = 1.0
B_mpc = 0.05
x0 = 0.6

u = cp.Variable(horizon)
x = cp.Variable(horizon + 1)

objective = cp.Minimize(
    cp.sum_squares(x[1:] - ref_voltage) + 0.01 * cp.sum_squares(u)
)

constraints = [x[0] == x0]
for t in range(horizon):
    constraints += [
        x[t + 1] == A_mpc * x[t] + B_mpc * u[t],
        u[t] >= -0.1,
        u[t] <= 0.1,
        x[t + 1] >= 0.4,
        x[t + 1] <= 0.85
    ]

cp.Problem(objective, constraints).solve()

# === Plot: Control Inputs ===
plt.figure(figsize=(10, 4))
plt.step(range(horizon), u.value, where='post', label='Control Input Δu', color='blue')
plt.title('Fig. 17.1: LE-SMPC Control Inputs (24 Hours)', fontsize=14)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Δu", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_17_1_LearningSMPC_ControlInputs.png", dpi=600)
plt.show()

# === Plot: Predicted vs Reference Voltage ===
plt.figure(figsize=(10, 4))
plt.plot(range(horizon + 1), x.value, label='Predicted Voltage', marker='o', color='green')
plt.plot(range(horizon), ref_voltage, 'r--', label='Reference Voltage')
plt.title('Fig. 17.2: LE-SMPC Voltage Prediction vs Reference', fontsize=14)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_17_2_LearningSMPC_VoltagePrediction.png", dpi=600)
plt.show()


# === Fig. 18: LE-SMPC Regime Switching (Deterministic) ===

T_steps = 500
# Create a clean, deterministic synthetic voltage profile
voltage_smpc = 0.65 + 0.05 * np.sin(np.linspace(0, 8 * np.pi, T_steps))
voltage_smpc += 0.01 * np.sin(np.linspace(0, 40 * np.pi, T_steps))  # Micro oscillations

# Regime Classification
regime_labels = []
regime_colors = {0: 'green', 1: 'orange', 2: 'red'}
regime_names = {0: 'Normal', 1: 'Adjustment', 2: 'Emergency'}

for v in voltage_smpc:
    if v > 0.7:
        regime_labels.append(0)  # Normal
    elif v > 0.6:
        regime_labels.append(1)  # Adjustment
    else:
        regime_labels.append(2)  # Emergency

# === Plot Regime Switching ===
plt.figure(figsize=(10, 3))
for regime in [0, 1, 2]:
    idx = [i for i, r in enumerate(regime_labels) if r == regime]
    plt.scatter(idx, voltage_smpc[idx], c=regime_colors[regime], label=regime_names[regime], s=20)

plt.title('Fig. 18: LE-SMPC Regime Switching Over Time', fontsize=14)
plt.xlabel('Time Step [h]', fontsize=14)
plt.ylabel('Voltage (V)', fontsize=14)
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_18_LearningSMPC_RegimeSwitching.png", dpi=600)
plt.show()



# --- Voltage Reference Placeholder (for SMPC or ML integration) ---
voltage_reference = np.full(T, 0.65)  # Can be replaced later by LSTM output


P_H2O = PsatH2O(Tk)  # Precompute H2O vapor pressure for Nernst equation

for t in range(T):
    # --- 1. Fuel Cell Operation ---
    power_required = load_profile[t]
    voltage_guess = voltage_reference[t]  # From ML controller or SMPC

    ifc = power_required / (voltage_guess * A_cell * N_cells)
    ifc = np.clip(ifc, 0.01, 1.3)  # Clamp to physical range

    E_nernst = calculate_nernst_voltage(P_H2, P_O2, P_H2O, Tk)
    V_act = activation_loss(ifc)
    V_ohmic = ohmic_loss(ifc)
    V_conc = concentration_loss(ifc)

    V_cell = max(E_nernst - V_act - V_ohmic - V_conc, 0)
    V_deg = k_vd * t
    V_degraded = max(V_cell - V_deg, 0)

    V_out[t] = V_cell
    P_out[t] = V_cell * ifc * A_cell * N_cells
    P_deg[t] = V_degraded * ifc * A_cell * N_cells

    # --- 2. Hydrogen Consumption ---
    I_fc = ifc * A_cell
    nH2_cons[t] = fuel_cell_hydrogen_consumed(N_cells, I_fc, U_H2)

    # --- 3. Hydrogen Production (Electrolyzer) ---
    P_max_t = max(P_ini_max_el - kelem * t, 0)
    P_el_used = min(solar_profile[t], P_max_t)
    kel_t = max(kel0 * (1 - kvi * t), 0)
    nH2_prod[t] = hydrogen_production_from_electrolyzer(P_el_used, kel_t)
    P_el_actual[t] = P_el_used

    # --- 4. Hydrogen Tank Update ---
    if t > 0:
        LOH[t] = max(LOH[t - 1] + nH2_prod[t] - nH2_cons[t], 0)

    # --- 5. Battery Operation ---
    net_load = load_profile[t] - P_deg[t]
    if t > 0:
        if net_load > 0 and SOC[t - 1] > 0.1:
            P_dis[t] = min(net_load, battery_capacity * SOC[t - 1] / delta_t)
            SOC[t] = update_SOC(SOC[t - 1], 0, P_dis[t], eta_ch, battery_capacity, delta_t)
            cycle_count += 0.5
        elif net_load < 0 and SOC[t - 1] < 0.95:
            P_ch[t] = min(-net_load, battery_capacity * (1 - SOC[t - 1]) / delta_t)
            SOC[t] = update_SOC(SOC[t - 1], P_ch[t], 0, eta_ch, battery_capacity, delta_t)
            cycle_count += 0.5
        else:
            SOC[t] = SOC[t - 1]

        battery_capacity = battery_capacity_remaining(C_ini, C_last, cycle_count, cycles_max)

    # --- 6. Thermal Storage Simulation ---
    if t > 0:
        hour_of_day = t % 24
        if 6 <= hour_of_day <= 18:  # Daytime heat charging
            Q_ch[t] = 5
            Q_dis[t] = 0
        else:                       # Nighttime discharging
            Q_ch[t] = 0
            Q_dis[t] = 3

        HS[t] = max(update_heat_storage(HS[t - 1], Q_ch[t], Q_dis[t], eta_hs_ch, eta_hs_dis, delta_t), 0)


# Save Directory
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\Results"
os.makedirs(save_dir, exist_ok=True)

# Common settings
months = [0, 2, 4, 6, 8, 10, 12]
colors = ['g', 'r', 'm', 'c', 'b', 'lime', 'gray']
linestyles = ['-', ':', '-', ':', '-', ':', '-']

# === Fuel Cell Voltage Degradation Plot ===
plt.figure(figsize=(6, 5))
I_fc = np.linspace(1, 200, 500)  # FC current (A)

for idx, month in enumerate(months):
    t_hr = month * 30 * 24  # Convert months to simulation hours
    rfc_t = krfc * k_vd * t_hr
    I_density = np.clip(I_fc / A_cell, 1e-6, None)  # Avoid log(0)

    V_fc = (
        EOC - rfc_t * I_fc / A_cell
        - a * np.log(I_density)
        - m * np.exp(I_density)
    )

    plt.plot(I_fc, V_fc * N_cells,
             label=f"{month} months",
             linestyle=linestyles[idx],
             color=colors[idx])

plt.axvline(200, color='darkred', linewidth=2, label="Max current")
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Fuel Cell Voltage vs Current (Degradation Over Time)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_2_FC_Degradation.png", dpi=600)
plt.show()

# === Electrolyzer Voltage Degradation Plot ===
plt.figure(figsize=(6, 5))
I_el = np.linspace(1, 200, 500)  # Electrolyzer current (A)
A_cell_el = 0.01
N_el_cells = 100
T_el = 298

for idx, month in enumerate(months):
    t_hr = month * 30 * 24
    r1_t = krele * kvi * t_hr
    I_density_el = np.clip(I_el / A_cell_el, 1e-6, None)

    V_el = (
        V_rev +
        r1_t * I_density_el +
        (s1 + s2 * T_el + s3 * T_el**2) * np.log(I_density_el) +
        np.log10(t1 + t2 * T_el + t3 * T_el**2)
    )

    V_total = V_el * N_el_cells * 0.01  # Scaled down for visualization
    plt.plot(I_el, V_total,
             label=f"{month} months",
             linestyle=linestyles[idx],
             color=colors[idx])

plt.axhline(200, color='darkred', linewidth=2, label="Max voltage")
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Electrolyzer Voltage vs Current (Degradation Over Time)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_3_Electrolyzer_Degradation.png", dpi=600)
plt.show()


import matplotlib.pyplot as plt

# === Efficiency Calculations ===
# --- Constants ---
LHV_H2 = 241800  # [J/mol] - Lower Heating Value of hydrogen
epsilon = 1e-6   # Small number to avoid division by zero

# --- Electrolyzer Efficiency [%] ---
el_eff = np.zeros(T)
for t in range(T):
    if P_el_actual[t] > 0:
        energy_out = nH2_prod[t] * LHV_H2  # J
        energy_in = P_el_actual[t] * 3600  # Convert W to J
        efficiency = (energy_out / (energy_in + epsilon)) * 100
        el_eff[t] = min(max(efficiency, 0), 100)

# --- Fuel Cell Efficiency [%] ---
fc_eff = np.zeros(T)
for t in range(T):
    if nH2_cons[t] > 0:
        energy_out = P_out[t] * 3600  # J
        energy_in = nH2_cons[t] * LHV_H2  # J
        efficiency = (energy_out / (energy_in + epsilon)) * 100
        fc_eff[t] = min(max(efficiency, 0), 100)



# --- Battery Efficiency (Round-trip approximation) ---
bat_eff = np.zeros(T)
for t in range(1, T):
    if P_ch[t] > 0:
        # Assume battery discharges previously charged energy with loss
        bat_eff[t] = (P_dis[t] * eta_dis) / (P_ch[t] / eta_ch + epsilon)

# === Plotting ===
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
time_range = np.arange(T)

# 1. SOC
axs[0, 0].plot(time_range, SOC, label='SOC', color='blue')
axs[0, 0].set_title("Battery State of Charge (SOC)")
axs[0, 0].set_ylabel("SOC (0-1)")
axs[0, 0].grid()

# 2. LOH
axs[0, 1].plot(time_range, LOH, label='LOH', color='green')
axs[0, 1].set_title("Hydrogen Tank Level (LOH)")
axs[0, 1].set_ylabel("Hydrogen [mol]")
axs[0, 1].grid()

# 3. Hydrogen Production
axs[1, 0].plot(time_range, nH2_prod, label='H2 Production', color='purple')
axs[1, 0].set_title("Hydrogen Production by Electrolyzer")
axs[1, 0].set_ylabel("mol/hour")
axs[1, 0].grid()

# 4. Hydrogen Consumption
axs[1, 1].plot(time_range, nH2_cons, label='H2 Consumption', color='orange')
axs[1, 1].set_title("Hydrogen Consumption by Fuel Cell")
axs[1, 1].set_ylabel("mol/hour")
axs[1, 1].grid()

# 5. Electrolyzer Efficiency
axs[2, 0].plot(time_range, el_eff, label='Electrolyzer Eff.', color='darkcyan')
axs[2, 0].set_title("Electrolyzer Efficiency")
axs[2, 0].set_ylabel("Efficiency")
axs[2, 0].grid()

# 6. Fuel Cell Efficiency
axs[2, 1].plot(time_range, fc_eff, label='Fuel Cell Eff.', color='crimson')
axs[2, 1].plot(time_range, bat_eff, label='Battery Eff.', color='gold')
axs[2, 1].set_title("Fuel Cell & Battery Efficiency")
axs[2, 1].set_ylabel("Efficiency")
axs[2, 1].legend()
axs[2, 1].grid()

plt.tight_layout()
plt.savefig(f"{save_dir}/Efficiency_Analysis.png", dpi=600)
plt.show()
