# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:24:09 2025

@author: nfpm5
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:27:54 2025

@author: nfpm5

Part 1 & 2: Imports, Data Loading, and System Parameter Definitions for Learning-Enhanced SMPC
"""

# === PART 1: Imports & Dataset Loading ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import R
import math
import random
import os

# For Learning-Enhanced SMPC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Create results directory if it doesn't exist
results_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/Results"
os.makedirs(results_path, exist_ok=True)

# --- Load Dataset ---
data = pd.read_csv(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/data_for_energyy.csv")
load_profile = data['consumption'].values
solar_profile = data['pv_production'].values

# --- Time Parameters ---
T = len(load_profile)                     # Total hours
time_hours = np.arange(T)                # Hourly time vector
days = T // 24
time_days = np.arange(days)              # Daily time vector


# === PART 2: System Constants & Parameters ===

# --- Universal Constants ---
F = 96487                                 # Faraday constant [C/mol]
Tk = 298.15                               # Temperature in Kelvin
P_H2 = 3                                  # Hydrogen pressure [atm]
P_O2 = 0.21 * 3                           # Oxygen partial pressure [atm]
Gf_liq = -228170                          # Gibbs free energy of H2O [J/mol]

# --- Fuel Cell Parameters ---
A_cell = 100                              # Active area [cm²]
N_cells = 90                              # Number of cells
i0_an, i0_cat = 1e-4, 2e-4                # Exchange current densities
Alpha, Alpha1 = 0.5, 0.085
R_ohm = 0.02                              # Ohmic resistance
k_conc = 1.1
il = 1.4                                  # Limiting current [A/cm²]

# --- FC Degradation Parameters ---
EOC = 0.7                                 # Open-circuit voltage [V]
a, m, n0 = 0.015, 1e-5, 1
k_vd = 3.736e-6                           # Voltage degradation rate [V/hour]
krfc = 1.0                                # Resistance degradation coefficient

# --- Electrolyzer Parameters ---
V_rev = 1.48
N_el = 80
kvi = 3e-5
krele = 0.02
kel0 = 0.02                               # Initial hydrogen efficiency
P_ini_max_el = 8000                      # Max electrolyzer power [W]
kelem = 0.004933                          # Power degradation rate
r2, s1, s2, s3 = 0.01, 0.12, 0.001, 0.0001
t1, t2, t3 = 0.9, 0.01, 0.0002

# --- Battery Parameters ---
C_ini, C_last = 100, 70                  # Initial and final capacity [kWh]
cycles_max = 4000
eta_ch = 0.95
eta_dis = 0.9
delta_t = 1                              # Time step (1 hour)

# --- Hydrogen Tank ---
U_H2 = 0.95                               # Hydrogen utilisation rate
LOH_init = 5000                          # Initial LOH [mol]

# --- Thermal Storage & Cooling ---
eta_hs_ch, eta_hs_dis = 0.9, 0.85
eta_ac = 3.5
eta_ahc = 0.7

# --- Cost Factors (per unit basis) ---
cost_factors = {
    "PV": 1000, "EL": 1200, "FC": 1000, "BAT": 300,
    "HTANK": 50, "HSYS": 200, "AC": 150, "AHC": 180
}

# --- Lifetime & Replacement ---
life = {
    "batt": 4000, "fc": 8000, "el": 8000,
    "hb": 10000, "ac": 10000, "ahc": 10000, "hs": 10000
}

# --- Start-up & Operational Costs ---
C_start_fc, C_start_el = 5, 5
C_op_fc, C_op_el = 0.03, 0.02

# --- Penalty Coefficients (from Eq. 45) ---
penalty_lambda = {
    "SOC": 2000,
    "LOH": 1500,
    "HS": 1000,
    "P_balance": 3000
}

# === PART 3: Core Equations (Fuel Cell, Electrolyzer, Battery, Thermal) ===

# ---- Nernst Voltage for Fuel Cell ----
def calculate_nernst_voltage(P_H2, P_O2, P_H2O, T):
    return -Gf_liq / (2 * F) - (R * T * np.log(P_H2O / (P_H2 * np.sqrt(P_O2)))) / (2 * F)

# ---- Fuel Cell Losses ----
def activation_loss(i): 
    b1 = np.arcsinh(i / (2 * i0_an))
    b2 = np.arcsinh(i / (2 * i0_cat))
    return (R * Tk / (Alpha * F)) * (b1 + b2)

def ohmic_loss(i): return i * R_ohm

def concentration_loss(i): 
    if i < il:
        return Alpha1 * (i ** k_conc) * np.log(1 - i / il)
    else:
        return 0

# ---- Degraded FC Voltage ----
def fuel_cell_degraded_voltage(ifc, t_day):
    rfc_t = krfc * k_vd * t_day
    return (EOC - rfc_t * ifc - a * np.log(ifc) - m * np.exp(ifc)) * (N_cells / (n0 * ifc))

# ---- Degraded Electrolyzer Voltage ----
def electrolyzer_voltage_degraded(I_A, T, t_day):
    r1 = krele * kvi * t_day
    log_term = np.log10(t1 + t2 * T + t3 * T**2)
    return N_el * (V_rev + (r1 + r2 * T) * I_A + (s1 + s2 * T + s3 * T**2) * np.log10(I_A) + log_term)

# ---- Water Vapour Pressure for Nernst ----
def PsatH2O(T):
    x = -2.1794 + 0.02953 * T - 9.1837e-5 * T**2 + 1.4454e-7 * T**3
    return 10 ** x / 101325  # Convert Pa to atm

# === PART 4: Dynamic System Models (Hydrogen, Battery, Thermal) ===

# ---- Hydrogen Handling ----
def hydrogen_production_from_electrolyzer(P_el, kel):
    return kel * P_el

def fuel_cell_hydrogen_consumed(N_fc, I_fc, U_H2):
    return (N_fc * I_fc) / (F * U_H2)

# ---- Battery Management ----
def update_SOC(SOC_prev, P_ch, P_dis, eta_ch, capacity, delta_t):
    return SOC_prev + (eta_ch * P_ch * delta_t - P_dis * delta_t) / capacity

def battery_capacity_remaining(C_ini, C_last, cycles, cycles_max):
    return C_ini - (C_ini - C_last) * (cycles / cycles_max)

# ---- Thermal Management ----
def heat_boiler_output(P_hb, eta_hb): return eta_hb * P_hb
def air_conditioner_output(P_ac, eta_ac): return eta_ac * P_ac
def absorption_chiller_output(Q_ahc, eta_ahc): return eta_ahc * Q_ahc

def update_heat_storage(HS_prev, Q_ch, Q_dis, eta_ch, eta_dis, delta_t):
    return HS_prev + eta_ch * Q_ch * delta_t - Q_dis * delta_t / eta_dis

# === PART 5: Initialization of System States and Arrays ===

# Time series length
T = len(load_profile)
time_hours = np.arange(T)

# Initialize Fuel Cell outputs
V_out = np.zeros(T)          # Fuel cell voltage
P_out = np.zeros(T)          # Power output from fuel cell
P_deg = np.zeros(T)          # Power output considering degradation

# Hydrogen
nH2_prod = np.zeros(T)       # Electrolyzer hydrogen production (mol)
nH2_cons = np.zeros(T)       # Fuel cell hydrogen consumption (mol)
LOH = np.zeros(T)            # Level of hydrogen tank (mol)
LOH[0] = LOH_init            # Initial hydrogen content

# Battery
SOC = np.zeros(T)            # State of charge
SOC[0] = 0.5
P_ch = np.zeros(T)           # Charging power
P_dis = np.zeros(T)          # Discharging power
cycle_count = 0              # Battery cycle counter
battery_capacity = C_ini     # Start with initial battery capacity

# Electrolyzer power usage
P_el_actual = np.zeros(T)

# Thermal Storage
Q_ch = np.zeros(T)           # Heat charging
Q_dis = np.zeros(T)          # Heat discharging
HS = np.zeros(T)             # Heat storage level (kWh)
HS[0] = 0

# Learning-enhanced SMPC signal placeholder (can later integrate LSTM prediction)
voltage_reference = np.full(T, 0.65)

# === PART 6: Main Simulation Loop for Each Hour ===

# Precompute saturation water vapour pressure
P_H2O = PsatH2O(Tk)

for t in range(T):
    # --- 1. Fuel Cell Calculation ---
    power_required = load_profile[t]
    voltage_guess = voltage_reference[t]  # LSTM-enhanced prediction later
    ifc = power_required / (voltage_guess * A_cell * N_cells)
    ifc = np.clip(ifc, 0.01, 1.3)

    E_nernst = calculate_nernst_voltage(P_H2, P_O2, P_H2O, Tk)
    V_act = activation_loss(ifc)
    V_ohmic = ohmic_loss(ifc)
    V_conc = concentration_loss(ifc)

    V_cell = E_nernst - V_act - V_ohmic - V_conc
    V_cell = max(V_cell, 0)
    V_deg = k_vd * time_hours[t]
    V_degraded = max(V_cell - V_deg, 0)

    V_out[t] = V_cell
    P_out[t] = V_cell * ifc * A_cell * N_cells
    P_deg[t] = V_degraded * ifc * A_cell * N_cells

    # --- 2. Hydrogen Consumption ---
    I_fc = ifc * A_cell
    nH2_cons[t] = fuel_cell_hydrogen_consumed(N_cells, I_fc, U_H2)

    # --- 3. Electrolyzer Hydrogen Production ---
    P_max_t = P_ini_max_el - kelem * t
    P_el_used = min(solar_profile[t], P_max_t)
    kel_t = kel0 * (1 - kvi * t)
    nH2_prod[t] = hydrogen_production_from_electrolyzer(P_el_used, kel_t)
    P_el_actual[t] = P_el_used

    # --- 4. Hydrogen Tank Update ---
    if t > 0:
        LOH[t] = LOH[t-1] + nH2_prod[t] - nH2_cons[t]
        LOH[t] = max(LOH[t], 0)

    # --- 5. Battery Charging/Discharging ---
    net_load = load_profile[t] - P_deg[t]
    if t > 0:
        if net_load > 0 and SOC[t-1] > 0.1:
            P_dis[t] = min(net_load, battery_capacity * SOC[t-1] / delta_t)
            SOC[t] = update_SOC(SOC[t-1], 0, P_dis[t], eta_ch, battery_capacity, delta_t)
            cycle_count += 0.5
        elif net_load < 0 and SOC[t-1] < 0.95:
            P_ch[t] = min(-net_load, battery_capacity * (1 - SOC[t-1]) / delta_t)
            SOC[t] = update_SOC(SOC[t-1], P_ch[t], 0, eta_ch, battery_capacity, delta_t)
            cycle_count += 0.5
        else:
            SOC[t] = SOC[t-1]

        battery_capacity = battery_capacity_remaining(C_ini, C_last, cycle_count, cycles_max)

    # --- 6. Thermal Storage Management (Day/Night split) ---
    if t > 0:
        if 6 <= t % 24 <= 18:
            Q_ch[t] = 5
            Q_dis[t] = 0
        else:
            Q_ch[t] = 0
            Q_dis[t] = 3

        HS[t] = update_heat_storage(HS[t-1], Q_ch[t], Q_dis[t], eta_hs_ch, eta_hs_dis, delta_t)
        HS[t] = max(0, HS[t])  # Prevent negative storage

# === PART 7: Operation Cost Estimation Based on Simulation ===

# Investment costs (per unit basis)
Cinv_batt = cost_factors["BAT"] * C_ini
Nbatt_life = life["batt"]
Cinv_el = cost_factors["EL"] * N_el
Cinv_fc = cost_factors["FC"] * N_cells
Nfc_life = life["fc"]

Cinv_hb = 1000
Cinv_ac = cost_factors["AC"] * 10
Cinv_ahc = cost_factors["AHC"] * 10
Cinv_hs = cost_factors["HSYS"] * 10

Nhb_life = life["hb"]
Nac_life = life["ac"]
Nahc_life = life["ahc"]
NHS_life = life["hs"]

C_op_total = 0  # Total operational cost

for t in range(T):
    # Battery cost per timestep
    BattCost = Cinv_batt / (2 * Nbatt_life) * (P_ch[t] + P_dis[t])

    # Fuel Cell cost
    delta_fc = 1 if P_out[t] > 0 else 0
    FC_cost = ((Cinv_fc / Nfc_life) + C_op_fc + C_start_fc * delta_fc) * delta_fc

    # Electrolyzer cost
    delta_el = 1 if P_el_actual[t] > 0 else 0
    EL_cost = ((Cinv_el / Nfc_life) + C_op_el + C_start_el * delta_el) * delta_el

    # Thermal system costs
    HB_cost = (Cinv_hb / Nhb_life) * heat_boiler_output(Q_ch[t], eta_hs_ch)
    AC_cost = (Cinv_ac / Nac_life) * air_conditioner_output(Q_dis[t], eta_ac)
    AHC_cost = (Cinv_ahc / Nahc_life) * absorption_chiller_output(Q_dis[t], eta_ahc)
    HS_cost = (Cinv_hs / NHS_life) * (Q_ch[t] + Q_dis[t])

    # Aggregate total operational cost
    C_op_total += BattCost + FC_cost + EL_cost + HB_cost + AC_cost + AHC_cost + HS_cost

# === PART 8: Strategy Sizing and Investment Summary ===

# Strategy design examples
strategy_params = {
    "S1": {"γ1": 0.1, "γ2": 1.0},
    "S2": {"γ1": 0.5, "γ2": 1.0},
    "S3": {"γ1": 0.9, "γ2": 1.0}
}

strategy_sizing = []

for name, config in strategy_params.items():
    γ1 = config["γ1"]

    Npv = int(100 + 200 * γ1)
    P_el_max = int(300 * (1 - γ1))
    P_fc_max = int(300 + 300 * γ1)
    delta_VH2 = int(1000 + 1000 * γ1)
    C_bat = int(300 + 400 * γ1)
    N_h2 = int(45 - 10 * γ1)
    P_hs_max = 120 + int(10 * γ1)
    delta_HS = 3000
    P_ac_max = 120 + int(40 * γ1)
    Q_ahc_max = 700 + int(500 * (1 - γ1))

    strategy_sizing.append([
        name, Npv, P_el_max, P_fc_max, delta_VH2, C_bat,
        N_h2, P_hs_max, delta_HS, P_ac_max, Q_ahc_max
    ])

# Display as DataFrame
df_sizing = pd.DataFrame(strategy_sizing, columns=[
    "Strategy", "Npv", "P_el_max [kW]", "P_fc_max [kW]", "ΔVH2 [Nm³]", "C_bat [kWh]",
    "N_h2 [m²]", "P_hs_max [kW]", "ΔHS [kWh]", "P_ac_max [kW]", "Q_ahc_max [kW]"
])

print("\n--- Table: Strategy Sizing Results ---")
print(df_sizing.to_string(index=False))

# === PART 9: Strategy CAPEX + OPEX + Total Cost Summary ===

strategy_costs = []

for row in strategy_sizing:
    name, Npv, Pel, Pfc, ΔVH2, Cbat, Nh2, Phs, ΔHS, Pac, Qahc = row
    γ1 = strategy_params[name]["γ1"]

    C_pv = Npv * cost_factors["PV"]
    C_el = Pel * cost_factors["EL"]
    C_fc = Pfc * cost_factors["FC"]
    C_bat = Cbat * cost_factors["BAT"]
    C_htank = ΔVH2 * cost_factors["HTANK"] / 1000
    C_hs = ΔHS * cost_factors["HSYS"] / 1000
    C_ac = Pac * cost_factors["AC"]
    C_ahc = Qahc * cost_factors["AHC"] / 1000

    C_cap = C_pv + C_el + C_fc + C_bat + C_htank + C_hs + C_ac + C_ahc
    C_op = 40000 + int(γ1 * 80000)  # mock annual OPEX per strategy size
    C_total = C_cap + C_op

    strategy_costs.append([name, f"{C_total:.2e}", f"{C_cap:.2e}", f"{C_op:.2e}"])

# Convert to DataFrame for summary
df_cost = pd.DataFrame(strategy_costs, columns=[
    "Strategy", "C_total [€]", "C_cap [€]", "C_op [€]"
])

print("\n--- Table: Strategy Cost Summary ---")
print(df_cost.to_string(index=False))

# === PART 10: Constraint Violation Tracking and Penalty Costs ===

# Counters for constraints
SOC_violations = 0
LOH_violations = 0
HS_violations = 0
P_balance_violations = 0

# Operational bounds
SOC_min, SOC_max = 0.1, 0.95
LOH_min, LOH_max = 1000, 50000
HS_min, HS_max = 0, 3000

for t in range(T):
    if SOC[t] < SOC_min or SOC[t] > SOC_max:
        SOC_violations += 1
    if LOH[t] < LOH_min or LOH[t] > LOH_max:
        LOH_violations += 1
    if HS[t] < HS_min or HS[t] > HS_max:
        HS_violations += 1

    net_power = P_out[t] + P_dis[t] - load_profile[t] + P_ch[t]
    if abs(net_power) > 10:  # 10 kW tolerance
        P_balance_violations += 1

print("\n--- Constraint Violation Report ---")
print(f"SOC violations: {SOC_violations}")
print(f"Hydrogen tank violations: {LOH_violations}")
print(f"Heat storage violations: {HS_violations}")
print(f"Power balance violations: {P_balance_violations}")

# Penalty Weights from penalty_lambda
lambda_SOC = penalty_lambda["SOC"]
lambda_LOH = penalty_lambda["LOH"]
lambda_HS = penalty_lambda["HS"]
lambda_PB = penalty_lambda["P_balance"]

# Final total penalty cost
C_penalty = (
    lambda_SOC * SOC_violations +
    lambda_LOH * LOH_violations +
    lambda_HS * HS_violations +
    lambda_PB * P_balance_violations
)

# Add to operation cost
C_total_with_penalty = C_op_total + C_penalty

# Final Cost Summary Output
print("\n--- Final Cost Breakdown ---")
print(f"Operational cost (C_op): €{C_op_total:,.2f}")
print(f"Constraint penalties (C_penalty): €{C_penalty:,.2f}")
print(f"Total cost with penalties: €{C_total_with_penalty:,.2f}")

import os

# Save directory
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/Results"
os.makedirs(save_dir, exist_ok=True)

# === PART 11: Fuel Cell & Electrolyzer Voltage vs Current with Degradation ===

# ---- Fig. 2: Fuel Cell Voltage vs Current Over Time ----
plt.figure(figsize=(6, 5))
months = [0, 2, 4, 6, 8, 10, 12]
colors = ['g', 'r', 'm', 'c', 'b', 'lime', 'gray']
I_fc = np.linspace(1, 200, 500)

for idx, month in enumerate(months):
    t_hr = month * 30 * 24
    rfc_t = krfc * k_vd * t_hr
    V_fc = EOC - rfc_t * I_fc / A_cell - a * np.log(I_fc / A_cell) - m * np.exp(I_fc / A_cell)
    plt.plot(I_fc, V_fc * N_cells, label=f"{month} months", linestyle='-' if month % 4 == 0 else ':', color=colors[idx])

plt.axvline(200, color='darkred', linewidth=2, label="Max current")
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Fig. 2. Fuel Cell Voltage vs Current (Degradation)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_2_FC_Degradation.png", dpi=600)
plt.show()

# ---- Fig. 3: Electrolyzer Voltage vs Current Over Time ----
plt.figure(figsize=(6, 5))
I_el = np.linspace(1, 200, 500)
A_cell_el = 0.01
N_el_cells = 100
T_el = 298
k_vi = -3e-5
krele = 0.01
s1, s2, s3 = 0.0001, 1e-6, 1e-6
t1, t2, t3 = 0.1, 0.001, 0.0001
linestyles = ['-', ':', '-', ':', '-', ':', '-']

for idx, month in enumerate(months):
    t_hr = month * 30 * 24
    r1_t = krele * k_vi * t_hr
    I_density = np.clip(I_el / A_cell_el, 1e-3, None)

    V_el = 0.01 * (
        V_rev
        + r1_t * I_el / A_cell_el
        + s2 * T_el + s3 * T_el**2
        + s3 * np.log(t_hr + 1) * T_el
        + (t1 + t2 * T_el + t3 * T_el**2) * np.log(I_density)
    )

    plt.plot(I_el, V_el * N_el_cells, label=f"{month} months", color=colors[idx], linestyle=linestyles[idx])

plt.axhline(95, color='darkred', linewidth=2, label="Max voltage")
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Fig. 3. Electrolyzer Voltage vs Current (Degradation)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_3_Electrolyzer_Degradation.png", dpi=600)
plt.show()

# === PART 12: Battery Capacity vs Cycles and Multi-Energy Demands ===

# ---- Fig. 4: Battery Capacity Degradation Curve ----
plt.figure(figsize=(6, 4))
cycles = np.arange(0, cycles_max + 1, 500)
capacity = C_ini - (C_ini - C_last) * cycles / cycles_max
plt.plot(cycles, capacity, color='r', linewidth=2)
plt.axhline(C_last, linestyle='--', color='purple')
plt.axhline(C_ini, linestyle='--', color='black')
plt.title("Fig. 4. Battery Capacity vs Cycles", fontsize=14)
plt.xlabel("Cycles", fontsize=14)
plt.ylabel("Capacity (kWh)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_4_Battery_Capacity.png", dpi=600)
plt.show()

# ---- Daily Average Data ----
electricity_daily = load_profile[:days * 24].reshape(-1, 24).mean(axis=1)
solar_daily = solar_profile[:days * 24].reshape(-1, 24).mean(axis=1)
heating_daily = 50 + 80 * np.cos((2 * np.pi * time_days) / 365)
cooling_daily = 60 + 100 * np.sin((2 * np.pi * (time_days - 30)) / 365)
H2_daily = 10 + 5 * np.random.randn(days)
H2_daily = np.clip(H2_daily, 0, 40)

# ---- Fig. 5 & 6: Demand Analysis ----
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(electricity_daily, label='Electricity', color='blue')
ax1.plot(heating_daily, label='Heating', color='red')
ax1.plot(cooling_daily, label='Cooling', color='green')
ax1.set_ylabel('Power (kW)', fontsize=14)
ax1.set_xlabel('Time (days)', fontsize=14)
ax1.set_ylim(0, 250)

ax2 = ax1.twinx()
ax2.plot(H2_daily, label='H2 Demand', color='gray', linewidth=1.5)
ax2.set_ylabel('H₂ (mol)', fontsize=14)
ax2.set_ylim(0, 40)

lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper right', fontsize=12)
plt.title("Fig. 6. Daily Energy Demand Profiles", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_6_Energy_Demand.png", dpi=600)
plt.show()

# ---- Fig. 7: Solar Radiation Profile ----
plt.figure(figsize=(8, 4))
plt.plot(time_days, solar_daily / 1000, 'r-', marker='o', linewidth=1)
plt.ylabel("Solar radiation (kWh/m²)", fontsize=14)
plt.xlabel("Time (days)", fontsize=14)
plt.title("Fig. 7. Solar Radiation Daily Profile", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_7_Solar_Radiation.png", dpi=600)
plt.show()

# === PART 13: Strategy S2 – Power Dispatch Over 24-Hour Window ===

# --- Time Window for 24h Simulation ---
start_hour = 100
duration = 24
t_range = np.arange(start_hour, start_hour + duration)

load = load_profile[t_range]
solar = solar_profile[t_range]

# Simulated Energy Flows
Pel = np.clip(solar - 50, 0, 300)
Pfc = np.clip(load - solar, 0, 300)
Pcharge = np.clip(solar - load, 0, 100)
Pdischarge = np.clip(load - solar - Pfc, 0, 100)
Pboiler = np.clip(100 - solar / 50, 0, 150)
Pac = np.clip(80 - solar / 100, 0, 100)

# --- Fig. 8: Electric Power Scheduling ---
plt.figure(figsize=(9, 5))
plt.plot(t_range, load, label='Load', color='red')
plt.plot(t_range, Pel, label='Electrolyzer', color='magenta')
plt.plot(t_range, Pfc, label='Fuel Cell', color='green')
plt.plot(t_range, Pcharge, label='Battery Charge', color='blue')
plt.plot(t_range, Pdischarge, label='Battery Discharge', color='black')
plt.plot(t_range, Pboiler, label='Heat Boiler', color='purple', linestyle='--')
plt.plot(t_range, Pac, label='Air Conditioner', color='lightblue', linestyle='--')

plt.title("Fig. 8. Strategy S2: Electric Power Dispatch (24h)", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Electricity (kW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_8_Electric_Dispatch_24h.png", dpi=600)
plt.show()

# --- Heat Contributions ---
Qload_heat = 250 - solar / 20
Qcharge = np.clip(solar / 40, 0, 100)
Qdischarge = np.clip(Qload_heat - Qcharge, 0, 100)
Qfc_heat = np.clip(Pfc * 0.3, 0, 80)
Qahc = np.clip(100 - Pac, 0, 80)

# --- Fig. 9: Heating Power Scheduling ---
plt.figure(figsize=(9, 5))
plt.plot(t_range, Qload_heat, label='Heat Demand', color='red')
plt.plot(t_range, Pboiler, label='Heat Boiler', color='blue')
plt.plot(t_range, Qcharge, label='Thermal Charge', color='green')
plt.plot(t_range, Qdischarge, label='Thermal Discharge', color='black')
plt.plot(t_range, Qfc_heat, label='Fuel Cell Heating', color='purple')
plt.plot(t_range, Qahc, label='AHC (Absorption Chiller)', color='pink')

plt.title("Fig. 9. Strategy S2: Heating Power Dispatch (24h)", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Heat (kW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_9_Heating_Dispatch_24h.png", dpi=600)
plt.show()

# === PART 14: Cooling Schedule & Storage State Visualization ===

# --- Fig. 10: Cooling Power Scheduling ---
cooling_load = 180 + 30 * np.sin(np.pi * (t_range - t_range[0]) / duration)
AC_supply = np.where((t_range % 24 > 6) & (t_range % 24 < 18), 50, 0)
AHC_supply = np.clip(cooling_load - AC_supply, 0, 200)

plt.figure(figsize=(9, 4))
plt.plot(t_range, cooling_load, label='Cooling Load', color='red')
plt.plot(t_range, AC_supply, label='Air Conditioner', color='blue')
plt.plot(t_range, AHC_supply, label='AHC', color='green')

plt.title("Fig. 10. Strategy S2: Cooling Power Dispatch", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Cooling Load (kW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_10_Cooling_Dispatch_24h.png", dpi=600)
plt.show()

# --- Fig. 11: LOH vs Heat Storage ---
plt.figure(figsize=(9, 4))
ax1 = plt.gca()
ax1.plot(t_range, LOH[t_range], label='LOH (mol)', color='blue')
ax1.set_ylabel("LOH (mol)", fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(t_range, HS[t_range], label='Heat Storage (kWh)', color='black')
ax2.set_ylabel("Heat Storage (kWh)", fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')

lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper left', fontsize=12)
plt.title("Fig. 11. LOH & Thermal Storage Over Time", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_11_LOH_HeatStorage_24h.png", dpi=600)
plt.show()

# --- Fig. 12: LOH vs Battery SOC ---
plt.figure(figsize=(9, 4))
ax1 = plt.gca()
ax1.plot(t_range, LOH[t_range], label='LOH (mol)', color='blue')
ax1.set_ylabel("LOH (mol)", fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(t_range, SOC[t_range] * 100, label='SOC (%)', color='black')
ax2.axhline(10, linestyle=':', color='gray', label='Min SOC')
ax2.axhline(95, linestyle=':', color='red', label='Max SOC')
ax2.set_ylabel("SOC (%)", fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')

lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper left', fontsize=12)
plt.title("Fig. 12. Battery SOC vs Hydrogen Storage", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_12_LOH_SOC_Coupling.png", dpi=600)
plt.show()

# === PART 15: Rolling Horizon Simulation (1 Week) ===

start = 2000
duration = 168  # 1 week = 168 hours
t_sim = np.arange(start, start + duration)

# Simulated power components
Pel_sim = np.clip(solar_profile[t_sim] - 50, 0, 300)
Pfc_sim = np.clip(load_profile[t_sim] - Pel_sim, 0, 300)
Pcharge_sim = np.clip(solar_profile[t_sim] - load_profile[t_sim], 0, 100)
Pdischarge_sim = np.clip(load_profile[t_sim] - solar_profile[t_sim] - Pfc_sim, 0, 100)
Pboiler_sim = np.clip(100 - solar_profile[t_sim] / 50, 0, 150)
Pac_sim = np.clip(80 - solar_profile[t_sim] / 100, 0, 100)

# --- Fig. 13: Electric Power Scheduling (1-week) ---
plt.figure(figsize=(12, 5))
plt.plot(t_sim, load_profile[t_sim], label='Power Demand', color='red')
plt.plot(t_sim, Pel_sim, label='Electrolyzer', color='magenta')
plt.plot(t_sim, Pfc_sim, label='Fuel Cell', color='green')
plt.plot(t_sim, Pcharge_sim, label='Battery Charge', color='blue')
plt.plot(t_sim, Pdischarge_sim, label='Battery Discharge', color='black')
plt.plot(t_sim, Pboiler_sim, label='Heat Boiler', color='purple', linestyle='--')
plt.plot(t_sim, Pac_sim, label='Air Conditioner', color='lightblue', linestyle='--')

plt.title("Fig. 13. 1-Week Rolling Horizon: Electric Power Dispatch", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Electricity (kW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_13_Rolling_Electric_Dispatch_Week.png", dpi=600)
plt.show()

# Simulated heating
Qload_heat_sim = 250 - solar_profile[t_sim] / 20
Qcharge_sim = np.clip(solar_profile[t_sim] / 40, 0, 100)
Qdischarge_sim = np.clip(Qload_heat_sim - Qcharge_sim, 0, 100)
Qfc_heat_sim = np.clip(Pfc_sim * 0.3, 0, 80)
Qahc_sim = np.clip(100 - Pac_sim, 0, 80)

# --- Fig. 14: Heat Power Scheduling (1-week) ---
plt.figure(figsize=(12, 5))
plt.plot(t_sim, Qload_heat_sim, label='Heat Demand', color='red')
plt.plot(t_sim, Pboiler_sim, label='Heat Boiler', color='blue')
plt.plot(t_sim, Qcharge_sim, label='Charge', color='green')
plt.plot(t_sim, Qdischarge_sim, label='Discharge', color='black')
plt.plot(t_sim, Qfc_heat_sim, label='Fuel Cell Heating', color='purple')
plt.plot(t_sim, Qahc_sim, label='AHC (Absorption Chiller)', color='pink')

plt.title("Fig. 14. 1-Week Rolling Horizon: Heat Power Dispatch", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Heat (kW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_14_Rolling_Heat_Dispatch_Week.png", dpi=600)
plt.show()

# === PART 16: Cooling Power Schedule & Final Storage State ===

# --- Simulated Cooling Load ---
cooling_load_sim = 180 + 30 * np.sin(np.pi * (t_sim - t_sim[0]) / duration)
AHC_sim = np.clip(cooling_load_sim - Pac_sim, 0, 200)

# --- Fig. 15: Cooling Power Schedule (1-week) ---
plt.figure(figsize=(12, 5))
plt.plot(t_sim, cooling_load_sim, label='Cooling Load', color='red')
plt.plot(t_sim, Pac_sim, label='Air Conditioner', color='blue')
plt.plot(t_sim, AHC_sim, label='AHC', color='green')

plt.title("Fig. 15. 1-Week Rolling Horizon: Cooling Power Dispatch", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.ylabel("Cooling Load (kW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_15_Rolling_Cooling_Dispatch_Week.png", dpi=600)
plt.show()

# --- Fig. 16: Final Storage State (SOC and LOH) ---
plt.figure(figsize=(12, 5))
ax1 = plt.gca()
ax1.plot(t_sim, LOH[t_sim], label='LOH (mol)', color='blue')
ax1.set_ylabel("LOH (mol)", fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(t_sim, SOC[t_sim] * 100, label='SOC (%)', color='black')
ax2.axhline(10, linestyle=':', color='gray', label='Min SOC')
ax2.axhline(95, linestyle=':', color='red', label='Max SOC')
ax2.set_ylabel("SOC (%)", fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')

lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper left', fontsize=12)

plt.title("Fig. 16. 1-Week Rolling Horizon: LOH and SOC Behaviour", fontsize=14)
plt.xlabel("Time (hours)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_16_Rolling_LOH_SOC_Week.png", dpi=600)
plt.show()

# === PART 17: Learning-Enhanced SMPC Voltage Control ===

import cvxpy as cp

# Simulated learned efficiency reference (LSTM or DL model could generate this)
ref_voltage = 0.65 + 0.02 * np.sin(np.linspace(0, 2 * np.pi, 24))
horizon = 24
A_mpc = 1.0
B_mpc = 0.05
x0 = 0.6  # Initial voltage state

u = cp.Variable(horizon)           # Control input
x = cp.Variable(horizon + 1)       # Voltage predictions

# --- Objective: Track voltage profile + penalize control effort ---
objective = cp.Minimize(cp.sum_squares(x[1:] - ref_voltage) + 0.01 * cp.sum_squares(u))

# --- Constraints ---
constraints = [x[0] == x0]
for t in range(horizon):
    constraints += [x[t+1] == A_mpc * x[t] + B_mpc * u[t]]
    constraints += [u[t] >= -0.1, u[t] <= 0.1]
    constraints += [x[t+1] >= 0.4, x[t+1] <= 0.85]

# Solve
prob = cp.Problem(objective, constraints)
prob.solve()

# Results
u_opt = u.value
x_opt = x.value

# --- Fig. 17.1: Control Inputs ---
plt.figure(figsize=(10, 4))
plt.step(range(horizon), u_opt, where='post', label='Control Input Δu')
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Control Input', fontsize=14)
plt.title('Fig. 17.1: LE-SMPC Control Inputs (24 Hours)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_17_1_LearningSMPC_ControlInputs.png", dpi=600)
plt.show()

# --- Fig. 17.2: Voltage Prediction ---
plt.figure(figsize=(10, 4))
plt.plot(range(horizon + 1), x_opt, marker='o', label='Predicted Voltage')
plt.plot(range(horizon), ref_voltage, 'r--', label='Reference Voltage')
plt.xlabel('Hour', fontsize=14)
plt.ylabel('Voltage (V)', fontsize=14)
plt.title('Fig. 17.2: LE-SMPC Voltage Prediction vs Reference', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_17_2_LearningSMPC_VoltagePrediction.png", dpi=600)
plt.show()

# === PART 18: SMPC Regime Classification from Predicted Voltage ===

T_steps = 500
np.random.seed(0)
voltage_smpc = 0.65 + 0.05 * np.sin(np.linspace(0, 8 * np.pi, T_steps)) + 0.02 * np.random.randn(T_steps)

regime_labels = []
regime_colors = {0: 'green', 1: 'orange', 2: 'red'}
regime_names = {0: 'Normal', 1: 'Adjustment', 2: 'Emergency'}

for v in voltage_smpc:
    if v > 0.7:
        regime_labels.append(0)  # Normal operation
    elif v > 0.6:
        regime_labels.append(1)  # Adjustment regime
    else:
        regime_labels.append(2)  # Emergency operation

# --- Fig. 18: Regime Switching Plot ---
plt.figure(figsize=(10, 3))

# Plot points per regime to enable legend
for regime in [0, 1, 2]:
    idx = [i for i, r in enumerate(regime_labels) if r == regime]
    plt.scatter(idx, voltage_smpc[idx], c=regime_colors[regime], label=regime_names[regime], s=20)

plt.title('Fig. 18: LE-SMPC Regime Switching Over Time', fontsize=14)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Voltage (V)', fontsize=14)
plt.grid(True)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_18_LearningSMPC_RegimeSwitching.png", dpi=600)
plt.show()


# === ENHANCED PART 19: Multi-Model Forecasting for Battery and Fuel Cell Efficiency ===

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Save path
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/Results"
os.makedirs(save_dir, exist_ok=True)

# === Simulated Efficiency Data ===
T_dl = 500
np.random.seed(42)
eff_batt = 0.9 + 0.05 * np.sin(np.linspace(0, 10, T_dl)) + 0.01 * np.random.randn(T_dl)
eff_fc = 0.92 + 0.03 * np.cos(np.linspace(0, 12, T_dl)) + 0.01 * np.random.randn(T_dl)

# === Prepare Sequence Data ===
def prepare_data(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 10
X_batt, y_batt = prepare_data(eff_batt, n_steps)
X_fc, y_fc = prepare_data(eff_fc, n_steps)

# LSTM reshaped inputs
X_batt_lstm = X_batt.reshape((X_batt.shape[0], X_batt.shape[1], 1))
X_fc_lstm = X_fc.reshape((X_fc.shape[0], X_fc.shape[1], 1))

# === LSTM Model (Learning-Enhanced SMPC) ===
def build_lstm():
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model_batt_lstm = build_lstm()
model_batt_lstm.fit(X_batt_lstm, y_batt, epochs=20, verbose=0)
pred_batt_lstm = model_batt_lstm.predict(X_batt_lstm).flatten()

model_fc_lstm = build_lstm()
model_fc_lstm.fit(X_fc_lstm, y_fc, epochs=20, verbose=0)
pred_fc_lstm = model_fc_lstm.predict(X_fc_lstm).flatten()

# === ML Models for Comparison ===
models = {
    "LR": LinearRegression(),
    "RF": RandomForestRegressor(n_estimators=100, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
}

# === Store Results ===
results_batt = {
    "LE-SMPC": {
        "y_pred": pred_batt_lstm,
        "MSE": mean_squared_error(y_batt, pred_batt_lstm),
        "RMSE": mean_squared_error(y_batt, pred_batt_lstm, squared=False),
        "R2": r2_score(y_batt, pred_batt_lstm)
    }
}
results_fc = {
    "LE-SMPC": {
        "y_pred": pred_fc_lstm,
        "MSE": mean_squared_error(y_fc, pred_fc_lstm),
        "RMSE": mean_squared_error(y_fc, pred_fc_lstm, squared=False),
        "R2": r2_score(y_fc, pred_fc_lstm)
    }
}

for name, model in models.items():
    # Battery
    model.fit(X_batt, y_batt)
    pred_batt = model.predict(X_batt)
    results_batt[name] = {
        "y_pred": pred_batt,
        "MSE": mean_squared_error(y_batt, pred_batt),
        "RMSE": mean_squared_error(y_batt, pred_batt, squared=False),
        "R2": r2_score(y_batt, pred_batt)
    }

    # Fuel Cell
    model.fit(X_fc, y_fc)
    pred_fc = model.predict(X_fc)
    results_fc[name] = {
        "y_pred": pred_fc,
        "MSE": mean_squared_error(y_fc, pred_fc),
        "RMSE": mean_squared_error(y_fc, pred_fc, squared=False),
        "R2": r2_score(y_fc, pred_fc)
    }

# === PLOT: Battery Forecast ===
plt.figure(figsize=(10, 4))
plt.plot(y_batt, 'k--', label='Actual', linewidth=2.5)
for name, vals in results_batt.items():
    plt.plot(vals["y_pred"], label=f"{name}")
plt.title("Fig. 19.1: Battery Efficiency Forecast - LE-SMPC vs ML Models", fontsize=14)
plt.xlabel("Time Step [h]", fontsize=14)
plt.ylabel("Efficiency", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_19_1_Battery_Forecast_AllModels.png", dpi=600)
plt.show()

# === PLOT: Fuel Cell Forecast ===
plt.figure(figsize=(10, 4))
plt.plot(y_fc, 'k--', label='Actual', linewidth=2.5)
for name, vals in results_fc.items():
    plt.plot(vals["y_pred"], label=f"{name}")
plt.title("Fig. 19.2: Fuel Cell Efficiency Forecast - LE-SMPC vs ML Models", fontsize=14)
plt.xlabel("Time Step [h]", fontsize=14)
plt.ylabel("Efficiency", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_19_2_FuelCell_Forecast_AllModels.png", dpi=600)
plt.show()

# === TABLE: Model Performance Summary ===
df_compare = pd.DataFrame([
    {
        "Model": name,
        "Battery MSE": round(results_batt[name]["MSE"], 6),
        "Battery RMSE": round(results_batt[name]["RMSE"], 6),
        "Battery R2": round(results_batt[name]["R2"], 4),
        "Fuel Cell MSE": round(results_fc[name]["MSE"], 6),
        "Fuel Cell RMSE": round(results_fc[name]["RMSE"], 6),
        "Fuel Cell R2": round(results_fc[name]["R2"], 4)
    }
    for name in results_batt
])

df_compare = df_compare.sort_values(by="Battery MSE")

# Save Table
df_compare.to_csv(f"{save_dir}/Performance_Comparison_Table.csv", index=False)

print("\n=== Model Performance Summary (MSE, RMSE, R²) ===")
print(df_compare.to_string(index=False))



# === PART 20: Feature Importance Analysis (Permutation + Heatmap) ===

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import seaborn as sns

# Flatten for ML models
X_batt_flat = X_batt.reshape((X_batt.shape[0], X_batt.shape[1]))
X_fc_flat = X_fc.reshape((X_fc.shape[0], X_fc.shape[1]))

lr_batt = LinearRegression().fit(X_batt_flat, y_batt)
lr_fc = LinearRegression().fit(X_fc_flat, y_fc)

imp_batt = permutation_importance(lr_batt, X_batt_flat, y_batt, n_repeats=10, random_state=42)
imp_fc = permutation_importance(lr_fc, X_fc_flat, y_fc, n_repeats=10, random_state=42)

# --- Fig. 20.1: Individual Feature Bar Charts ---
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].bar(range(n_steps), imp_batt.importances_mean)
axs[0].set_title("Battery Forecast Feature Importance", fontsize=14)
axs[0].set_xlabel("Lag Step", fontsize=14)
axs[0].set_ylabel("Importance", fontsize=14)
axs[0].grid(True)

axs[1].bar(range(n_steps), imp_fc.importances_mean)
axs[1].set_title("Fuel Cell Forecast Feature Importance", fontsize=14)
axs[1].set_xlabel("Lag Step", fontsize=14)
axs[1].set_ylabel("Importance", fontsize=14)
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_20_1_FeatureImportance_Battery_FC.png", dpi=600)
plt.show()

# --- Fig. 20.2: Heatmap Dashboard ---
importance_df = pd.DataFrame({
    'Battery': imp_batt.importances_mean,
    'Fuel Cell': imp_fc.importances_mean
})

plt.figure(figsize=(6, 4))
sns.heatmap(importance_df.T, cmap='viridis', annot=True, fmt=".3f", cbar=True)
plt.title("Fig. 20.2: Feature Importance Heatmap (Battery & FC)", fontsize=14)
plt.xlabel("Lag Step", fontsize=14)
plt.ylabel("System", fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_20_2_Heatmap_FeatureImportance.png", dpi=600)
plt.show()
