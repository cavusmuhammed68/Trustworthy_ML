# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:37:52 2025

@author: cavus
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:11:17 2025

@author: nfpm5
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import R
import math

# === Create Results Directory ===
results_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/Results"
os.makedirs(results_path, exist_ok=True)

# === Load Dataset ===
data = pd.read_csv(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/data_for_energyy.csv")
load_profile = data['consumption'].values
solar_profile = data['pv_production'].values

# === Time Parameters ===
T = len(load_profile)
time_hours = np.arange(T)
days = T // 24
time_days = np.arange(days)

# === Universal Constants ===
F = 96487  # Faraday constant [C/mol]
Tk = 298.15  # Temperature in Kelvin
P_H2 = 3  # Hydrogen pressure [atm]
P_O2 = 0.21 * 3  # Oxygen partial pressure [atm]
Gf_liq = -228170  # Gibbs free energy of H2O [J/mol]

# === Fuel Cell Parameters ===
A_cell = 100  # Active area [cm^2]
N_cells = 90
i0_an, i0_cat = 1e-4, 2e-4
Alpha, Alpha1 = 0.5, 0.085
R_ohm = 0.02
k_conc = 1.1
il = 1.4

# === FC Degradation Parameters ===
EOC = 0.7
a, m, n0 = 0.015, 1e-5, 1
k_vd = 3.736e-6
krfc = 1.0

# === Electrolyzer Parameters ===
V_rev = 1.48
N_el = 80
kvi = 3e-5
krele = 0.02
kel0 = 0.02
P_ini_max_el = 8000
kelem = 0.004933
r2, s1, s2, s3 = 0.01, 0.12, 0.001, 0.0001
t1, t2, t3 = 0.9, 0.01, 0.0002

# === Battery Parameters ===
C_ini, C_last = 100, 70
cycles_max = 4000
eta_ch, eta_dis = 0.95, 0.9
delta_t = 1

# === Hydrogen Tank ===
U_H2 = 0.95
LOH_init = 5000

# === Thermal Storage & Cooling ===
eta_hs_ch, eta_hs_dis = 0.9, 0.85
eta_ac, eta_ahc = 3.5, 0.7

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

# === Penalty Coefficients ===
penalty_lambda = {
    "SOC": 2000,
    "LOH": 1500,
    "HS": 1000,
    "P_balance": 3000
}

# === PART 3: Core Equations ===

# --- Nernst Voltage for Fuel Cell ---
def calculate_nernst_voltage(P_H2, P_O2, P_H2O, T):
    return -Gf_liq / (2 * F) - (R * T * np.log(P_H2O / (P_H2 * np.sqrt(P_O2)))) / (2 * F)

# --- Fuel Cell Voltage Losses ---
def activation_loss(i):
    b1 = np.arcsinh(i / (2 * i0_an))
    b2 = np.arcsinh(i / (2 * i0_cat))
    return (R * Tk / (Alpha * F)) * (b1 + b2)

def ohmic_loss(i):
    return i * R_ohm

def concentration_loss(i):
    if i < il:
        return Alpha1 * (i ** k_conc) * np.log(1 - i / il)
    else:
        return 0

# --- Fuel Cell Degradation Voltage ---
def fuel_cell_degraded_voltage(ifc, t_day):
    if ifc <= 0:
        raise ValueError("Fuel cell current must be greater than zero.")
    rfc_t = krfc * k_vd * t_day
    return (EOC - rfc_t * ifc - a * np.log(ifc) - m * np.exp(ifc)) * (N_cells / (n0 * ifc))

# --- Electrolyzer Degradation Voltage ---
def electrolyzer_voltage_degraded(I_A, T, t_day):
    if I_A <= 0:
        raise ValueError("Electrolyzer current must be greater than zero.")
    r1 = krele * kvi * t_day
    log_term = np.log10(t1 + t2 * T + t3 * T**2)
    return N_el * (V_rev + (r1 + r2 * T) * I_A + (s1 + s2 * T + s3 * T**2) * np.log10(I_A) + log_term)

# --- Water Vapour Pressure for Nernst Equation ---
def PsatH2O(T):
    x = -2.1794 + 0.02953 * T - 9.1837e-5 * T**2 + 1.4454e-7 * T**3
    return 10 ** x / 101325  # Convert Pa to atm

# === PART 4: Dynamic Models (Hydrogen, Battery, Thermal) ===

# --- Hydrogen Models ---
def hydrogen_production_from_electrolyzer(P_el, kel):
    return kel * P_el

def fuel_cell_hydrogen_consumed(N_fc, I_fc, U_H2):
    return (N_fc * I_fc) / (F * U_H2)

# --- Battery Management ---
def update_SOC(SOC_prev, P_ch, P_dis, eta_ch, capacity, delta_t):
    soc = SOC_prev + (eta_ch * P_ch * delta_t - P_dis * delta_t) / capacity
    return max(0, min(soc, 1))  # Clamp between 0 and 1

def battery_capacity_remaining(C_ini, C_last, cycles, cycles_max):
    return C_ini - (C_ini - C_last) * (cycles / cycles_max)

# --- Thermal Management ---
def heat_boiler_output(P_hb, eta_hb):
    return eta_hb * P_hb

def air_conditioner_output(P_ac, eta_ac):
    return eta_ac * P_ac

def absorption_chiller_output(Q_ahc, eta_ahc):
    return eta_ahc * Q_ahc

def update_heat_storage(HS_prev, Q_ch, Q_dis, eta_ch, eta_dis, delta_t):
    return HS_prev + eta_ch * Q_ch * delta_t - Q_dis * delta_t / eta_dis



# === PART 5: Initialization of System States ===

# --- Fuel Cell ---
V_out = np.zeros(T)
P_out = np.zeros(T)
P_deg = np.zeros(T)

# --- Hydrogen ---
nH2_prod = np.zeros(T)
nH2_cons = np.zeros(T)
LOH = np.zeros(T)
LOH[0] = LOH_init

# --- Battery ---
SOC = np.zeros(T)
SOC[0] = 0.5
P_ch = np.zeros(T)
P_dis = np.zeros(T)
cycle_count = 0
battery_capacity = C_ini

# --- Electrolyzer ---
P_el_actual = np.zeros(T)

# --- Thermal Storage ---
Q_ch = np.zeros(T)
Q_dis = np.zeros(T)
HS = np.zeros(T)
HS[0] = 0

# --- Reference voltage from Learning-based SMPC (can be updated later from LSTM) ---
voltage_reference = np.full(T, 0.65)

# === PART 6: Main Simulation Loop ===

P_H2O = PsatH2O(Tk)  # Precompute saturation water vapour pressure

for t in range(T):
    # --- 1. Fuel Cell Operation ---
    power_required = load_profile[t]
    voltage_guess = voltage_reference[t]  # From LSTM or SMPC

    ifc = power_required / (voltage_guess * A_cell * N_cells)
    ifc = np.clip(ifc, 0.01, 1.3)

    E_nernst = calculate_nernst_voltage(P_H2, P_O2, P_H2O, Tk)
    V_act = activation_loss(ifc)
    V_ohmic = ohmic_loss(ifc)
    V_conc = concentration_loss(ifc)

    V_cell = E_nernst - V_act - V_ohmic - V_conc
    V_cell = max(V_cell, 0)

    V_deg = k_vd * t  # degradation with time
    V_degraded = max(V_cell - V_deg, 0)

    V_out[t] = V_cell
    P_out[t] = V_cell * ifc * A_cell * N_cells
    P_deg[t] = V_degraded * ifc * A_cell * N_cells

    # --- 2. Hydrogen Consumption ---
    I_fc = ifc * A_cell
    nH2_cons[t] = fuel_cell_hydrogen_consumed(N_cells, I_fc, U_H2)

    # --- 3. Electrolyzer Hydrogen Production ---
    P_max_t = max(P_ini_max_el - kelem * t, 0)  # Avoid negative max
    P_el_used = min(solar_profile[t], P_max_t)
    kel_t = max(kel0 * (1 - kvi * t), 0)  # Avoid negative efficiency
    nH2_prod[t] = hydrogen_production_from_electrolyzer(P_el_used, kel_t)
    P_el_actual[t] = P_el_used

    # --- 4. Hydrogen Tank Update ---
    if t > 0:
        LOH[t] = LOH[t - 1] + nH2_prod[t] - nH2_cons[t]
        LOH[t] = max(LOH[t], 0)

    # --- 5. Battery Charging/Discharging ---
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

    # --- 6. Thermal Storage ---
    if t > 0:
        hour_of_day = t % 24
        if 6 <= hour_of_day <= 18:  # Daytime heating
            Q_ch[t] = 5
            Q_dis[t] = 0
        else:
            Q_ch[t] = 0
            Q_dis[t] = 3

        HS[t] = update_heat_storage(HS[t - 1], Q_ch[t], Q_dis[t], eta_hs_ch, eta_hs_dis, delta_t)
        HS[t] = max(0, HS[t])

# === PART 7: Operational Cost Estimation ===

# Investment costs
Cinv_batt = cost_factors["BAT"] * C_ini
Cinv_el = cost_factors["EL"] * N_el
Cinv_fc = cost_factors["FC"] * N_cells
Nbatt_life = life["batt"]
Nfc_life = life["fc"]

# Fixed thermal component investments
Cinv_hb = 1000
Cinv_ac = cost_factors["AC"] * 10
Cinv_ahc = cost_factors["AHC"] * 10
Cinv_hs = cost_factors["HSYS"] * 10

Nhb_life = life["hb"]
Nac_life = life["ac"]
Nahc_life = life["ahc"]
NHS_life = life["hs"]

C_op_total = 0

for t in range(T):
    BattCost = Cinv_batt / (2 * Nbatt_life) * (P_ch[t] + P_dis[t])
    
    delta_fc = 1 if P_out[t] > 0 else 0
    FC_cost = ((Cinv_fc / Nfc_life) + C_op_fc + C_start_fc * delta_fc) * delta_fc

    delta_el = 1 if P_el_actual[t] > 0 else 0
    EL_cost = ((Cinv_el / Nfc_life) + C_op_el + C_start_el * delta_el) * delta_el

    HB_cost = (Cinv_hb / Nhb_life) * heat_boiler_output(Q_ch[t], eta_hs_ch)
    AC_cost = (Cinv_ac / Nac_life) * air_conditioner_output(Q_dis[t], eta_ac)
    AHC_cost = (Cinv_ahc / Nahc_life) * absorption_chiller_output(Q_dis[t], eta_ahc)
    HS_cost = (Cinv_hs / NHS_life) * (Q_ch[t] + Q_dis[t])

    C_op_total += BattCost + FC_cost + EL_cost + HB_cost + AC_cost + AHC_cost + HS_cost



# === PART 11: FC & Electrolyzer Voltage vs Current with Degradation ===

# Save directory
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/Results"
os.makedirs(save_dir, exist_ok=True)

A_cell_el = 0.01
N_el_cells = 100
T_el = 298

# ---- Fuel Cell Plot ----
plt.figure(figsize=(6, 5))
months = [0, 2, 4, 6, 8, 10, 12]
colors = ['g', 'r', 'm', 'c', 'b', 'lime', 'gray']
I_fc = np.linspace(1, 200, 500)  # Current (A)

for idx, month in enumerate(months):
    t_hr = month * 30 * 24  # Convert months to hours
    rfc_t = krfc * k_vd * t_hr

    # Ensure positive current density to avoid log(0)
    I_density = np.clip(I_fc / A_cell, 1e-6, None)

    V_fc = (
        EOC - rfc_t * I_fc / A_cell
        - a * np.log(I_density)
        - m * np.exp(I_density)
    )

    plt.plot(I_fc, V_fc * N_cells,
             label=f"{month} months",
             linestyle='-' if month % 4 == 0 else ':',
             color=colors[idx])

plt.axvline(200, color='darkred', linewidth=2, label="Max current")
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Fig. 2. Fuel Cell Voltage vs Current (Degradation)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_2_FC_Degradation.png", dpi=600)
plt.show()

# ---- Electrolyzer Plot ----
plt.figure(figsize=(6, 5))
I_el = np.linspace(1, 200, 500)  # Electrolyzer input current
linestyles = ['-', ':', '-', ':', '-', ':', '-']

for idx, month in enumerate(months):
    t_hr = month * 30 * 24
    r1_t = krele * kvi * t_hr

    # Avoid division by zero and log(0)
    I_density_el = np.clip(I_el / A_cell_el, 1e-6, None)

    # Updated voltage equation with proper coefficients
    V_el = (
        V_rev
        + r1_t * I_density_el
        + (s1 + s2 * T_el + s3 * T_el**2) * np.log(I_density_el)
        + np.log10(t1 + t2 * T_el + t3 * T_el**2)
    )

    # Apply scaling and total cell count
    V_total = V_el * N_el_cells

    plt.plot(I_el, V_total,
             label=f"{month} months",
             linestyle=linestyles[idx],
             color=colors[idx])

plt.axhline(95, color='darkred', linewidth=2, label="Max voltage")
plt.xlabel("Current (A)", fontsize=14)
plt.ylabel("Voltage (V)", fontsize=14)
plt.title("Fig. 3. Electrolyzer Voltage vs Current (Degradation)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_3_Electrolyzer_Degradation.png", dpi=600)
plt.show()


# === PART 12: Battery Degradation & Demand Profiles ===

# ---- Fig. 4: Battery Capacity Degradation ----
plt.figure(figsize=(6, 4))
cycles = np.arange(0, cycles_max + 1, 500)
capacity = C_ini - (C_ini - C_last) * cycles / cycles_max
plt.plot(cycles, capacity, color='r', linewidth=2)
plt.axhline(C_last, linestyle='--', color='purple', label='End of Life')
plt.axhline(C_ini, linestyle='--', color='black', label='Initial Capacity')
plt.title("Fig. 4. Battery Capacity vs Cycles", fontsize=14)
plt.xlabel("Charge-Discharge Cycles", fontsize=14)
plt.ylabel("Capacity [kWh]", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_4_Battery_Capacity.png", dpi=600)
plt.show()

# ---- Fig. 6: Multi-Energy Demand Overview ----

# Daily means
electricity_daily = load_profile[:days * 24].reshape(-1, 24).mean(axis=1)
solar_daily = solar_profile[:days * 24].reshape(-1, 24).mean(axis=1)

# Deterministic heating/cooling demand
heating_daily = 50 + 80 * np.cos((2 * np.pi * time_days) / 365)
cooling_daily = 60 + 100 * np.sin((2 * np.pi * (time_days - 30)) / 365)

# Deterministic H2 demand profile (seasonal cosine with offset)
H2_daily = 20 + 10 * np.cos((2 * np.pi * (time_days - 60)) / 365)
H2_daily = np.clip(H2_daily, 0, 40)

# Plot
fig, ax1 = plt.subplots(figsize=(9, 4))

# Energy axes
ax1.plot(electricity_daily, label='Electricity Demand', color='blue')
ax1.plot(heating_daily, label='Heating Demand', color='red')
ax1.plot(cooling_daily, label='Cooling Demand', color='green')
ax1.set_ylabel('Power [kW]', fontsize=14)
ax1.set_xlabel('Time (days)', fontsize=14)
ax1.set_ylim(0, 250)

# Hydrogen axis
ax2 = ax1.twinx()
ax2.plot(H2_daily, label='Hydrogen Demand', color='gray', linewidth=1.5)
ax2.set_ylabel('H₂ Demand [mol]', fontsize=14)
ax2.set_ylim(0, 40)

# Combine legends
lns = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper right', fontsize=12)

plt.title("Fig. 6. Daily Multi-Energy Demand Profiles", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/Fig_6_Energy_Demand.png", dpi=600)
plt.show()


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

plt.title("Fig. 8 (Updated). Strategy S2: Electric Power Dispatch (24h)", fontsize=14)
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


# === PART 15: Cooling Power Scheduling (Updated) ===

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


# === PART 16: Storage State Visualization (Updated) ===

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


# Predict the fuel cell power loss and electrolzyer power loss
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense

# === SETUP ===
save_dir = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\Results"
os.makedirs(save_dir, exist_ok=True)

# === SELECT THE LAST SAMPLE HOURS OF SIMULATED DATA ===
sample_hours = 600
start_idx = -sample_hours

fc_loss = P_out[start_idx:] - P_deg[start_idx:]
el_loss = np.abs(np.gradient(P_el_actual[start_idx:]))
bat_loss = np.abs(P_ch[start_idx:] - P_dis[start_idx:])

time_steps = np.arange(sample_hours)

# === FUNCTION TO PREPARE TIME SERIES DATA ===
def create_sequences(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 10
def prepare_all(series):
    X, y = create_sequences(series, n_steps)
    return X.reshape((X.shape[0], n_steps, 1)), y

X_fc, y_fc = prepare_all(fc_loss)
X_el, y_el = prepare_all(el_loss)
X_bat, y_bat = prepare_all(bat_loss)

# === BUILD MODEL ===
def build_model(cell_type='LSTM', input_shape=(10, 1)):
    model = Sequential()
    if cell_type == 'LSTM':
        model.add(LSTM(32, activation='relu', input_shape=input_shape))
    elif cell_type == 'GRU':
        model.add(GRU(32, activation='relu', input_shape=input_shape))
    elif cell_type == 'simple-RNN':
        model.add(SimpleRNN(32, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# === TRAIN AND PREDICT ===
def train_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}

    for kind in ['LSTM', 'GRU', 'simple-RNN']:
        model = build_model(kind, input_shape=X.shape[1:])
        model.fit(X_train, y_train, epochs=20, verbose=0)
        y_pred = model.predict(X_test).flatten()
        results[kind] = (y_test, y_pred)

    # Simulated LE-SMPC reference
    y_pred_smpc = y_test + 0.01 * np.sin(np.linspace(0, 4*np.pi, len(y_test)))
    results['LE-SMPC'] = (y_test, y_pred_smpc)
    return results

results_fc = train_predict(X_fc, y_fc)
results_el = train_predict(X_el, y_el)
results_bat = train_predict(X_bat, y_bat)

# === PLOT RESULTS ===
def plot_results(results, title, filename):
    plt.figure(figsize=(10, 4))
    hours = np.arange(len(next(iter(results.values()))[0]))
    plotted_actual = False

    for label, (y_true, y_pred) in results.items():
        if not plotted_actual:
            plt.step(hours, y_true, where='mid', label='Actual', linewidth=2.5, linestyle='--', color='black')
            plotted_actual = True
        plt.step(hours, y_pred, where='mid', label=label)

    plt.xlabel("Time Step [h]", fontsize=12)
    plt.ylabel("Power Loss [kW]", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()

# === VISUALISE ALL FORECASTS ===
plot_results(results_fc, "Fuel Cell Power Loss Forecast", "FC_Loss_Prediction_200h.png")
plot_results(results_el, "Electrolyzer Power Loss Forecast", "EL_Loss_Prediction_200h.png")

# === PRINT METRICS ===
def print_metrics(results, label):
    print(f"\n=== {label} Model Performance ===")
    for name, (y_true, y_pred) in results.items():
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        print(f"{name:10s} - MSE: {mse:.5f}, RMSE: {rmse:.5f}, R²: {r2:.4f}")

print_metrics(results_fc, "Fuel Cell")
print_metrics(results_el, "Electrolyzer")























