# Part 1: Setup, Imports, Constants ----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R
from scipy.optimize import fsolve
import cvxpy as cp
import pandas as pd


data = pd.read_csv(r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated/data_for_energyy.csv")

# Access columns correctly
load_profile = data['consumption'].values
solar_profile = data['pv_production'].values


# Constants and Fuel Cell Parameters
F = 96487  # Faraday's constant [C/mol]
Tk = 298.15  # Temperature in Kelvin (25°C)
P_H2 = 3  # Hydrogen partial pressure [atm]
P_O2 = 0.21 * 3  # Oxygen partial pressure [atm]
Gf_liq = -228170  # Gibbs free energy of H2O [J/mol]
i0_an = 0.0001  # Anode exchange current density [A/cm2]
i0_cat = 0.0002  # Cathode exchange current density [A/cm2]
Alpha = 0.5  # Charge transfer coefficient
Alpha1 = 0.085  # Concentration loss factor
R_ohm = 0.02  # Ohmic resistance [Ohm]
k_conc = 1.1
il = 1.4  # Limiting current [A/cm2]
A_cell = 100  # Active area per cell [cm2]
N_cells = 90  # Number of cells

# Degradation Parameters
k_vd = 3.736e-6  # Voltage degradation rate [V/hour]
V_init = -Gf_liq / (2 * F)

# Time Setup
T = len(load_profile)  # Total time steps from dataset
V_out = np.zeros(T)
P_out = np.zeros(T)
P_deg = np.zeros(T)

def calculate_nernst_voltage(P_H2, P_O2, P_H2O, Tk):
    return -Gf_liq / (2 * F) - (R * Tk * np.log(P_H2O / (P_H2 * np.sqrt(P_O2)))) / (2 * F)

def activation_loss(i):
    b1 = np.arcsinh(i / (2 * i0_an))
    b2 = np.arcsinh(i / (2 * i0_cat))
    return (R * Tk / (Alpha * F)) * (b1 + b2)

def ohmic_loss(i):
    return i * R_ohm

def concentration_loss(i):
    term = 1 - i / il
    return Alpha1 * (i ** k_conc) * np.log(term) if term > 0 else 0


# Assume saturated water vapour pressure at cell temperature (in atm)
def PsatH2O(T):
    x = -2.1794 + 0.02953 * T - 9.1837e-5 * T**2 + 1.4454e-7 * T**3
    return 10 ** x / 101325  # Convert Pa to atm

P_H2O = PsatH2O(Tk)
time_hours = np.arange(T)  # For degradation model (1 hour resolution)

for t in range(T):
    # Calculate total current density needed to meet load (approximate)
    power_required = load_profile[t]
    voltage_guess = 0.7  # Start with a reasonable voltage guess
    current_density_guess = power_required / (voltage_guess * A_cell * N_cells)
    current_density_guess = np.clip(current_density_guess, 0.01, 1.3)

    # Compute individual voltage losses
    E_nernst = calculate_nernst_voltage(P_H2, P_O2, P_H2O, Tk)
    V_act = activation_loss(current_density_guess)
    V_ohmic = ohmic_loss(current_density_guess)
    V_conc = concentration_loss(current_density_guess)

    # Degraded voltage calculation
    V_cell = E_nernst - V_act - V_ohmic - V_conc
    V_cell = max(V_cell, 0)
    V_deg = k_vd * time_hours[t]
    V_degraded = max(V_cell - V_deg, 0)

    # Store outputs
    V_out[t] = V_cell
    P_out[t] = V_cell * current_density_guess * A_cell * N_cells
    P_deg[t] = V_degraded * current_density_guess * A_cell * N_cells


import os

# Define the save path
save_path = r"C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\Adib_updated\Results"

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# Plot Voltage over time
plt.figure(figsize=(10, 5))
plt.plot(time_hours, V_out, label='Voltage without Degradation')
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
plt.title('Fuel Cell Voltage Profile Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(save_path, "Voltage_Profile_Over_Time.png"), dpi=600)
plt.show()

# Plot Power comparison
plt.figure(figsize=(10, 5))
plt.plot(time_hours, P_out, label='Power without Degradation')
plt.plot(time_hours, P_deg, label='Power with Degradation', linestyle='--')
plt.plot(time_hours, load_profile, label='Load Demand', linestyle=':')
plt.xlabel('Time (hours)')
plt.ylabel('Power (W)')
plt.title('Power Output vs Load Demand')
plt.grid(True)
plt.legend()
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(save_path, "Power_Output_vs_Load_Demand.png"), dpi=600)
plt.show()



# MPC Parameters
horizon = 24  # 24-hour horizon
x0 = 0.6  # Initial output (normalised voltage)
ref = 0.65 * np.ones(horizon)  # Target reference trajectory
A_mpc = 1.0
B_mpc = 0.05

# Variables
u = cp.Variable(horizon)           # Control input: small adjustments
x = cp.Variable(horizon + 1)       # Output voltage predictions

# Objective: minimise deviation from reference + control effort
objective = cp.Minimize(cp.sum_squares(x[1:] - ref) + 0.01 * cp.sum_squares(u))

# Constraints
constraints = [x[0] == x0]
for t in range(horizon):
    constraints += [x[t+1] == A_mpc * x[t] + B_mpc * u[t]]
    constraints += [u[t] >= -0.1, u[t] <= 0.1]         # Bounded control
    constraints += [x[t+1] >= 0.4, x[t+1] <= 0.85]     # Safe voltage limits

# Solve MPC problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Extract results
u_opt = u.value
x_opt = x.value


# Plot MPC predicted voltage and control input
plt.figure(figsize=(10, 4))
plt.step(range(horizon), u_opt, where='post', label='Control Input Δu')
plt.xlabel('Hour')
plt.ylabel('Control Input')
plt.title('MPC Control Actions Over 24 Hours')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "MPC Control Actions Over 24 Hours.png"), dpi=600)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(horizon + 1), x_opt, marker='o', label='Predicted Voltage')
plt.plot(range(horizon), ref, 'r--', label='Voltage Reference')
plt.xlabel('Hour')
plt.ylabel('Voltage (V)')
plt.title('MPC Predicted Voltage vs Reference')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "MPC Predicted Voltage vs Reference.png"), dpi=600)
plt.show()


rolling_horizon = 24
x0 = 0.6
voltage_mpc_log = []
control_mpc_log = []

for start in range(0, T - rolling_horizon, rolling_horizon):
    # Define reference voltage profile (e.g., based on load scaling)
    ref_segment = np.clip(0.55 + 0.001 * load_profile[start:start+rolling_horizon], 0.5, 0.85)

    # Define variables
    u = cp.Variable(rolling_horizon)
    x = cp.Variable(rolling_horizon + 1)
    
    # Objective function
    objective = cp.Minimize(cp.sum_squares(x[1:] - ref_segment) + 0.01 * cp.sum_squares(u))
    
    # Constraints
    constraints = [x[0] == x0]
    for t in range(rolling_horizon):
        constraints += [x[t+1] == A_mpc * x[t] + B_mpc * u[t]]
        constraints += [u[t] >= -0.1, u[t] <= 0.1]
        constraints += [x[t+1] >= 0.4, x[t+1] <= 0.85]
    
    # Solve and store
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    # Update state
    x0 = x.value[1]
    voltage_mpc_log.append(x.value[1])
    control_mpc_log.append(u.value[0])  # Only apply first control input


# Reconstruct predicted profile
mpc_voltage_full = np.zeros(T)
mpc_voltage_full[:len(voltage_mpc_log)] = voltage_mpc_log
mpc_power_full = np.zeros(T)

for t in range(len(voltage_mpc_log)):
    # Assume same current as before for fair comparison
    current_density = load_profile[t] / (mpc_voltage_full[t] * A_cell * N_cells)
    current_density = np.clip(current_density, 0.01, 1.3)
    
    # Degraded voltage output
    V_deg_t = max(mpc_voltage_full[t] - k_vd * t, 0)
    mpc_power_full[t] = V_deg_t * current_density * A_cell * N_cells


import random

# Simplified GA settings
num_generations = 30
population_size = 20
mutation_rate = 0.2

# Bounds for sizing
fc_area_range = (50, 150)        # cm^2
battery_capacity_range = (10, 100)  # kWh
best_fitness = float('inf')
best_solution = None

def fitness_function(fc_area, battery_capacity):
    # Simple scaling of existing power model
    predicted_power = V_out * fc_area * N_cells
    error = np.mean((predicted_power - load_profile) ** 2)

    # Capital costs (mocked)
    cost_fc = 800 * fc_area / 100
    cost_battery = 300 * battery_capacity
    total_cost = cost_fc + cost_battery + 1000 * error
    return total_cost

# Initialize population
population = [
    (random.uniform(*fc_area_range), random.uniform(*battery_capacity_range))
    for _ in range(population_size)
]

for generation in range(num_generations):
    fitness_scores = [fitness_function(fc, bat) for fc, bat in population]
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]

    if fitness_scores[0] < best_fitness:
        best_fitness = fitness_scores[0]
        best_solution = sorted_population[0]

    # Elitism + mutation
    new_population = sorted_population[:5]
    while len(new_population) < population_size:
        parent = random.choice(sorted_population[:10])
        child = list(parent)
        if random.random() < mutation_rate:
            child[0] = random.uniform(*fc_area_range)
        if random.random() < mutation_rate:
            child[1] = random.uniform(*battery_capacity_range)
        new_population.append(tuple(child))
    population = new_population


best_fc_area, best_battery_capacity = best_solution
predicted_power = V_out * best_fc_area * N_cells
op_error = np.mean((predicted_power - load_profile) ** 2)

# Capital cost estimation
capex_fc = 800 * best_fc_area / 100
capex_battery = 300 * best_battery_capacity
op_cost = 1000 * op_error
total_cost = capex_fc + capex_battery + op_cost

print("---- Optimised Sizing Results ----")
print(f"Best FC Area: {best_fc_area:.2f} cm²")
print(f"Best Battery Capacity: {best_battery_capacity:.2f} kWh")
print(f"CAPEX (Fuel Cell): €{capex_fc:.2f}")
print(f"CAPEX (Battery): €{capex_battery:.2f}")
print(f"OPEX (Penalty): €{op_cost:.2f}")
print(f"Total Estimated Cost: €{total_cost:.2f}")









