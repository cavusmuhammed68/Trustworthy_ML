
# Multi-Energy Microgrid with Hydrogen, Battery, and Thermal Integration

This repository contains the simulation code, figures, and supplementary material 
for the paper:

> Learning-Enhanced Stochastic Predictive Control for Multi-Energy Microgrids 


The framework integrates renewable energy sources (RES), hydrogen systems, 
batteries, and thermal/cooling assets under a trustworthy optimisation and 
digital twin environment.

---

## 🔋 System Overview

The system couples **electric, thermal, and hydrogen subsystems**:

- **Electricity hub**  
  - PV panels, wind turbines, utility grid  
  - Battery Energy Storage System (BESS)  

- **Hydrogen hub**  
  - Electrolyser (EL)  
  - Hydrogen storage tank (H₂ ESS)  
  - Fuel cell (FC)  

- **Thermal and cooling hub**  
  - Heat boiler  
  - Thermal storage (heat buffer)  
  - Air conditioner (AC)  
  - Absorption chiller (AHC)  

- **Supervisory layer**  
  - Learning-Enhanced Stochastic Model Predictive Control (LE-SMPC)  
  - Trustworthy Machine Learning forecasting  
  - Digital Twin for system monitoring and resilience  

---

This will:

Simulate the dispatch of electricity, heating, cooling, and hydrogen demand

Apply degradation models (battery, electrolyser, fuel cell)

Compute strategy sizing and cost metrics (S1–S3)

Generate plots for state of charge (SoC), hydrogen balance, and cost evolution

📊 Key Features

Component degradation modelling

Battery capacity fade

Electrolyser voltage rise

Fuel cell voltage drop

Strategy comparison (S1–S3)

Different capacities for PV, FC, EL, BESS, and thermal systems

Cost evaluation (CAPEX, OPEX, penalties)

Explainable forecasting

SHAP values and feature importance for trustworthy predictions

Digital Twin integration

Real-time system state mapping

Supports adaptive optimisation and resilience

