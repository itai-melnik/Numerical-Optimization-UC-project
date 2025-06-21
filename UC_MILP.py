from pyomo.environ import *


import pandas as pd
from pyomo.opt import SolverFactory

# Load data from CSVs
gen_df = pd.read_csv("Data/IEEE73_Data_Gen.csv")
load_df = pd.read_csv("Data/IEEE73_Data_Load.csv")

# Extract data
# demand_series = load_df.iloc[0]
buses = list(load_df.index.astype(str))  # One row = one bus
time_periods = list(range(1, 25))
generators = list(gen_df.index.astype(str))


# --- Model Definition ---
model = ConcreteModel()

# --- Sets ---
model.G = Set(initialize=generators)
model.T = Set(initialize=time_periods, ordered=True)
# model.L = Set(initialize=[]) transmission lines
model.B = Set(initialize=buses)

# --- Parameters ---
model.Pmin = Param(model.G, initialize={str(i): gen_df.loc[i, "generation_lower_limit"] for i in gen_df.index})
model.Pmax = Param(model.G, initialize={str(i): gen_df.loc[i, "generation_upper_limit"] for i in gen_df.index})
model.C_fuel = Param(model.G, initialize={str(i): gen_df.loc[i, "generation_variable_cost_pu"] for i in gen_df.index})
model.C_nl = Param(model.G, initialize={str(i): gen_df.loc[i, "commitment_no_load_cost"] for i in gen_df.index})
model.C_su = Param(model.G, initialize={str(i): gen_df.loc[i, "commitment_start_up_cost"] for i in gen_df.index})
model.RU = Param(model.G, initialize={str(i): gen_df.loc[i, "hourly_ramping_limit"] for i in gen_df.index})
model.RD = Param(model.G, initialize={str(i): gen_df.loc[i, "hourly_ramping_limit"] for i in gen_df.index})
model.UT = Param(model.G, initialize={str(i): gen_df.loc[i, "commitment_minimum_up_time"] for i in gen_df.index})
model.DT = Param(model.G, initialize={str(i): gen_df.loc[i, "commitment_minimum_down_time"] for i in gen_df.index})
model.D = Param(model.B, model.T, initialize={(str(b), t): load_df.loc[b, str(t)] for b in load_df.index for t in time_periods})
model.R = Param(model.T, initialize={t: 0.10 * sum(load_df.loc[b, str(t)] for b in load_df.index)for t in time_periods}) #10% of total demand

# --- Variables ---
model.u = Var(model.G, model.T, domain=Binary)
model.y = Var(model.G, model.T, domain=Binary)
model.z = Var(model.G, model.T, domain=Binary)
model.p = Var(model.G, model.T, domain=NonNegativeReals)

# --- Objective Function ---
def total_cost_rule(m):
    return sum(
        m.C_fuel[g] * m.p[g, t] + m.C_nl[g] * m.u[g, t] + m.C_su[g] * m.y[g, t]
        for g in m.G for t in m.T
    )
model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

# --- Constraints ---
model.logic = ConstraintList()
for g in model.G:
    for t in model.T:
        if model.T.ord(t) > 1:
            t_prev = model.T.prev(t)
            model.logic.add(model.u[g, t] - model.u[g, t_prev] == model.y[g, t] - model.z[g, t])
        model.logic.add(model.y[g, t] + model.z[g, t] <= 1)

model.gen_limits = ConstraintList()
for g in model.G:
    for t in model.T:
        model.gen_limits.add(model.p[g, t] >= model.Pmin[g] * model.u[g, t])
        model.gen_limits.add(model.p[g, t] <= model.Pmax[g] * model.u[g, t])

def demand_balance_rule(m, t): #not with transmission network model
    return sum(m.p[g, t] for g in m.G) == sum(m.D[b, t] for b in m.B)
model.DemandBalance = Constraint(model.T, rule=demand_balance_rule)

def reserve_rule(m, t):
    return sum(m.Pmax[g] * m.u[g, t] for g in m.G) >= sum(m.D[b, t] for b in m.B) + m.R[t] #check this. had to change to linear (prev was return sum((m.Pmax[g] - m.p[g, t]) * m.u[g, t] for g in m.G) >= m.R[t])
model.Reserve = Constraint(model.T, rule=reserve_rule)

model.ramp = ConstraintList()
for g in model.G:
    for t in model.T:
        if model.T.ord(t) > 1:
            t_prev = model.T.prev(t)
            model.ramp.add(model.p[g, t] - model.p[g, t_prev] <= model.RU[g] * model.u[g, t_prev] + model.Pmin[g] * model.y[g, t])
            model.ramp.add(model.p[g, t_prev] - model.p[g, t] <= model.RD[g] * model.u[g, t] + model.Pmin[g] * model.z[g, t])

model.updown = ConstraintList()
for g in model.G:
    for t in model.T:
        if model.T.ord(t) <= len(model.T) - value(model.UT[g]) + 1:
            model.updown.add(sum(model.u[g, tau] for tau in model.T if model.T.ord(tau) >= model.T.ord(t) and model.T.ord(tau) < model.T.ord(t) + value(model.UT[g])) >= model.UT[g] * model.y[g, t])
        if model.T.ord(t) <= len(model.T) - value(model.DT[g]) + 1:
            model.updown.add(sum(1 - model.u[g, tau] for tau in model.T if model.T.ord(tau) >= model.T.ord(t) and model.T.ord(tau) < model.T.ord(t) + value(model.DT[g])) >= model.DT[g] * model.z[g, t])

# --- Solve ---
solver = SolverFactory('cbc')
#options just testing
solver.options.update({
    'seconds': 120,        # Max time in seconds
    'ratioGap': 0.01,      # 1% optimality gap
    'maxSolutions': 1      # Stop after first feasible solution
})
results = solver.solve(model, tee=True, )

# # --- Output ---
# print("\n--- Generator Commitment Schedule (u[g, t]) ---")
# for g in model.G:
#     for t in model.T:
#         if value(model.u[g, t]) > 0.5:  # Binary variable
#             print(f"Generator {g} ON at hour {t}")
            
            
# print("\n--- Generator Power Output (p[g, t]) ---")
# for g in model.G:
#     for t in model.T:
#         p_val = value(model.p[g, t])
#         if p_val > 0.01:  # Filter tiny values
#             print(f"Generator {g} produces {p_val:.2f} MW at hour {t}")
# #model.u.display()
# # model.y.display()
# # model.z.display()


#TODO: move to utils.py
# --- Export Results to CSV ---
import pandas as pd

# Create empty DataFrames for u and p
u_df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))
p_df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))

# Fill DataFrames with model values
for g in model.G:
    for t in model.T:
        u_df.at[t, g] = int(round(value(model.u[g, t])))
        p_df.at[t, g] = float(value(model.p[g, t]))

# Save to CSV
u_df.index.name = "Hour"
p_df.index.name = "Hour"
u_df.to_csv("generator_commitment.csv")
p_df.to_csv("generator_output.csv")