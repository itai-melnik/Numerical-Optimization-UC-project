from pyomo.environ import *

# --- Model Definition ---
model = ConcreteModel()

# --- Sets ---
model.G = Set()   # Generators
model.T = Set(ordered=True)   # Time periods
model.L = Set()   # Transmission lines (optional)

# --- Parameters ---
model.Pmin = Param(model.G)
model.Pmax = Param(model.G)
model.C_fuel = Param(model.G)
model.C_nl = Param(model.G)
model.C_su = Param(model.G)
model.RU = Param(model.G)
model.RD = Param(model.G)
model.UT = Param(model.G)
model.DT = Param(model.G)
model.D = Param(model.T)
model.R = Param(model.T)
model.PTDF = Param(model.L, model.G, default=0)
model.Fmax = Param(model.L)

# --- Variables ---
model.u = Var(model.G, model.T, domain=Binary)  # On/off status
model.y = Var(model.G, model.T, domain=Binary)  # Startup
model.z = Var(model.G, model.T, domain=Binary)  # Shutdown
model.p = Var(model.G, model.T, domain=NonNegativeReals)  # Power output

# --- Objective Function ---
def total_cost_rule(m):
    return sum(
        m.C_fuel[g] * m.p[g, t] + m.C_nl[g] * m.u[g, t] + m.C_su[g] * m.y[g, t]
        for g in m.G for t in m.T
    )
model.TotalCost = Objective(rule=total_cost_rule, sense=minimize)

# --- Constraints ---

# Logical constraint linking u, y, z
model.logic = ConstraintList()
for g in model.G:
    for t in model.T:
        if model.T.ord(t) > 1:
            t_prev = model.T.prev(t)
            model.logic.add(model.u[g, t] - model.u[g, t_prev] == model.y[g, t] - model.z[g, t])
        model.logic.add(model.y[g, t] + model.z[g, t] <= 1)

# Generation limits
model.gen_limits = ConstraintList()
for g in model.G:
    for t in model.T:
        model.gen_limits.add(model.p[g, t] >= model.Pmin[g] * model.u[g, t])
        model.gen_limits.add(model.p[g, t] <= model.Pmax[g] * model.u[g, t])

# Demand balance
def demand_balance_rule(m, t):
    return sum(m.p[g, t] for g in m.G) == m.D[t]
model.DemandBalance = Constraint(model.T, rule=demand_balance_rule)

# Reserve constraint
def reserve_rule(m, t):
    return sum((m.Pmax[g] - m.p[g, t]) * m.u[g, t] for g in m.G) >= m.R[t]
model.Reserve = Constraint(model.T, rule=reserve_rule)

# Ramping constraints
model.ramp = ConstraintList()
for g in model.G:
    for t in model.T:
        if model.T.ord(t) > 1:
            t_prev = model.T.prev(t)
            model.ramp.add(model.p[g, t] - model.p[g, t_prev] <= model.RU[g] * model.u[g, t_prev] + model.Pmin[g] * model.y[g, t])
            model.ramp.add(model.p[g, t_prev] - model.p[g, t] <= model.RD[g] * model.u[g, t] + model.Pmin[g] * model.z[g, t])

# Minimum up/down time constraints
model.updown = ConstraintList()
for g in model.G:
    for t in model.T:
        if model.T.ord(t) <= len(model.T) - value(model.UT[g]) + 1:
            model.updown.add(sum(model.u[g, tau] for tau in model.T if model.T.ord(tau) >= model.T.ord(t) and model.T.ord(tau) < model.T.ord(t) + value(model.UT[g])) >= model.UT[g] * model.y[g, t])
        if model.T.ord(t) <= len(model.T) - value(model.DT[g]) + 1:
            model.updown.add(sum(1 - model.u[g, tau] for tau in model.T if model.T.ord(tau) >= model.T.ord(t) and model.T.ord(tau) < model.T.ord(t) + value(model.DT[g])) >= model.DT[g] * model.z[g, t])

# Network constraints (optional)
model.network = ConstraintList()
for l in model.L:
    for t in model.T:
        model.network.add(sum(model.PTDF[l, g] * model.p[g, t] for g in model.G) <= model.Fmax[l])
        model.network.add(sum(model.PTDF[l, g] * model.p[g, t] for g in model.G) >= -model.Fmax[l])