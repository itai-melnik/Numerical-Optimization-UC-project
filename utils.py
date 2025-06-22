import os
import pyomo.environ
from UC_MILP import model

import pandas as pd

#TODO: not saving to results need to fix

def save_csv(model: pyomo.environ.ConcreteModel):
    """
    Export Results to CSV
    """
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create empty DataFrames for u and p
    u_df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))
    p_df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))

    # Fill DataFrames with model values
    for g in model.G:
        for t in model.T:
            u_df.at[t, g] = int(round(pyomo.value(model.u[g, t])))
            p_df.at[t, g] = float(pyomo.value(model.p[g, t]))

    # Save to CSV in the results directory
    u_df.index.name = "Hour"
    p_df.index.name = "Hour"
    u_df.to_csv(os.path.join(results_dir, "generator_commitment.csv"))
    p_df.to_csv(os.path.join(results_dir, "generator_output.csv"))