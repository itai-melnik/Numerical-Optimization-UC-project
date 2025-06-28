import os
import pandas as pd
import pyomo.environ as pyo

def save_csv(model: pyo.ConcreteModel):
    """
    Export **all** active Pyomo variables contained in *model* to CSV files
    inside a `results` folder that lives next to this utils.py.

    Special handling:
    - `u[g,t]` (commitment) and `p[g,t]` (generation) are written in
      wide matrix form with rows = hours (T) and columns = generators (G).
    - Every other variable is written in a generic *long* format with one
      column per index dimension plus a `value` column.

    This makes it easy to open the CSVs in Excel/Pandas for further analysis.
    """
    # Ensure the results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # === Unified wide‑form export for all (g, t) variables ================
    for var in model.component_objects(pyo.Var, active=True):
        # Attempt to treat the variable as (g, t) indexed
        df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))

        has_any_data = False
        for idx in var:
            # We only handle two‑dimensional indices of the form (g, t)
            if not isinstance(idx, tuple) or len(idx) != 2:
                break  # leave the inner loop and skip this variable
            g, t = idx
            if g in model.G and t in model.T:
                df.at[t, g] = pyo.value(var[idx])
                has_any_data = True
            else:
                break  # index structure is incompatible; skip variable

        if has_any_data:
            df.index.name = "Hour"
            df.to_csv(os.path.join(results_dir, f"{var.name}.csv"))