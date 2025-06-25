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

    # === Explicit wide‑form export for u and p ===============================
    if hasattr(model, "u"):
        u_df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))
        for g in model.G:
            for t in model.T:
                u_df.at[t, g] = int(round(pyo.value(model.u[g, t])))
        u_df.index.name = "Hour"
        u_df.to_csv(os.path.join(results_dir, "u.csv"))

    if hasattr(model, "p"):
        p_df = pd.DataFrame(index=sorted(model.T), columns=sorted(model.G))
        for g in model.G:
            for t in model.T:
                p_df.at[t, g] = float(pyo.value(model.p[g, t]))
        p_df.index.name = "Hour"
        p_df.to_csv(os.path.join(results_dir, "p.csv"))

    # === Generic long‑form export for *all* remaining Pyomo Vars ============
    for var in model.component_objects(pyo.Var, active=True):
        if var.name in {"u", "p"}:
            continue  # already exported

        rows = []
        for idx in var:
            # idx may be a simple index or a tuple of indices
            idx_tuple = idx if isinstance(idx, tuple) else (idx,)
            rows.append((*idx_tuple, pyo.value(var[idx])))

        if not rows:
            continue  # skip empty vars

        n_idx = len(rows[0]) - 1
        col_names = [f"idx_{i}" for i in range(n_idx)] + ["value"]
        df = pd.DataFrame(rows, columns=col_names)
        df.to_csv(os.path.join(results_dir, f"{var.name}.csv"), index=False)