import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.analysis import (
    adaptive_xi_plateau_threshold,
    classify_regime,
    identifiability_gate,
    pass_c_additional_runs,
    precursor_guard_active,
)
from simulator.hfuncs import h_compression
from simulator.simulate import run_sigma_tap
from simulator.state import ModelParams

params = ModelParams(alpha=1e-4, a=8.0, mu=0.01, beta=0.05, eta=0.02)
rows = run_sigma_tap(
    initial_M=3,
    steps=12,
    params=params,
    sigma0=1.0,
    gamma=0.0,
    h_func=lambda s: h_compression(s, decay=0.02),
    append_terminal_state=True,
)

transition_rows = [r for r in rows if "M_t1" in r]
terminal_row = rows[-1]

print("rows_total", len(rows), "transitions", len(transition_rows))
print("last_transition", round(transition_rows[-1]["M_t1"], 6), round(transition_rows[-1]["Xi_t1"], 6))
print("terminal_state", terminal_row["t"], round(terminal_row["M"], 6), round(terminal_row["Xi"], 6))
print("gate", identifiability_gate({"M_t"}))
print("gate2", identifiability_gate({"M_t", "Xi_proxy"}))

xi = [transition_rows[0]["Xi"]] + [r["Xi_t1"] for r in transition_rows]
m = [transition_rows[0]["M"]] + [r["M_t1"] for r in transition_rows]
thr = adaptive_xi_plateau_threshold(xi)
latest_rate = xi[-1] - xi[-2]

# Two-channel visibility check: compare beta*B and eta*H contributions.
beta_B_last = params.beta * transition_rows[-1]["B"]
eta_H_last = params.eta * transition_rows[-1]["H"]

print("xi_traj", [round(v, 8) for v in xi])
print("xi_plateau", round(thr, 12), "guard", precursor_guard_active(latest_rate, thr))
print("channel_last", {"beta_B": round(beta_B_last, 12), "eta_H": round(eta_H_last, 12)})
print("regime", classify_regime(xi, m, thr))
print("pass_c_runs_k9", pass_c_additional_runs(9))
