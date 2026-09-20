"""
data_manager.py
---------------
Handles persistent JSON storage of the 8 solids and their trial data,
plus building/exporting the final comparison table (Excel or CSV).
"""

import json
import os
import pandas as pd

import calculations as calc

DATA_FILE = "rolling_bodies_data.json"
MAX_SOLIDS = 8


def _default_solid(slot_no):
    return {
        "slot": slot_no,
        "name": None,
        "mass_kg": None,
        "radius_m": None,
        "theta_deg": None,
        "distance_m": None,
        "I_theory": None,
        "use_friction": False,
        "mu_friction": 0.0,
        "use_drag": False,
        "cd_drag": 0.0,
        "trials_ms": [None, None, None],
    }


def load_data(path=DATA_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {"solids": [_default_solid(i + 1) for i in range(MAX_SOLIDS)]}
    return data


def save_data(data, path=DATA_FILE):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_solid(data, slot_no):
    for s in data["solids"]:
        if s["slot"] == slot_no:
            return s
    raise ValueError(f"No such slot: {slot_no}")


def build_results_table(data):
    """
    Builds the full comparison DataFrame with one row per solid:
    Solid | Trial1 | Trial2 | Trial3 | Avg Time (ms) | Distance (m) |
    Mass (kg) | Radius (m) | Theta (deg) | a_exp (m/s^2) |
    I_exp (kg.m^2) | I_corr (kg.m^2) | Correction (kg.m^2) |
    I_theory (kg.m^2) | % Error | % Error corr | Loss model

    I_exp is the measured moment of inertia with all losses folded in;
    I_corr removes the enabled friction / air-drag losses (equal to I_exp
    when no losses are enabled). Correction = I_exp - I_corr.
    """
    rows = []
    for s in data["solids"]:
        trials = s["trials_ms"]
        avg_ms = calc.average_time_ms(trials)

        valid = [t for t in trials if t is not None]
        std_ms = spread = None
        if len(valid) >= 2:
            mean = sum(valid) / len(valid)
            std_ms = (sum((t - mean) ** 2 for t in valid) / (len(valid) - 1)) ** 0.5
            spread = (max(valid) - min(valid)) / mean * 100.0

        a_exp = i_exp = err = None
        i_corr = err_corr = None
        loss_model = []
        correction = None
        ready = (avg_ms and s["distance_m"] and s["mass_kg"] and
                 s["radius_m"] and s["theta_deg"] is not None)
        if ready:
            t_s = avg_ms / 1000.0
            try:
                a_exp = calc.experimental_acceleration(s["distance_m"], t_s)
                i_exp = calc.experimental_moment_of_inertia(
                    s["mass_kg"], s["radius_m"], s["theta_deg"], a_exp)
                if s["I_theory"] is not None:
                    err = calc.percent_error(i_exp, s["I_theory"])
            except (ValueError, ZeroDivisionError):
                pass

            use_friction = bool(s.get("use_friction", False))
            use_drag = bool(s.get("use_drag", False))
            try:
                mu = s.get("mu_friction", 0.0) or 0.0
                cd = s.get("cd_drag", 0.0) or 0.0
                v_avg = s["distance_m"] / t_s
                f_drag = calc.drag_force(v_avg, cd=cd if use_drag else 0.0,
                                         area_m2=calc.circle_area(s["radius_m"]))
                i_corr = calc.corrected_moment_of_inertia(
                    s["mass_kg"], s["radius_m"], s["theta_deg"], a_exp,
                    mu=mu if use_friction else 0.0, drag_force_n=f_drag)
                if s["I_theory"] is not None:
                    err_corr = calc.percent_error(i_corr, s["I_theory"])
                if i_exp is not None:
                    correction = i_exp - i_corr
            except (ValueError, ZeroDivisionError):
                pass

            if s.get("use_friction", False):
                loss_model.append(f"friction mu={s.get('mu_friction', 0):g}")
            if s.get("use_drag", False):
                loss_model.append(f"drag Cd={s.get('cd_drag', 0):g}")

        rows.append({
            "Solid": s["name"] or f"Slot {s['slot']} (unconfigured)",
            "Trial 1 (ms)": trials[0],
            "Trial 2 (ms)": trials[1],
            "Trial 3 (ms)": trials[2],
            "Avg Time (ms)": round(avg_ms, 3) if avg_ms else None,
            "Std Dev (ms)": round(std_ms, 2) if std_ms is not None else None,
            "Spread (%)": round(spread, 2) if spread is not None else None,
            "Distance (m)": s["distance_m"],
            "Mass (kg)": s["mass_kg"],
            "Radius (m)": s["radius_m"],
            "Theta (deg)": s["theta_deg"],
            "a_exp (m/s^2)": round(a_exp, 5) if a_exp else None,
            "I_exp (kg.m^2)": f"{i_exp:.4e}" if i_exp is not None else None,
            "I_corr (kg.m^2)": f"{i_corr:.4e}" if i_corr is not None else None,
            "Correction (kg.m^2)": f"{correction:.4e}" if correction is not None else None,
            "I_theory (kg.m^2)": f"{s['I_theory']:.4e}" if s["I_theory"] else None,
            "% Error": round(err, 2) if err is not None else None,
            "% Error corr": round(err_corr, 2) if err_corr is not None else None,
            "Loss model": ", ".join(loss_model) or "none",
        })

    return pd.DataFrame(rows)


def export_table(data, path="rolling_bodies_results.xlsx"):
    df = build_results_table(data)
    if path.lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path, df