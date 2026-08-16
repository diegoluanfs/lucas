import tkinter as tk
from tkinter import messagebox


DEFAULT_VALUES = {
    "fps_real": 10000.0,
    "escala": 0.01,
    "area_min_queda": 20.0,
    "area_min_pos": 50.0,
    "dt": 1 / 10000.0,
    "frames_antes": 3,
    "tolerancia": 3.0,
    "gray_thresh": 5.0,
    "densidade": 1000.0,
    "viscosidade": 0.001,
    "tensao": 0.072,
}

DISPLAY_NAMES = {
    "fps_real": "Real Frames Per Second (FPS)",
    "escala": "Scale (mm/pixel)",
    "area_min_queda": "Minimum Falling Area (px²)",
    "area_min_pos": "Minimum Post-Impact Area (px²)",
    "dt": "Time Delta (dt)",
    "frames_antes": "Reference Frames Before Impact",
    "tolerancia": "Base Line Tolerance (px)",
    "gray_thresh": "Grayscale Threshold",
    "densidade": "Fluid Density (kg/m³)",
    "viscosidade": "Dynamic Viscosity (Pa·s)",
    "tensao": "Surface Tension (N/m)",
}


def configure_drop_impact_parameters(parent):
    """Show the Drop Impact parameter dialog and return numeric values."""
    dialog = tk.Toplevel(parent)
    dialog.title("Analysis Parameters Configuration")
    dialog.geometry("450x500")
    dialog.grab_set()

    entries = {}
    for row, (key, value) in enumerate(DEFAULT_VALUES.items()):
        tk.Label(dialog, text=f"{DISPLAY_NAMES[key]}:").grid(
            row=row, column=0, padx=20, pady=5, sticky="e"
        )
        entry = tk.Entry(dialog)
        entry.insert(0, str(value))
        entry.grid(row=row, column=1, padx=20, pady=5)
        entries[key] = entry

    params = {}

    def confirm():
        try:
            for key in DEFAULT_VALUES:
                params[key] = float(entries[key].get())
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")

    tk.Button(
        dialog,
        text="Confirm and Continue",
        command=confirm,
        height=2,
        width=20,
    ).grid(row=len(DEFAULT_VALUES), columnspan=2, pady=25)

    parent.wait_window(dialog)
    return params if params else DEFAULT_VALUES.copy()
