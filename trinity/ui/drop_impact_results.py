import tkinter as tk


def show_dimensionless_results(parent, velocity, mean_diameter, reynolds, weber, ohnesorge, mean_area):
    """Show the physical quantities calculated by a Drop Impact analysis."""
    result_window = tk.Toplevel(parent)
    result_window.title("Physical Analysis Results")
    result_window.geometry("350x380")
    result_window.update_idletasks()
    result_window.wait_visibility()
    result_window.grab_set()

    results = [
        ("Impact Velocity (v):", f"{velocity:.4f} m/s"),
        ("Mean Pre-impact Diameter (D):", f"{mean_diameter:.4f} mm"),
        ("Reynolds Number (Re):", f"{reynolds:.2f}"),
        ("Weber Number (We):", f"{weber:.2f}"),
        ("Ohnesorge Number (Oh):", f"{ohnesorge:.4f}"),
        ("Mean Pre-Impact Area:", f"{mean_area:.2f} mm2\n"),
    ]

    for row, (label, value) in enumerate(results):
        tk.Label(result_window, text=label, font=("Arial", 10, "bold")).grid(
            row=row, column=0, padx=20, pady=10, sticky="e"
        )
        tk.Label(result_window, text=value, font=("Arial", 10)).grid(
            row=row, column=1, padx=20, pady=10, sticky="w"
        )

    tk.Button(
        result_window,
        text="Close",
        command=result_window.destroy,
        width=15,
    ).grid(row=6, columnspan=2, pady=15)
    return result_window
