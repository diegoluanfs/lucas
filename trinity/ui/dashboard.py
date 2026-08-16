from tkinter import messagebox


def show_home_view(app):
    app.show_edges = False
    app.table_container.pack_forget()
    app.canvas_view.pack(fill="both", expand=True)
    if app.img_result_original is not None:
        app.display_frame(app.img_result_original, update_original=False)
    app.lbl_status.configure(text="Active display screen")


def show_data_table(app):
    app.canvas_view.pack_forget()
    app.table_container.pack(fill="both", expand=True)

    headers = {
        "WCA": ["Frame", "Ângulo Esq (°)", "Ângulo Dir (°)", "Média (°)"],
        "HIS": ["Frame", "Tempo (s)", "Diâmetro (px)", "Diâm. Base (px)", "Avanc. Esq (px)", "Avanc. Dir (px)", "Âng. Esq (°)", "Âng. Dir (°)", "Histérese (°)", "Área Base (px²)"],
        "DROP": ["Tempo (s)", "Posição (mm)", "Diâmetro (mm)", "Beta", "Altura (mm)", "Área (mm²)", "dD/dt (mm/s)"]
    }

    cols = headers.get(app.current_mode, ["Dados"])
    app.tree["columns"] = cols
    for col in cols:
        app.tree.heading(col, text=col)
        app.tree.column(col, width=150, anchor="center")

    app.lbl_status.configure(text=f"Tabela de Dados: Modo {app.current_mode}")


def reset_analysis(app):
    """Clear measurement data, table entries, and the current media display."""
    if hasattr(app, 'after_id') and app.after_id:
        app.after_cancel(app.after_id)
        app.after_id = None

    for item in app.tree.get_children():
        app.tree.delete(item)

    if hasattr(app, 'vel_data'):
        app.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": []}

    if hasattr(app, 'vel_state'):
        app.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0}

    app.current_frame_idx = 0
    app.analysis_active = False
    app.vel_active = False
    app.img_result_original = None

    app.canvas_view.configure(image='', text="Awaiting media...")
    app.canvas_view.image = None

    app.lbl_status.configure(text="System reset. Ready for new measurements.")
    messagebox.showinfo("Reset", "All data and images were successfully cleaned.")
