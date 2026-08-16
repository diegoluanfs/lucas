import customtkinter as ctk
from tkinter import messagebox

from trinity.ui.tooltip import Tooltip


def show_analysis_options(app):
    """Create a small analysis-selection popup for the current mode."""
    app.option_win = ctk.CTkToplevel(app)
    app.option_win.title("Analysis Options")
    app.option_win.geometry("300x400")
    app.option_win.attributes("-topmost", True)

    ctk.CTkLabel(app.option_win, text="Select Chart Type", font=("Roboto", 16, "bold")).pack(pady=10)

    if app.current_mode == "WCA":
        ctk.CTkButton(app.option_win, text="Angles vs sample", command=lambda: app.plot_graph("WCA_EVO")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Distribuição (Histograma)", command=lambda: app.plot_graph("WCA_HIST")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Angles vs sample (Tree Points)", command=lambda: app.plot_graph("WCA_EVO_TWO")).pack(pady=5, padx=20, fill="x")

    elif app.current_mode == "HIS":
        ctk.CTkButton(app.option_win, text="Diameter vs. Time", command=lambda: app.plot_graph("HIS_DIA_TIME")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="CA vs. Time", command=lambda: app.plot_graph("HIS_TIME_CA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Angles vs. Diameter", command=lambda: app.plot_graph("HIS_ANG_DIA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Angles vs Contact Area", command=lambda: app.plot_graph("HIS_ANG_AREA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Contact Angle (medio) vs Diameter", command=lambda: app.plot_graph("HIS_DIA_CA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Contact Angle (medio) vs Diameter (FIT)", command=lambda: app.plot_graph("HIS_DIA_CA_FIT")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Contact Angles vs Diameter (FIT)", command=lambda: app.plot_graph("HIS_ANG_DIA_FIT")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Multiple plots", command=lambda: app.plot_graph("HIS_MULT")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Multiple plots", command=lambda: app.plot_graph("HIS_MULT_FIT")).pack(pady=5, padx=20, fill="x")

    elif app.current_mode == "DROP":
        ctk.CTkButton(app.option_win, text="Diameter vs. Time", command=lambda: app.plot_graph("DROP_DIA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Beta vs Time", command=lambda: app.plot_graph("DROP_BETA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Beta vs Diameter", command=lambda: app.plot_graph("DROP_BETA_DIA")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Beta vs Time", command=lambda: app.plot_graph("DROP_BETA_MAX")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Beta vs Time (FIT)", command=lambda: app.plot_graph("DROP_BETA_MAX_FIT")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Beta vs Time (Complete)", command=lambda: app.plot_graph("DROP_BETA_MAX_FIT_BEFORE_AFTER")).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(app.option_win, text="Diameter X H", command=lambda: app.plot_graph("DROP_DIA_ALT")).pack(pady=5, padx=20, fill="x")


def create_main_menu(app):
    """Create the main analysis mode menu for the sidebar."""
    if hasattr(app, 'ajustes_container') and app.ajustes_container:
        app.ajustes_container.destroy()
        app.ajustes_container = None

    for child in app.menu_container.winfo_children():
        child.destroy()

    btn_static = ctk.CTkButton(app.menu_container, text="Static Angle", command=lambda: show_sub_menu(app, "WCA"))
    btn_static.pack(pady=10, fill="x")
    Tooltip(btn_static, "Measure the contact angle of a stationary droplet using different fitting methods.")

    btn_hysteresis = ctk.CTkButton(app.menu_container, text="Hysteresis", command=lambda: show_sub_menu(app, "HIS"))
    btn_hysteresis.pack(pady=10, fill="x")
    Tooltip(btn_hysteresis, "Analyze advancing and receding angles to determine surface wettability properties.")

    btn_dropimpact = ctk.CTkButton(app.menu_container, text="Drop Impact", command=lambda: show_sub_menu(app, "DROP"))
    btn_dropimpact.pack(pady=10, fill="x")
    Tooltip(btn_dropimpact, "Calculate the velocity and spreading factor of a droplet during impact.")

    btn_info = ctk.CTkButton(
        app.menu_container,
        text="Important Information",
        fg_color="#A93226",
        hover_color="#7B241C",
        command=app.show_important_info,
    )
    btn_info.pack(padx=20, pady=20)
    Tooltip(btn_info, "Important information about analyses and files.")


def show_sub_menu(app, mode):
    app.current_mode = mode
    for child in app.menu_container.winfo_children():
        child.destroy()

    titles = {
        "WCA": ["Tangent method", "Polynomial Fit", "Three points"],
        "HIS": ["Adv-Rec", "Polinomial", "Weighted Least Squares"],
        "DROP": ["First method", "Second method", "Third method", "Four method"],
    }

    ctk.CTkButton(app.menu_container, text="← Return", fg_color="#444", command=lambda: create_main_menu(app)).pack(pady=(0, 10), fill="x")

    for method in titles[mode]:
        if method == "Tangent method":
            btn_normal = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_static(app), app.run_analise_normal()])
            btn_normal.pack(pady=5, fill="x")
            Tooltip(btn_normal, "To measure the equilibrium contact angle of a stationary droplet on a horizontal surface.")
        elif method == "Polynomial Fit":
            btn_fitting = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_static_pol(app), app.run_analise_avancada()])
            btn_fitting.pack(pady=5, fill="x")
            Tooltip(btn_fitting, r"This method uses a second-degree equation to automatically model the droplet's curvature and calculate the tangent at the contact point")
        elif method == "Three points":
            btn_points = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_static_pol(app), app.run_analise_manual()])
            btn_points.pack(pady=5, fill="x")
            Tooltip(btn_points, r"This manual approach defines the contact angle by connecting three user-selected points to form the base and the slope of the droplet.")
        elif method == "Adv-Rec":
            btn_adv_rec = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_histerese(app), app.run_histerese_avancante()])
            btn_adv_rec.pack(pady=5, fill="x")
            Tooltip(btn_adv_rec, r"To measure the Advancing Angle ($\theta_A$) and the Receding Angle ($\theta_R$)")
        elif method == "Polinomial":
            btn_polinomial = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_histerese(app), app.run_histerese_polinomial()])
            btn_polinomial.pack(pady=5, fill="x")
            Tooltip(btn_polinomial, r"To measure the Advancing Angle ($\theta_A$) and the Receding Angle ($\theta_R$)")
        elif method == "Weighted Least Squares":
            btn_run_wls = ctk.CTkButton(
                app.menu_container,
                text=method,
                command=lambda app_ref=app: app_ref.run_histerese_wls(),
            )
            btn_run_wls.pack(pady=5, fill="x")
            Tooltip(btn_run_wls, "Calculate the contact angle using polynomials with Weighted Least Squares (WLS).")
        elif method == "First method":
            btn_first = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_impact(app), app.run_analise_velocidade()])
            btn_first.pack(pady=5, fill="x")
            Tooltip(btn_first, "Performs impact detection based on point filtering at the surface level, providing a simplified tracking of the drop's vertical approach")
        elif method == "Second method":
            btn_second = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_impact(app), app.run_velocity_analysis()])
            btn_second.pack(pady=5, fill="x")
            Tooltip(btn_second, "Utilizes advanced morphological segmentation and convex hull geometry to accurately measure the maximum spreading factor and expansion dynamics.")
        elif method == "Third method":
            btn_third = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_impact(app), app.run_analise_impacto_tres()])
            btn_third.pack(pady=5, fill="x")
            Tooltip(btn_third, "test.")
        elif method == "Four method":
            btn_four = ctk.CTkButton(app.menu_container, text=method, command=lambda: [mostrar_ajustes_impact(app), app.run_drop_impact_analysis()])
            btn_four.pack(pady=5, fill="x")
            Tooltip(btn_four, "This method analyzes droplet impact by measuring the droplet diameter at the baseline during spreading using image segmentation.")
        else:
            ctk.CTkButton(app.menu_container, text=method, command=app.load_media).pack(pady=5, fill="x")


def mostrar_ajustes_histerese(app):
    """Create the hysteresis threshold adjustment panel."""
    if app.ajustes_container:
        app.ajustes_container.destroy()

    app.ajustes_container = ctk.CTkFrame(app.sidebar, fg_color="transparent")
    app.ajustes_container.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(app.ajustes_container, text="Parameter adjustment", font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))
    ctk.CTkLabel(app.ajustes_container, text="Threshold:").pack(anchor="w")

    app.lbl_thresh_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.threshold_histerese}")
    app.lbl_thresh_val.pack(anchor="e")

    app.slider_thresh = ctk.CTkSlider(app.ajustes_container, from_=0, to=255, command=lambda val, app_ref=app: update_thresh_value(app_ref, val))
    app.slider_thresh.set(app.threshold_histerese)
    app.slider_thresh.pack(fill="x", pady=5)

    ctk.CTkLabel(app.ajustes_container, text="Tangent Points:").pack(anchor="w", pady=(10, 0))
    app.lbl_tangente_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.n_pontos_tangente_val}")
    app.lbl_tangente_val.pack(anchor="e")
    app.slider_tangente = ctk.CTkSlider(app.ajustes_container, from_=5, to=100, command=lambda val, app_ref=app: update_tangente_value(app_ref, val))
    app.slider_tangente.set(app.n_pontos_tangente_val)
    app.slider_tangente.pack(fill="x", pady=5)

    ctk.CTkLabel(app.ajustes_container, text="Minimum Fit Points:").pack(anchor="w", pady=(10, 0))
    app.lbl_min_pontos_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.min_pontos_fit}")
    app.lbl_min_pontos_val.pack(anchor="e")
    app.slider_min_pontos = ctk.CTkSlider(app.ajustes_container, from_=4, to=100, command=lambda val, app_ref=app: update_min_pontos_fit(app_ref, val))
    app.slider_min_pontos.set(app.min_pontos_fit)
    app.slider_min_pontos.pack(fill="x", pady=5)


def update_thresh_value(app, val):
    app.threshold_histerese = int(val)
    if hasattr(app, 'lbl_thresh_val'):
        app.lbl_thresh_val.configure(text=f"{app.threshold_histerese}")


def update_tangente_value(app, val):
    app.n_pontos_tangente_val = int(val)
    if hasattr(app, 'lbl_tangente_val'):
        app.lbl_tangente_val.configure(text=f"{app.n_pontos_tangente_val}")


def update_min_pontos_fit(app, val):
    app.min_pontos_fit = int(val)
    if hasattr(app, 'lbl_min_pontos_val'):
        app.lbl_min_pontos_val.configure(text=f"{app.min_pontos_fit}")


def mostrar_ajustes_impact(app):
    """Create the impact threshold adjustment panel."""
    if app.ajustes_container:
        app.ajustes_container.destroy()

    app.ajustes_container = ctk.CTkFrame(app.sidebar, fg_color="transparent")
    app.ajustes_container.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(app.ajustes_container, text="Parameter adjustment", font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))
    ctk.CTkLabel(app.ajustes_container, text="Threshold:").pack(anchor="w")
    app.lbl_thresh_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.threshold_impact}")
    app.lbl_thresh_val.pack(anchor="e")
    app.slider_thresh = ctk.CTkSlider(app.ajustes_container, from_=0, to=255, command=lambda val: update_thresh_value_impact(app, val))
    app.slider_thresh.set(app.threshold_impact)
    app.slider_thresh.pack(fill="x", pady=5)


def update_thresh_value_impact(app, val):
    app.threshold_impact = int(val)
    if hasattr(app, 'lbl_thresh_val'):
        app.lbl_thresh_val.configure(text=f"{app.threshold_impact}")


def mostrar_ajustes_static(app):
    """Create the static-angle tangent panel."""
    if app.ajustes_container:
        app.ajustes_container.destroy()

    app.ajustes_container = ctk.CTkFrame(app.sidebar, fg_color="transparent")
    app.ajustes_container.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(app.ajustes_container, text="Parameter adjustment", font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))
    ctk.CTkLabel(app.ajustes_container, text="Number of points:").pack(anchor="w")
    app.lbl_thresh_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.tangent_static}")
    app.lbl_thresh_val.pack(anchor="e")
    app.slider_thresh = ctk.CTkSlider(app.ajustes_container, from_=0, to=50, command=lambda val: update_thresh_value_static(app, val))
    app.slider_thresh.set(app.tangent_static)
    app.slider_thresh.pack(fill="x", pady=5)


def update_thresh_value_static(app, val):
    app.tangent_static = int(val)
    if hasattr(app, 'lbl_thresh_val'):
        app.lbl_thresh_val.configure(text=f"{app.tangent_static}")


def mostrar_ajustes_static_pol(app):
    """Create the polynomial static-angle adjustment panel."""
    if app.ajustes_container:
        app.ajustes_container.destroy()

    app.ajustes_container = ctk.CTkFrame(app.sidebar, fg_color="transparent")
    app.ajustes_container.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(app.ajustes_container, text="Parameter adjustment", font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))
    ctk.CTkLabel(app.ajustes_container, text="Number of points:").pack(anchor="w")
    app.lbl_static_pol_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.tangent_static_pol}")
    app.lbl_static_pol_val.pack(anchor="e")
    app.slider_static_pol = ctk.CTkSlider(app.ajustes_container, from_=0, to=50, command=lambda val, app_ref=app: update_thresh_value_static_pol(app_ref, val))
    app.slider_static_pol.set(app.tangent_static_pol)
    app.slider_static_pol.pack(fill="x", pady=5)

    ctk.CTkLabel(app.ajustes_container, text="Number of points base:").pack(anchor="w")
    app.lbl_static_base_val = ctk.CTkLabel(app.ajustes_container, text=f"{app.tangent_static_base_points}")
    app.lbl_static_base_val.pack(anchor="e")
    app.slider_static_base = ctk.CTkSlider(app.ajustes_container, from_=0, to=50, command=lambda val, app_ref=app: update_thresh_value_static_base_points(app_ref, val))
    app.slider_static_base.set(app.tangent_static_base_points)
    app.slider_static_base.pack(fill="x", pady=5)


def update_thresh_value_static_pol(app, val):
    app.tangent_static_pol = int(val)
    if hasattr(app, 'lbl_static_pol_val'):
        app.lbl_static_pol_val.configure(text=f"{app.tangent_static_pol}")


def update_thresh_value_static_base_points(app, val):
    app.tangent_static_base_points = int(val)
    if hasattr(app, 'lbl_static_base_val'):
        app.lbl_static_base_val.configure(text=f"{app.tangent_static_base_points}")
