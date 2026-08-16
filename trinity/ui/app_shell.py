import customtkinter as ctk
from pathlib import Path
from tkinter import ttk

from trinity.ui.tooltip import Tooltip

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - Pillow is optional for the logo fallback
    Image = None
    ImageTk = None


def build_main_ui(app):
    """Initialize the main application layout and widgets."""
    app.main_layout = ctk.CTkFrame(app, fg_color="transparent")
    app.main_layout.pack(fill="both", expand=True, padx=10, pady=10)

    app.sidebar = ctk.CTkFrame(app.main_layout, width=300, corner_radius=12, fg_color="#1d1d1d")
    app.sidebar.pack(side="left", fill="y", padx=(0, 10))

    app.logo_container = ctk.CTkFrame(app.sidebar, fg_color="#161616", corner_radius=12)
    app.logo_container.pack(fill="x", padx=10, pady=(10, 8))

    logo_path = Path(__file__).resolve().parents[2] / "vids" / "test.png"
    if logo_path.exists() and Image is not None and ImageTk is not None:
        try:
            image = Image.open(logo_path).resize((220, 90), Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
            app.logo_image = ImageTk.PhotoImage(image)
            app.logo_label = ctk.CTkLabel(
                app.logo_container,
                image=app.logo_image,
                text="",
                width=220,
                height=90,
            )
        except Exception:
            app.logo_label = ctk.CTkLabel(
                app.logo_container,
                text="TRINITY",
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color="#dfeafc",
                anchor="center",
                height=60,
            )
    else:
        app.logo_label = ctk.CTkLabel(
            app.logo_container,
            text="TRINITY",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#dfeafc",
            anchor="center",
            height=60,
        )

    app.logo_label.pack(fill="x", padx=10, pady=10)

    app.menu_container = ctk.CTkFrame(app.sidebar, fg_color="transparent")
    app.menu_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    app.content = ctk.CTkFrame(app.main_layout, fg_color="transparent")
    app.content.pack(side="left", fill="both", expand=True)

    app.top_toolbar = ctk.CTkFrame(app.content, height=60, fg_color="#181818", corner_radius=10)
    app.top_toolbar.pack(fill="x", pady=(0, 10))

    app.view_container = ctk.CTkFrame(app.content, fg_color="transparent")
    app.view_container.pack(fill="both", expand=True)

    btn_import = ctk.CTkButton(app.top_toolbar, text="Import", width=100, command=app.load_media)
    btn_import.pack(side="left", padx=5, pady=5)
    Tooltip(btn_import, "Load image or video files from your computer for analysis.")
    btn_Edge = ctk.CTkButton(app.top_toolbar, text="Edge detection", width=100, command=app.edge_detection)
    btn_Edge.pack(side="left", padx=10, pady=5)
    Tooltip(btn_Edge, "Toggle the silhouette detection to highlight the droplet's boundaries.")
    btn_fitting = ctk.CTkButton(app.top_toolbar, text="Fitting", width=100, fg_color="#28a745", command=app.fitting)
    btn_fitting.pack(side="left", padx=5, pady=5)
    Tooltip(btn_fitting, "Apply a circular regression to calculate the contact angle within a selected area.")
    btn_analyze = ctk.CTkButton(app.top_toolbar, text="Analyze", width=100, command=app.show_analyze_options)
    btn_analyze.pack(side="left", padx=10, pady=5)
    Tooltip(btn_analyze, "Open the menu to generate evolution graphs and statistical distributions.")
    btn_datatable = ctk.CTkButton(app.top_toolbar, text="Data Table", width=100, command=app.show_data_table)
    btn_datatable.pack(side="left", padx=10, pady=5)
    Tooltip(btn_datatable, "View and manage the numerical results of your measurements in a spreadsheet format.")
    btn_save = ctk.CTkButton(app.top_toolbar, text="Save Data", width=100, command=app.save_data_table)
    btn_save.pack(side="left", padx=5, pady=5)
    btn_save_img = ctk.CTkButton(app.top_toolbar, text="Save Image", width=100, command=app.save_current_image)
    btn_save_img.pack(side="left", padx=5, pady=5)
    Tooltip(btn_save, "Export the current measurement table to an external file.")
    btn_reset = ctk.CTkButton(app.top_toolbar, text="Reset", width=100, fg_color="#6c757d", command=app.reset_analysis)
    btn_reset.pack(side="left", padx=5, pady=5)
    Tooltip(btn_reset, "Clear all current progress and return the application to its initial state.")

    app.center_stack = ctk.CTkFrame(app.view_container, fg_color="transparent")
    app.center_stack.pack(fill="both", expand=True, padx=15, pady=15)

    app.canvas_view = ctk.CTkLabel(
        app.center_stack,
        text="Awaiting media...",
        fg_color="#0a0a0a",
        corner_radius=10,
    )
    app.canvas_view.pack(fill="both", expand=True)

    app.table_container = ctk.CTkFrame(app.center_stack, fg_color="#1a1a1a")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
    style.map("Treeview", background=[('selected', '#1f538d')])

    app.tree = ttk.Treeview(app.table_container, columns=(), show="headings")
    app.tree.pack(side="left", fill="both", expand=True)

    app.scrollbar = ttk.Scrollbar(app.table_container, orient="vertical", command=app.tree.yview)
    app.tree.configure(yscrollcommand=app.scrollbar.set)
    app.scrollbar.pack(side="right", fill="y")

    app.canvas_view.bind("<MouseWheel>", app.on_mouse_wheel)
    app.canvas_view.bind("<Button-1>", app.on_click_press)
    app.canvas_view.bind("<B1-Motion>", app.on_mouse_drag)
    app.canvas_view.bind("<ButtonRelease-1>", app.on_click_release)
    app.canvas_view.bind("<MouseWheel>", app.on_mouse_scroll_windows)
    app.canvas_view.bind("<Motion>", app.on_mouse_move)
    app.canvas_view.bind("<Leave>", app.hide_zoom_window)

    app.controls_panel = ctk.CTkFrame(app.view_container, height=100, fg_color="#1a1a1a", corner_radius=10)
    app.controls_panel.pack(fill="x", side="bottom", padx=15, pady=(0, 15))

    app.btns_wrapper = ctk.CTkFrame(app.controls_panel, fg_color="transparent")
    app.btns_wrapper.pack(expand=True)

    app.btn_prev = ctk.CTkButton(app.btns_wrapper, text="⏪", width=40, height=20, command=app.prev_frame)
    app.btn_prev.grid(row=0, column=0, padx=5, pady=10)
    app.btn_play = ctk.CTkButton(app.btns_wrapper, text="START", width=90, height=20, command=app.toggle_video)
    app.btn_play.grid(row=0, column=1, padx=5, pady=10)
    app.btn_next = ctk.CTkButton(app.btns_wrapper, text="⏩", width=40, height=20, command=app.next_frame)
    app.btn_next.grid(row=0, column=2, padx=5, pady=10)

    ctk.CTkLabel(app.btns_wrapper, text=" | ").grid(row=0, column=3, padx=10)

    app.btn_start_an = ctk.CTkButton(
        app.btns_wrapper,
        text="START ANALYSIS",
        width=40,
        height=20,
        fg_color="#28a745",
        hover_color="#218838",
        command=app.start_analysis,
    )
    app.btn_start_an.grid(row=0, column=4, padx=5, pady=10)

    app.btn_pause_an = ctk.CTkButton(
        app.btns_wrapper,
        text="PAUSE ANALYSIS",
        width=40,
        height=20,
        fg_color="#dc3545",
        hover_color="#c82333",
        command=app.pause_analysis,
    )
    app.btn_pause_an.grid(row=0, column=5, padx=5, pady=10)

    app.btn_reset_an = ctk.CTkButton(
        app.btns_wrapper,
        text="RESET",
        width=40,
        height=20,
        fg_color="#6c757d",
        hover_color="#5a6268",
        command=app.reset_all,
    )
    app.btn_reset_an.grid(row=0, column=6, padx=5, pady=10)

    app.btn_restart_an = ctk.CTkButton(
        app.btns_wrapper,
        text="RESTART",
        width=40,
        height=20,
        fg_color="#6c757d",
        hover_color="#5a6268",
        command=app.restart_video,
    )
    app.btn_restart_an.grid(row=0, column=7, padx=5, pady=10)

    app.btn_zoom_in = ctk.CTkButton(app.btns_wrapper, text="🔍+", width=40, height=20, command=lambda: app.apply_zoom(1.2))
    app.btn_zoom_in.grid(row=0, column=8, padx=2)
    app.btn_zoom_out = ctk.CTkButton(app.btns_wrapper, text="🔍-", width=40, height=20, command=lambda: app.apply_zoom(0.8))
    app.btn_zoom_out.grid(row=0, column=9, padx=2)

    app.video_slider_var = app.video_slider_var if hasattr(app, 'video_slider_var') else None
    if app.video_slider_var is None:
        import tkinter as tk
        app.video_slider_var = tk.DoubleVar(value=0)

    app.video_slider = ctk.CTkSlider(
        app.btns_wrapper,
        from_=0,
        to=100,
        variable=app.video_slider_var,
        command=app.on_slider_move,
    )
    app.video_slider.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(5, 5), padx=2)

    app.lbl_status = ctk.CTkLabel(app.view_container, text="Waiting for file to load...")
    app.lbl_status.pack(side="bottom")

    app.create_main_menu()
    app.show_home_view()
