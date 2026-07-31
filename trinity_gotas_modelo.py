# =======================================================================================================================
# PROJECT TITLE: TRINITY - AUTOMATED WETTABILITY AND CONTACT ANGLE ANALYSIS SYSTEM
# =======================================================================================================================
# Authors: Lucas Luís Bergonzi Barcellos, Carlo Antonini, Marcos José Leite Santos.
# Institution: Federal University of Rio Grande do Sul
# Department: [Nome do Departamento - ex: Engenharia Mecânica / Física / Química]
# Contact: [seu_email@instituicao.org]
# Date: June 2026
# Version: 1.0.0
# License: [Open Source / MIT / Proprietary - Conforme a política do seu laboratório]
#
# -----------------------------------------------------------------------------------------------------------------------
# SCIENTIFIC ABSTRACT & PURPOSE:
# This software provides a high-throughput computational framework for automated Contact Angle (CA) and hysteresis 
# measurements in surface wettability characterization. By processing static images and high-speed video streams, 
# the algorithm evaluates droplet geometry, base diameter, and dynamic contact angles (advancing and receding) 
# to quantify solid-liquid-gas interfacial properties under diverse experimental conditions.
#
# -----------------------------------------------------------------------------------------------------------------------
# METHODOLOGICAL REFERENCES & ALGORITHMIC FOUNDATIONS:
# The mathematical and computer vision core of this architecture is built upon the following established scientific works:
#
# 1. Edge Detection & Contour Extraction:
#    - Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine 
#      Intelligence, PAMI-8(6), 679-698.
#    - Suzuki, S., & Abe, K. (1985). Topological structural analysis of digitized binary images by border following. 
#      Computer Vision, Graphics, and Image Processing, 30(1), 32-46.
#
# 2. Polynomial Fitting & Local Gradient Analysis (DroPy Logic Variant):
#    - Bateni, A., et al. (2003). Complex optimization of contact angle measurement using polynomial fitting. 
#      Journal of Colloid and Interface Science.
#    - Incorporates specialized geometric formulations derived from standard Axisymmetric Drop Shape Analysis (ADSA).
#
# 3. Weighted Least Squares (WLS) Optimization for Contact Angles:
#    - Cleveland, W. S. (1979). Robust Locally Weighted Regression and Smoothing Scatterplots. Journal of the American 
#      Statistical Association, 74(368), 829-836. (Adapted here for spatial point proximity weighting near the three-phase contact line).
#
# 4. Signal Smoothing (Dynamic Video Analysis):
#    - Savitzky, A., & Golay, M. J. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. 
#      Analytical Chemistry, 36(8), 1627-1639. (Implemented via Savitzky-Golay filtering for temporal hysteresis curves).
#
# -----------------------------------------------------------------------------------------------------------------------
# AI-ASSISTED PROGRAMMING DECLARATION:
# In compliance with international publishing standards on research integrity (e.g., COPE, Elsevier, and IEEE guidelines 
# regarding generative AI tools):
# Large Language Models (LLMs), specifically Google Gemini, were utilized as an assistive technology during the 
# development of this source code. The AI was strictly employed for structural code refactoring, translation of 
# internal user interfaces from Portuguese to English, optimization of graphical layout rendering via CustomTkinter, 
# and implementation of documentation headers. 
# All scientific logic, mathematical equations, data interpretation routines, and final code validations were exclusively 
# designed, reviewed, and approved by the human authors.
# =======================================================================================================================


import cv2
import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.integrate as spint
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import math
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from trinity.analysis.plot_service import render_extracted_plot
from trinity.ui.tooltip import Tooltip
from trinity.utils.contact_angles import (
    angulo_interno_base,
    angulo_interno_base_polinomial,
    angulo_interno_base_wls,
    calcular_angulo_wls,
)
from trinity.utils.drawing import desenhar_arco_angulo, desenhar_tangentes
from trinity.utils.geometry import (
    calcular_angulo_dropy,
    calcular_diametro,
    pontos_proximos_base,
    selecionar_ponto_callback,
    separar_lados,
)
# Appearance Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
class TrinityApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dropen3 - Wettability Analysis")
        self.geometry("1280x900")

        self.cap = None
        self.video_playing = False
        self.analysis_active = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.after_id = None
       # self.current_mode = "WCA" # Modo padrão inicial
        self.show_edges = False
        # Variáveis de Zoom para Análise Estática
        self.zoom_factor = 1.0
        self.img_result_original = None
        self.roi_zoom = None 
        self.dragging = False
        self.start_x = 0
        self.start_y = 0

        # No __init__:
        self.zoom_window_size = 150  # Tamanho da janelinha de zoom (quadrada)
        self.magnification = 5.0     # Nível de ampliação da lupa
        self.zoom_label = None       # Widget da lupa
        self.ultimo_metodo_selecionado = None
        self._syncing_slider = False
        # Variáveis para a nova análise de Velocidade
        self.cap_vel = None
        self.vel_active = False
        
        self.setup_ui()
        # No final do __init__ (após self.setup_ui())
        self.threshold_histerese = 200  # Valor padrão
        self.threshold_impact = 30  # Valor padrão
        self.tangent_static = 40  # Valor padrão
        self.tangent_static_pol = 15
        self.tangent_static_base_points = 20
        self.n_pontos_tangente_val = 15  # Novo valor padrão
        self.min_pontos_fit = 4  # Valor inicial padrão para a trava de segurança
        self.ajustes_container = None    # Referência para o frame dinâmico
        # AGORA você desativa a propagação
        self.view_container.grid_propagate(False)
        self.view_container.pack_propagate(False)
    def setup_ui(self):
        self.menu_bar = ctk.CTkFrame(self, height=25, corner_radius=0)
        self.menu_bar.pack(side="top", fill="x")
        for item in ["Help"]:
            ctk.CTkButton(self.menu_bar, text=item, width=80, fg_color="transparent").pack(side="right", padx=5)

        self.main_layout = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_layout.pack(fill="both", expand=True)

        self.sidebar = ctk.CTkFrame(self.main_layout, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        # --- NOVO BLOCO PARA INSERIR LOGO ---
        try:
            img_logo_path = os.path.join(os.path.dirname(__file__), "logo-1.png")
            if not os.path.exists(img_logo_path):
                img_logo_path = None

            if img_logo_path is None:
                raise FileNotFoundError("logo-1.png")

            img_logo_pil = Image.open(img_logo_path)
            # Ajusta o size=(largura, altura) conforme necessário
            self.logo_image = ctk.CTkImage(light_image=img_logo_pil, 
                                           dark_image=img_logo_pil, 
                                           size=(100, 100))
            
            self.logo_label = ctk.CTkLabel(self.sidebar, image=self.logo_image, text="")
            self.logo_label.pack(pady=(0, 0)) # Espaçamento superior
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                print(f"Error loading logo: {e}")
        # ------------------------------------

        # O label do título já existente (ajustei o pady para ficar logo abaixo da imagem)
        ctk.CTkLabel(self.sidebar, text="Dropen3", font=("Roboto", 20, "bold")).pack(pady=(10, 30))
        #ctk.CTkLabel(self.sidebar, text="TRINITY", font=("Roboto", 24, "bold")).pack(pady=30)
        self.menu_container = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent", width=200)
        self.menu_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.create_main_menu()

        self.view_container = ctk.CTkFrame(self.main_layout, corner_radius=15)
        self.view_container.pack(side="right", fill="both", expand=True, padx=20, pady=20)
            
        # --- NOVO MENU SUPERIOR ---
        self.top_toolbar = ctk.CTkFrame(self.view_container, height=40, fg_color="#222", corner_radius=10)
        self.top_toolbar.pack(fill="x", padx=15, pady=(15, 0))
        
        btn_home = ctk.CTkButton(self.top_toolbar, text="Home", width=80, fg_color="#1f538d", command=self.show_home_view)
        btn_home.pack(side="left", padx=5, pady=5)
        Tooltip(btn_home, "Return to the main dashboard and reset the current view.")
        #ctk.CTkButton(self.top_toolbar, text="Import", width=100, command=self.load_media).pack(side="left", padx=5, pady=5)
        btn_import = ctk.CTkButton(self.top_toolbar, text="Import", width=100, command=self.load_media)
        btn_import.pack(side="left", padx=5, pady=5)
        Tooltip(btn_import, "Load image or video files from your computer for analysis.") # Adicione esta linha
        btn_Edge = ctk.CTkButton(self.top_toolbar, text="Edge detection", width=100, command=self.edge_detection)
        btn_Edge.pack(side="left", padx=10, pady=5)
        Tooltip(btn_Edge, "Toggle the silhouette detection to highlight the droplet's boundaries.")
        btn_fitting = ctk.CTkButton(self.top_toolbar, text="Fitting", width=100, fg_color="#28a745", command=self.fitting)
        btn_fitting.pack(side="left", padx=5, pady=5)
        Tooltip(btn_fitting, "Apply a circular regression to calculate the contact angle within a selected area.")
        btn_analyze = ctk.CTkButton(self.top_toolbar, text="Analyze", width=100, command=self.show_analyze_options)
        btn_analyze.pack(side="left", padx=10, pady=5)
        Tooltip(btn_analyze, "Open the menu to generate evolution graphs and statistical distributions.")
        btn_datatable = ctk.CTkButton(self.top_toolbar, text="Data Table", width=100, command=self.show_data_table)
        btn_datatable.pack(side="left", padx=10, pady=5)
        Tooltip(btn_datatable, "View and manage the numerical results of your measurements in a spreadsheet format.")
        btn_save = ctk.CTkButton(self.top_toolbar, text="Save Data", width=100, command=self.save_data_table)
        btn_save.pack(side="left", padx=5, pady=5)
        btn_save_img = ctk.CTkButton(self.top_toolbar, text="Save Image", width=100, command=self.save_current_image)
        btn_save_img.pack(side="left", padx=5, pady=5)
        Tooltip(btn_save, "Export the current measurement table to an external file.")
        btn_reset = ctk.CTkButton(self.top_toolbar, text="Reset", width=100, fg_color="#6c757d", command=self.reset_analysis)
        btn_reset.pack(side="left", padx=5, pady=5)
        Tooltip(btn_reset, "Clear all current progress and return the application to its initial state.")

        # Área Central - Stack de visualização (Canvas e Tabela)
        self.center_stack = ctk.CTkFrame(self.view_container, fg_color="transparent")
        self.center_stack.pack(fill="both", expand=True, padx=15, pady=15)

        self.canvas_view = ctk.CTkLabel(self.center_stack, text="Awaiting media...", 
                                       fg_color="#0a0a0a", corner_radius=10)
        self.canvas_view.pack(fill="both", expand=True)

        # Widget de Tabela (Oculto inicialmente)
        self.table_container = ctk.CTkFrame(self.center_stack, fg_color="#1a1a1a")
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map("Treeview", background=[('selected', '#1f538d')])
        
        self.tree = ttk.Treeview(self.table_container, columns=(), show="headings")
        self.tree.pack(side="left", fill="both", expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")

        # Bindings para Zoom
        self.canvas_view.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas_view.bind("<Button-1>", self.on_click_press)
        self.canvas_view.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas_view.bind("<ButtonRelease-1>", self.on_click_release)
        
        # Adicione esta linha para escutar o scroll no Windows:
        self.canvas_view.bind("<MouseWheel>", self.on_mouse_scroll_windows)
        
        self.canvas_view.bind("<Motion>", self.on_mouse_move)
        self.canvas_view.bind("<Leave>", self.hide_zoom_window) # Esconde quando o mouse sai
        
        self.controls_panel = ctk.CTkFrame(self.view_container, height=100, fg_color="#1a1a1a", corner_radius=10)
        self.controls_panel.pack(fill="x", side="bottom", padx=15, pady=(0, 15))

        self.btns_wrapper = ctk.CTkFrame(self.controls_panel, fg_color="transparent")
        self.btns_wrapper.pack(expand=True)

        self.btn_prev = ctk.CTkButton(self.btns_wrapper, text="⏪", width=40, height= 20, command=self.prev_frame)
        self.btn_prev.grid(row=0, column=0, padx=5, pady=10)
        self.btn_play = ctk.CTkButton(self.btns_wrapper, text="START", width=90, height= 20, command=self.toggle_video)
        self.btn_play.grid(row=0, column=1, padx=5, pady=10)
        self.btn_next = ctk.CTkButton(self.btns_wrapper, text="⏩", width=40, height= 20, command=self.next_frame)
        self.btn_next.grid(row=0, column=2, padx=5, pady=10)

        ctk.CTkLabel(self.btns_wrapper, text=" | ").grid(row=0, column=3, padx=10)

        self.btn_start_an = ctk.CTkButton(self.btns_wrapper, text="START ANALYSIS", width=40, height= 20, 
                                         fg_color="#28a745", hover_color="#218838", 
                                         command=self.start_analysis)
        self.btn_start_an.grid(row=0, column=4, padx=5, pady=10)
        
        self.btn_pause_an = ctk.CTkButton(self.btns_wrapper, text="PAUSE ANALYSIS", width=40, height= 20, 
                                         fg_color="#dc3545", hover_color="#c82333", 
                                         command=self.pause_analysis)
        self.btn_pause_an.grid(row=0, column=5, padx=5, pady=10)
        
        self.btn_reset_an = ctk.CTkButton(self.btns_wrapper, text="RESET", width=40, height= 20, 
                                         fg_color="#6c757d", hover_color="#5a6268", 
                                         command=self.reset_all)
        self.btn_reset_an.grid(row=0, column=6, padx=5, pady=10)
        
        self.btn_restart_an = ctk.CTkButton(self.btns_wrapper, text="RESTART", width=40, height= 20, 
                                         fg_color="#6c757d", hover_color="#5a6268", 
                                         command=self.restart_video)
        self.btn_restart_an.grid(row=0, column=7, padx=5, pady=10)
        
        

        self.btn_zoom_in = ctk.CTkButton(self.btns_wrapper, text="🔍+", width=40, height= 20, command=lambda: self.apply_zoom(1.2))
        self.btn_zoom_in.grid(row=0, column=8, padx=2)
        self.btn_zoom_out = ctk.CTkButton(self.btns_wrapper, text="🔍-", width=40, height= 20, command=lambda: self.apply_zoom(0.8))
        self.btn_zoom_out.grid(row=0, column=9, padx=2)
        # --- NOVA ADIÇÃO: O SLIDER INTEGRADO LOGO ABAIXO NA LINHA 1 (row=1) ---
        self.video_slider_var = tk.DoubleVar(value=0)
        self.video_slider = ctk.CTkSlider(
            self.btns_wrapper, 
            from_=0, 
            to=100, # Valor genérico inicial, será reajustado ao carregar a mídia
            variable=self.video_slider_var,
            command=self.on_slider_move
        )
        # O slider ocupa toda a largura abaixo dos botões (columnspan engloba todas as colunas)
        self.video_slider.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(5, 5), padx=2)

        self.lbl_status = ctk.CTkLabel(self.view_container, text="Waiting for file to load...")
        self.lbl_status.pack(side="bottom")
    
    def edge_detection(self):
        """
        Aplica Canny e findContours para destacar a silhueta da gota.
        """
        if self.img_result_original is None:
            messagebox.showwarning("Warning", "Upload an image first!")
            return
        self.show_edges = not self.show_edges
        # Converter para escala de cinza
        gray = cv2.cvtColor(self.img_result_original, cv2.COLOR_BGR2GRAY)
        
        # Suavização para reduzir ruído
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detecção de bordas adaptativa (usando a mediana para os limiares)
        v = np.median(blur)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edged = cv2.Canny(blur, lower, upper)

        # Dilatação para fechar possíveis gaps no contorno
        kernel = np.ones((3,3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)

        # Converter para BGR para exibir colorido (contorno verde)
        result_vis = self.img_result_original.copy()
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            c_max = max(contours, key=cv2.contourArea)
            cv2.drawContours(result_vis, [c_max], -1, (0, 255, 0), 2)
            self.current_contour = c_max # Armazena para o Fitting
            self.display_frame(result_vis, update_original=False)
            self.lbl_status.configure(text="Edges successfully detected.")
        else:
            self.lbl_status.configure(text="No outline found.")
            self.display_frame(self.img_result_original, update_original=False)
            self.lbl_status.configure(text="Border Preview: DISABLED")
    def show_important_info(self):
        """Exibe orientações importantes diretamente no canvas do programa."""
        if self.img_result_original is None:
            # Cria um fundo preto genérico caso não haja imagem carregada
            info_bg = np.zeros((600, 800, 3), dtype=np.uint8)
        else:
            # Cria uma sobreposição escura sobre a imagem atual
            info_bg = self.img_result_original.copy()
            info_bg = cv2.addWeighted(info_bg, 0.2, info_bg, 0, 0)

        texto_ajuda = [
            "IMPORTANT INFORMATION:",
            "- All videos should start without the drop appearing.",
            "- Select the method that best suits your videos and images.",
            "- In some analyses, it is necessary to select the area that",
            "  includes the droplet (ROI).",
            "- The Baseline must be selected horizontally in the interface.",
            "- Adjust the parameters if you notice any errors in the analysis.",
            "",
            "Clique em 'Home' ou 'Import' para retornar."
        ]

        # Desenha o texto na imagem
        y0, dy = 50, 40
        for i, linha in enumerate(texto_ajuda):
            y = y0 + i * dy
            cv2.putText(info_bg, linha, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

        self.display_frame(info_bg)
    def save_current_image(self):
        """Salva a imagem que está sendo exibida no momento no canvas."""
        if self.img_result_original is None:
            messagebox.showwarning("Warning", "There is no image to save!")
            return

        # Se houver uma imagem com marcações sendo exibida, usamos ela, caso contrário a original
        img_to_save = getattr(self, 'current_vis_img', self.img_result_original)

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Measurement Image"
        )

        if file_path:
            try:
                # O imwrite do OpenCV espera BGR, que é o padrão usado no seu programa
                cv2.imwrite(file_path, img_to_save)
                messagebox.showinfo("Success", f"Image saved in:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
    def fitting(self):
        """
        Realiza o ajuste circular permitindo que o usuário selecione um ROI
        para isolar a gota de interferências como a agulha.
        """
        if self.img_result_original is None:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Upload an image first!")
            return

        # 1. Seleção do ROI via OpenCV
        # Abre uma janela para selecionar a área; aperte ENTER ou ESPAÇO para confirmar
        roi = cv2.selectROI("Select the Drop (Press ENTER)", self.img_result_original, fromCenter=False)
        cv2.destroyWindow("Select the Drop (Press ENTER)")
        
        x_roi, y_roi, w_roi, h_roi = roi
        
        # Se o usuário cancelar ou não selecionar nada, interrompe
        if w_roi == 0 or h_roi == 0:
            return

        # 2. Recorte e Processamento apenas no ROI
        # Criamos uma cópia apenas da área selecionada
        roi_img = self.img_result_original[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            self.lbl_status.configure(text="Error: No outline in ROI.")
            return
            
        cnt = max(contours, key=cv2.contourArea)
        points = cnt.reshape(-1, 2)
        
        # Ajustamos os pontos para as coordenadas globais da imagem original
        x_data = (points[:, 0] + x_roi).astype(float)
        y_data = (points[:, 1] + y_roi).astype(float)

        # 3. Ajuste Circular (Mínimos Quadrados)
        x_m, y_m = np.mean(x_data), np.mean(y_data)
        
        def calc_R(xc, yc):
            return np.sqrt((x_data - xc)**2 + (y_data - yc)**2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center_2, _ = spopt.leastsq(f_2, center_estimate)
        xc, yc = center_2
        R = calc_R(*center_2).mean()

        # 4. Definição da Baseline
        y_base = np.max(y_data)
        x_min, x_max = np.min(x_data), np.max(x_data)
        
        xbas1, ybas1 = x_min - xc, y_base - yc
        xbas2, ybas2 = x_max - xc, y_base - yc
        dx, dy = xbas2 - xbas1, ybas2 - ybas1
        dr = np.sqrt(dx**2 + dy**2)
        D = xbas1*ybas2 - xbas2*ybas1
        
        discriminant = R**2 * dr**2 - D**2
        if discriminant < 0:
            self.lbl_status.configure(text="Error: Invalid intersection in ROI.")
            return

        delta_root = np.sqrt(discriminant)
        sign_dy = np.sign(dy) if dy != 0 else 1
        
        pt1_x = (D * dy + sign_dy * dx * delta_root) / dr**2 + xc
        pt2_x = (D * dy - sign_dy * dx * delta_root) / dr**2 + xc
        pt1_y = (-D * dx + abs(dy) * delta_root) / dr**2 + yc
        pt2_y = (-D * dx - abs(dy) * delta_root) / dr**2 + yc

        # 5. Cálculo dos Ângulos
        theta1 = abs(np.degrees(-np.pi/2 + np.arctan2((yc - pt1_y), (xc - pt1_x))))
        theta2 = abs(np.degrees(np.pi/2 + np.arctan2((yc - pt2_y), (xc - pt2_x))))

        # 6. Visualização
        vis_img = self.img_result_original.copy()
        
        # Desenha o retângulo do ROI para conferência (Verde)
        cv2.rectangle(vis_img, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), (0, 255, 0), 1)
        
        cv2.circle(vis_img, (int(xc), int(yc)), int(R), (255, 0, 0), 2)
        cv2.line(vis_img, (int(x_min), int(y_base)), (int(x_max), int(y_base)), (0, 255, 255), 2)
        cv2.circle(vis_img, (int(pt1_x), int(pt1_y)), 6, (0, 0, 255), -1)
        cv2.circle(vis_img, (int(pt2_x), int(pt2_y)), 6, (0, 0, 255), -1)
        
        self.display_frame(vis_img, update_original=False)
        self.lbl_status.configure(text=f"WCA (ROI): L={theta1:.2f}° | R={theta2:.2f}°")
        
        self.tree.insert("", "end", values=(len(self.tree.get_children())+1, 
                                            f"{theta1:.2f}", 
                                            f"{theta2:.2f}", 
                                            "Circle Fit (ROI)"))
    # --- FIM DAS NOVAS FUNÇÕES ---
    def show_home_view(self):
        self.show_edges = False
        self.table_container.pack_forget()
        self.canvas_view.pack(fill="both", expand=True)
        if self.img_result_original is not None:
            self.display_frame(self.img_result_original, update_original=False)
        self.lbl_status.configure(text="Active display screen")
    def show_analyze_options(self):
        # Cria uma pequena janela acima do botão
        self.option_win = ctk.CTkToplevel(self)
        self.option_win.title("Analysis Options")
        self.option_win.geometry("300x400")
        self.option_win.attributes("-topmost", True) # Garante que fique na frente

        ctk.CTkLabel(self.option_win, text="Select Chart Type", font=("Roboto", 16, "bold")).pack(pady=10)

        if self.current_mode == "WCA":
            ctk.CTkButton(self.option_win, text="Angles vs sample", command=lambda: self.plot_graph("WCA_EVO")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Distribuição (Histograma)", command=lambda: self.plot_graph("WCA_HIST")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Angles vs sample (Tree Points)", command=lambda: self.plot_graph("WCA_EVO_TWO")).pack(pady=5, padx=20, fill="x")
            
        elif self.current_mode == "HIS":
            ctk.CTkButton(self.option_win, text="Diameter vs. Time", command=lambda: self.plot_graph("HIS_DIA_TIME")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="CA vs. Time", command=lambda: self.plot_graph("HIS_TIME_CA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Angles vs. Diameter", command=lambda: self.plot_graph("HIS_ANG_DIA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Angles vs Contact Area", command=lambda: self.plot_graph("HIS_ANG_AREA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Contact Angle (medio) vs Diameter", command=lambda: self.plot_graph("HIS_DIA_CA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Contact Angle (medio) vs Diameter (FIT)", command=lambda: self.plot_graph("HIS_DIA_CA_FIT")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Contact Angles vs Diameter (FIT)", command=lambda: self.plot_graph("HIS_ANG_DIA_FIT")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Multiple plots", command=lambda: self.plot_graph("HIS_MULT")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Multiple plots", command=lambda: self.plot_graph("HIS_MULT_FIT")).pack(pady=5, padx=20, fill="x")
            
            
            
        elif self.current_mode == "DROP":
            ctk.CTkButton(self.option_win, text="Diameter vs. Time", command=lambda: self.plot_graph("DROP_DIA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Beta vs Time", command=lambda: self.plot_graph("DROP_BETA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Beta vs Diameter", command=lambda: self.plot_graph("DROP_BETA_DIA")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Beta vs Time", command=lambda: self.plot_graph("DROP_BETA_MAX")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Beta vs Time (FIT)", command=lambda: self.plot_graph("DROP_BETA_MAX_FIT")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Beta vs Time (Complete)", command=lambda: self.plot_graph("DROP_BETA_MAX_FIT_BEFORE_AFTER")).pack(pady=5, padx=20, fill="x")
            ctk.CTkButton(self.option_win, text="Diameter X H", command=lambda: self.plot_graph("DROP_DIA_ALT")).pack(pady=5, padx=20, fill="x")
            
            
    def reset_analysis(self):
        """
        Limpa todos os dados de medições, reinicia a tabela, o status 
        e remove a mídia da visualização.
        """
        # 1. Parar reprodução de vídeo/loops ativos imediatamente
        if hasattr(self, 'after_id') and self.after_id:
            self.root.after_cancel(self.after_id) # Corrigido para self.root ou o widget mestre
            self.after_id = None

        # 2. Limpar a Tabela (Treeview)
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 3. Limpar dados internos de análise
        if hasattr(self, 'vel_data'):
            self.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": []}
        
        if hasattr(self, 'vel_state'):
            self.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0}

        # 4. Resetar variáveis de controle
        self.current_frame_idx = 0
        self.analysis_active = False
        self.vel_active = False
        
        # 5. Limpar a Visualização (Imagem/Canvas)
        # Removemos a referência da imagem original e limpamos o widget
        self.img_result_original = None 
        
        # Se for um Label:
        self.canvas_view.configure(image='', text="Awaiting media...")
        self.canvas_view.image = None # Importante para evitar que o Garbage Collector falhe
        
        # Se você usa um Canvas do Tkinter em vez de Label, use:
        # self.canvas_view.delete("all")

        # 6. Atualizar Status e Feedback
        self.lbl_status.configure(text="System reset. Ready for new measurements.")
        messagebox.showinfo("Reset", "All data and images were successfully cleaned.")
    def mostrar_ajustes_histerese(self):
        """Cria o painel de ajuste de threshold na barra lateral"""
        if self.ajustes_container:
            self.ajustes_container.destroy()
            

        self.ajustes_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ajustes_container.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.ajustes_container, text="Parameter adjustment", 
                     font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))

        ctk.CTkLabel(self.ajustes_container, text="Threshold:").pack(anchor="w")
        
        self.lbl_thresh_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.threshold_histerese}")
        self.lbl_thresh_val.pack(anchor="e")

        self.slider_thresh = ctk.CTkSlider(self.ajustes_container, from_=0, to=255, 
                                           command=self.update_thresh_value)
        self.slider_thresh.set(self.threshold_histerese)
        self.slider_thresh.pack(fill="x", pady=5)
        # --- NOVO: Controle de n_pontos_tangente ---
        ctk.CTkLabel(self.ajustes_container, text="Tangent Points:").pack(anchor="w", pady=(10, 0))
        self.lbl_tangente_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.n_pontos_tangente_val}")
        self.lbl_tangente_val.pack(anchor="e")
        self.slider_tangente = ctk.CTkSlider(self.ajustes_container, from_=5, to=100, 
                                         command=self.update_tangente_value) #
        self.slider_tangente.set(self.n_pontos_tangente_val)
        self.slider_tangente.pack(fill="x", pady=5)
         # --- NOVO: Controle de min_pontos. ---
        ctk.CTkLabel(self.ajustes_container, text="Minimum Fit Points:").pack(anchor="w", pady=(10, 0))
        self.lbl_min_pontos_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.min_pontos_fit}")
        self.lbl_min_pontos_val.pack(anchor="e")
        self.slider_min_pontos = ctk.CTkSlider(self.ajustes_container, from_=4, to=100, 
                                         command=self.update_min_pontos_fit) #
        self.slider_min_pontos.set(self.min_pontos_fit)
        self.slider_min_pontos.pack(fill="x", pady=5)       

    def update_thresh_value(self, val):
        """Atualiza o valor da variável e do label em tempo real"""
        self.threshold_histerese = int(val)
        self.lbl_thresh_val.configure(text=f"{self.threshold_histerese}")
    def update_tangente_value(self, val):
        """Atualiza o valor de n_pontos_tangente em tempo real"""
        self.n_pontos_tangente_val = int(val)
        self.lbl_tangente_val.configure(text=f"{self.n_pontos_tangente_val}")
    def update_min_pontos_fit(self, val):
        """Atualiza o número mínimo de pontos necessários para o polyfit"""
        self.min_pontos_fit = int(val)
        self.lbl_min_pontos.configure(text=f"{self.min_pontos_fit}")
    def mostrar_ajustes_impact(self):
        """Cria o painel de ajuste de threshold na barra lateral"""
        if self.ajustes_container:
            self.ajustes_container.destroy()
            

        self.ajustes_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ajustes_container.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.ajustes_container, text="Parameter adjustment", 
                     font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))

        ctk.CTkLabel(self.ajustes_container, text="Threshold:").pack(anchor="w")
        
        self.lbl_thresh_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.threshold_impact}")
        self.lbl_thresh_val.pack(anchor="e")

        self.slider_thresh = ctk.CTkSlider(self.ajustes_container, from_=0, to=255, 
                                           command=self.update_thresh_value_impact)
        self.slider_thresh.set(self.threshold_impact)
        self.slider_thresh.pack(fill="x", pady=5)
             

    def update_thresh_value_impact(self, val):
        """Atualiza o valor da variável e do label em tempo real"""
        self.threshold_impact = int(val)
        self.lbl_thresh_val.configure(text=f"{self.threshold_impact}")
    def mostrar_ajustes_static(self):
        """Cria o painel de ajuste de pontos da tangente na barra lateral"""
        if self.ajustes_container:
            self.ajustes_container.destroy()
            

        self.ajustes_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ajustes_container.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.ajustes_container, text="Parameter adjustment", 
                     font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))

        ctk.CTkLabel(self.ajustes_container, text="Number of points:").pack(anchor="w")
        
        self.lbl_thresh_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.tangent_static}")
        self.lbl_thresh_val.pack(anchor="e")

        self.slider_thresh = ctk.CTkSlider(self.ajustes_container, from_=0, to=50, 
                                           command=self.update_thresh_value_static)
        self.slider_thresh.set(self.tangent_static)
        self.slider_thresh.pack(fill="x", pady=5)
             

    def update_thresh_value_static(self, val):
        """Atualiza o valor da variável e do label em tempo real"""
        self.tangent_static = int(val)
        self.lbl_thresh_val.configure(text=f"{self.tangent_static}")
    def mostrar_ajustes_static_pol(self):
        """Cria o painel de ajuste de pontos da tangente na barra lateral"""
        if self.ajustes_container:
            self.ajustes_container.destroy()
            

        self.ajustes_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.ajustes_container.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.ajustes_container, text="Parameter adjustment", 
                     font=("Roboto", 12, "bold"), text_color="#1f538d").pack(pady=(0, 10))

        ctk.CTkLabel(self.ajustes_container, text="Number of points:").pack(anchor="w")
        
        self.lbl_thresh_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.tangent_static_pol}")
        self.lbl_thresh_val.pack(anchor="e")

        self.slider_thresh = ctk.CTkSlider(self.ajustes_container, from_=0, to=50, 
                                           command=self.update_thresh_value_static_pol)
        self.slider_thresh.set(self.tangent_static_pol)
        self.slider_thresh.pack(fill="x", pady=5)
        #para os ponto da base
        ctk.CTkLabel(self.ajustes_container, text="Number of points base:").pack(anchor="w")
        
        self.lbl_thresh_val = ctk.CTkLabel(self.ajustes_container, text=f"{self.tangent_static_base_points}")
        self.lbl_thresh_val.pack(anchor="e")

        self.slider_thresh = ctk.CTkSlider(self.ajustes_container, from_=0, to=50, 
                                           command=self.update_thresh_value_static_base_points)
        self.slider_thresh.set(self.tangent_static_base_points)
        self.slider_thresh.pack(fill="x", pady=5)
             
 
    def update_thresh_value_static_pol(self, val):
        """Atualiza o valor da variável e do label em tempo real"""
        self.tangent_static_pol = int(val)
        self.lbl_thresh_val.configure(text=f"{self.tangent_static_pol}")
    def update_thresh_value_static_base_points(self, val):
        """Atualiza o valor da variável e do label em tempo real"""
        self.tangent_static_base_points = int(val)
        self.lbl_thresh_val.configure(text=f"{self.tangent_static_base_points}")
    def plot_graph(self, type):
        # Coleta dados da Treeview (independente de quantos arquivos/linhas existam)
        data = []
        for item in self.tree.get_children():
            data.append(self.tree.item(item)['values'])

        if not data:
            messagebox.showwarning("Warning", "The data table is empty!")
            return

        df = pd.DataFrame(data)
        
        # WCA e DROP ja estao modularizados em funcoes pequenas no servico de plots.
        if render_extracted_plot(type, df):
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            return

        
        plt.figure(figsize=(8, 5)) # verificar isso para corrigir as outras funções
        
        if type == "HIS_TIME_CA":
            

            t = pd.to_numeric(df[1], errors='coerce')
            e = pd.to_numeric(df[6], errors='coerce')
            d = pd.to_numeric(df[7], errors='coerce')

            
            mask = (~t.isna()) & (~d.isna()) & (~e.isna()) & (e <= 170) & (d <= 170)

            t = t[mask]
            e = np.round(e[mask]) # Aplica o tratamento para o ângulo esquerdo
            d = np.round(d[mask]) # Mantém o tratamento do ângulo direito


            # Plota os dois lados de forma independente com cores/legendas diferentes
            plt.scatter(t, e, s=10, label="Left angle (°)", color="blue")
            plt.scatter(t, d, s=10, label="Right angle (°)", color="red")

            plt.xlabel("Time (s)")
            plt.ylabel("Contact Angle (°)")
            plt.title("Contact Angle vs Time")
            
            # Habilita a legenda para identificar o que é o df[6] e o df[7]
            plt.legend()
            

        elif type == "HIS_DIA_CA":
            # 1. Converter colunas para numérico (garante que valores inválidos virem NaN)
            df[2] = pd.to_numeric(df[2], errors='coerce')
            df[6] = pd.to_numeric(df[6], errors='coerce')
            df[7] = pd.to_numeric(df[7], errors='coerce')

            # 2. Filtrar o DataFrame original: Mantém apenas linhas onde ambos os ângulos são <= 180
            # Isso automaticamente remove o diâmetro (coluna 2) daquela linha também
            df_filtered = df[(df[6] <= 170) & (df[7] <= 170)].copy()

            # 3. Verificar se sobraram dados após o filtro
            if not df_filtered.empty:
                t = df_filtered[2]
                ang_esq = df_filtered[6]
                ang_dir = df_filtered[7]
                
                # Cálculo da média
                ang_medio = ((ang_esq + ang_dir) / 2).round().astype(int)
                
                plt.scatter(t, ang_medio, color='red', s=10)
                plt.xlabel("Diameter (mm)")
                plt.ylabel("Contact Angle (°)")
                plt.title("Drop Diameter vs Contact Angle")
            else:
                print("Warning: Todos os dados foram filtrados (ângulos > 180°).")
        
        elif type == "HIS_DIA_CA_FIT":
            # 1. Converter colunas para numérico (garante que valores inválidos virem NaN)
           # df[2] = pd.to_numeric(df[2], errors='coerce')
           # df[3] = pd.to_numeric(df[3], errors='coerce')
           # df[4] = pd.to_numeric(df[4], errors='coerce')
            df[2] = np.round(pd.to_numeric(df[2], errors='coerce'), 0)
            df[3] = np.round(pd.to_numeric(df[6], errors='coerce'), 0)
            df[4] = np.round(pd.to_numeric(df[3], errors='coerce'), 0)

            # 2. Filtrar o DataFrame original: Mantém apenas linhas onde ambos os ângulos são <= 180
            # Adicionado .dropna() para garantir que o fit não receba valores nulos
            df_filtered = df[(df[3] <= 170) & (df[4] <= 170)].dropna().copy()

            # 3. Verificar se sobraram dados após o filtro
            if not df_filtered.empty:
                # Extração dos dados para o ajuste (X = Diâmetro, Y = Ângulo Médio)
                t = df_filtered[2].values
                ang_esq = df_filtered[3]
                ang_dir = df_filtered[4]
                
                # Cálculo da média dos ângulos
                ang_medio = ((ang_esq + ang_dir) / 2).values
                
                # Plot dos pontos reais (Scatter)
                plt.scatter(t, ang_medio, color='red', s=10, label="Dados Experimentais")
                
                # --- APLICAÇÃO DO AJUSTE DE CURVA (FIT) ---
                # Verifica se a quantidade de pontos atende ao seu slider dinâmico
                if len(t) >= self.min_pontos_fit:
                    try:
                        # np.polyfit(x, y, 1) faz uma regressão linear (grau 1)
                        # Retorna os coeficientes [inclinação, intercepto]
                        z = np.polyfit(t, ang_medio, 1)
                        p = np.poly1d(z)
                        
                        # Criar pontos para a linha de tendência (do menor ao maior diâmetro)
                        t_linha = np.linspace(t.min(), t.max(), 100)
                        
                        # Plot da linha de tendência
                        plt.plot(t_linha, p(t_linha), "--", color='blue', linewidth=2, label="Tendência Linear")
                        
                        # Opcional: Imprime no console os parâmetros do ajuste
                        print(f"Fit realizado: y = {z[0]:.4f}x + {z[1]:.4f} (n={len(t)})")
                        
                    except Exception as e:
                        print(f"Error calculating the adjustment: {e}")

                plt.xlabel("Diameter (mm)")
                plt.ylabel("Contact Angle (°)")
                plt.title("Drop Diameter vs Contact Angle")
                plt.legend() # Adiciona a legenda para identificar dados vs fit
            else:
                print("Warning: All data has been filtered. (ângulos > 180°).")
            
        elif type == "HIS_DIA_TIME":
               # No modo HIS: Coluna 0 é Tempo, Coluna 3 é Diâmetro
            tempo = df[1].astype(float)
            diametros = df[2].astype(float)  
            plt.plot(tempo, diametros, marker='o', markersize=3, linestyle='-', color='purple')
            plt.xlabel("Time (s)")
            plt.ylabel("Diameter (px)")
            plt.title("Evolution of Droplet Diameter")
            
        elif type == "HIS_ANG_DIA":
            # No modo HIS configurado anteriormente: 
            # df[0] = Tempo, df[1] = Ang Esq, df[2] = Ang Dir, df[3] = Diâmetro

            ang_esq = pd.to_numeric(df[6], errors='coerce')
            ang_dir = pd.to_numeric(df[7], errors='coerce')
            diametro = pd.to_numeric(df[2], errors='coerce')
            
            # --- NOVA CONDIÇÃO DE FILTRAGEM ---
            # Mantém apenas as linhas onde ambos os ângulos são menores ou iguais a 180
            mask = (ang_esq <= 170) & (ang_dir <= 170)
    
            ang_esq = np.round(ang_esq[mask])
            ang_dir = np.round(ang_dir[mask])
            diametro = diametro[mask]
            # ----------------------------------

            fig, ax1 = plt.subplots(figsize=(8, 5))

            # Eixo para os Ângulos
            color_esq, color_dir = 'tab:blue', 'tab:red'
            ax1.set_xlabel('Diameter (px)')
            ax1.set_ylabel('Contact Angle (°)', color='black')
            ax1.scatter(diametro, ang_esq, color=color_esq, label='Left Angle', alpha=0.6)
            ax1.scatter(diametro, ang_dir, color=color_dir, label='Right Angle', alpha=0.6)
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.legend(loc='upper left')

            ax1.set_title("Correlation: Contact Angles vs. Diameter")
            
            """Gera gráficos empilhados compartilhando o eixo do tempo, inspirados em dadosde molhabilidade/histérese de gotas.
            """
            
        elif type == "HIS_MULT":
                
            # 1. Conversão e tratamento dos dados numéricos
            t = pd.to_numeric(df[1], errors="coerce")
            diam_base = pd.to_numeric(df[3], errors="coerce")
            av_esq = pd.to_numeric(df[4], errors="coerce")
            av_dir = pd.to_numeric(df[5], errors="coerce")
            ang_esq = pd.to_numeric(df[6], errors="coerce")
            ang_dir = pd.to_numeric(df[7], errors="coerce")

            # Máscara para garantir que linhas com NaNs em dados essenciais sejam limpas
            mask = (~t.isna()) & (~ang_esq.isna()) & (~ang_dir.isna()) & (ang_esq <= 170) & (ang_dir <= 170)

            t = t[mask]
            diam_base = diam_base[mask]
            av_esq = av_esq[mask]
            av_dir = av_dir[mask]
            ang_esq = np.round(ang_esq[mask])
            ang_dir = np.round(ang_dir[mask])

            # 2. Criar a estrutura de subplots empilhados (3 gráficos verticais)
            # sharex=True garante o alinhamento temporal perfeito entre os painéis
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

            # -------------------------------------------------------------------------
            # PAINÉL (a): Ângulos de Contato (Esquerdo e Direito) vs Tempo
            # -------------------------------------------------------------------------
            ax1.scatter(t, ang_esq, s=12, color="blue", label="Left angle. (°)")
            ax1.scatter(t, ang_dir, s=12, color="red", label="Right angle. (°)")
            ax1.set_ylabel(r"$\theta$ [degree]")
            ax1.set_title("Hysteresis Dynamics vs Time")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            # -------------------------------------------------------------------------
            # PAINÉL (b): Diâmetro da Base vs Tempo
            # -------------------------------------------------------------------------
            ax2.scatter(
                t,
                diam_base,
                s=12,
                color="purple",
                label="Diâm. Base",
                facecolors="none",
                edgecolors="purple",
            )
            ax2.set_ylabel("Base Diameter [px]")
            ax2.grid(True, linestyle="--", alpha=0.5)

            # -------------------------------------------------------------------------
            # PAINÉL (c): Avanço Horizontal (Esquerdo e Direito) vs Tempo
            # -------------------------------------------------------------------------
            ax3.scatter(t, av_esq, s=12, color="teal", label="Left shift (px)")
            ax3.scatter(t, av_dir, s=12, color="darkorange", label="Right shift (px)")
            ax3.set_xlabel("Time [sec]")
            ax3.set_ylabel("Displacement [px]")
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right")
        elif type == "HIS_MULT_FIT":
            from scipy.interpolate import UnivariateSpline  # Import necessário para a suavização

            # 1. Conversão e tratamento dos dados numéricos
            t = pd.to_numeric(df[1], errors="coerce")
            diam_base = pd.to_numeric(df[3], errors="coerce")
            av_esq = pd.to_numeric(df[4], errors="coerce")
            av_dir = pd.to_numeric(df[5], errors="coerce")
            ang_esq = pd.to_numeric(df[6], errors="coerce")
            ang_dir = pd.to_numeric(df[7], errors="coerce")

            # Máscara para garantir que linhas com NaNs em dados essenciais sejam limpas
            mask = (~t.isna()) & (~ang_esq.isna()) & (~ang_dir.isna()) & (ang_esq <= 170) & (ang_dir <= 170)

            t = t[mask].to_numpy()
            diam_base = diam_base[mask].to_numpy()
            av_esq = av_esq[mask].to_numpy()
            av_dir = av_dir[mask].to_numpy()
            ang_esq = np.round(ang_esq[mask].to_numpy())
            ang_dir = np.round(ang_dir[mask].to_numpy())

            # Garantir ordenação temporal para que o spline funcione corretamente
            sort_idx = np.argsort(t)
            t, diam_base, av_esq, av_dir, ang_esq, ang_dir = (
                t[sort_idx], diam_base[sort_idx], av_esq[sort_idx], av_dir[sort_idx], ang_esq[sort_idx], ang_dir[sort_idx]
            )

            # Vetor de tempo contínuo e denso para plotar as linhas de tendência suaves
            t_smooth = np.linspace(t.min(), t.max(), 300)

            # 2. Criar a estrutura de subplots empilhados (3 gráficos verticais)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

            # -------------------------------------------------------------------------
            # PAINÉL (a): Ângulos de Contato (Esquerdo e Direito) vs Tempo
            # -------------------------------------------------------------------------
            ax1.scatter(t, ang_esq, s=12, color="blue", alpha=0.4, label="Left angle (°)")
            ax1.scatter(t, ang_dir, s=12, color="red", alpha=0.4, label="Right angle (°)")
            
            # Ajuste das Splines (Aumente o valor de 's' para curvas mais suaves)
            spl_ang_esq = UnivariateSpline(t, ang_esq, s=len(t) * 2)
            spl_ang_dir = UnivariateSpline(t, ang_dir, s=len(t) * 2)
            
            ax1.plot(t_smooth, spl_ang_esq(t_smooth), color="darkblue", lw=2, label="Trend (Left)")
            ax1.plot(t_smooth, spl_ang_dir(t_smooth), color="darkred", lw=2, label="Trend (Right)")
            
            ax1.set_ylabel(r"$\theta$ [degree]")
            ax1.set_title("Hysteresis Dynamics vs Time")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            # -------------------------------------------------------------------------
            # PAINÉL (b): Diâmetro da Base vs Tempo
            # -------------------------------------------------------------------------
            ax2.scatter(t, diam_base, s=12, color="purple", alpha=0.4, facecolors="none", edgecolors="purple", label="Diâm. Base")
            
            # Ajuste da Spline para o diâmetro da base
            spl_diam = UnivariateSpline(t, diam_base, s=len(t) * 1.5)
            ax2.plot(t_smooth, spl_diam(t_smooth), color="indigo", lw=2, label="Trend (Base)")
            
            ax2.set_ylabel("Base Diameter [px]")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            # -------------------------------------------------------------------------
            # PAINÉL (c): Avanço Horizontal (Esquerdo e Direito) vs Tempo
            # -------------------------------------------------------------------------
            ax3.scatter(t, av_esq, s=12, color="teal", alpha=0.4, label="Left shift (px)")
            ax3.scatter(t, av_dir, s=12, color="darkorange", alpha=0.4, label="Right shift (px)")
            
            # Ajuste das Splines para os deslocamentos horizontais
            spl_av_esq = UnivariateSpline(t, av_esq, s=len(t) * 2)
            spl_av_dir = UnivariateSpline(t, av_dir, s=len(t) * 2)
            
            ax3.plot(t_smooth, spl_av_esq(t_smooth), color="darkslategray", lw=2, label="Trend (Left Shift)")
            ax3.plot(t_smooth, spl_av_dir(t_smooth), color="chocolate", lw=2, label="Trend (Right Shift)")
            
            ax3.set_xlabel("Time [sec]")
            ax3.set_ylabel("Displacement [px]")
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="upper right")
        elif type == "HIS_ANG_DIA_FIT":

            # df[0] = Frame, df[1] = Tempo (s), df[2] = Diam. (px)
            # df[3] = Ângulo Avançante (°), df[4] = Ângulo Recuante (°)
            frame = pd.to_numeric(df[0], errors='coerce')
            tempo = pd.to_numeric(df[1], errors='coerce')
            diametro_raw = pd.to_numeric(df[2], errors='coerce')
            ang_esq_raw = pd.to_numeric(df[6], errors='coerce')
            ang_dir_raw = pd.to_numeric(df[7], errors='coerce')

            # 1. Remove linhas contendo N/A, NaN ou valores inválidos
            mask_validos = (
                ~diametro_raw.isna() &
                ~ang_esq_raw.isna() &
                ~ang_dir_raw.isna() &
                (ang_esq_raw >= 0) & (ang_esq_raw <= 170) &
                (ang_dir_raw >= 0) & (ang_dir_raw <= 170)
            )

            frame = frame[mask_validos].to_numpy()
            tempo = tempo[mask_validos].to_numpy()
            #diametro = diametro_raw[mask_validos].to_numpy()
            #ang_esq = ang_esq_raw[mask_validos].to_numpy()
            #ang_dir = ang_dir_raw[mask_validos].to_numpy()
            ang_esq = np.round(ang_esq_raw[mask_validos].to_numpy(), 0)
            ang_dir = np.round(ang_dir_raw[mask_validos].to_numpy(), 0)
            diametro = np.round(diametro_raw[mask_validos].to_numpy(), 0)

            if len(diametro) < 5:
                print("Pontos insuficientes para realizar a segmentação e os ajustes.")
                return

            # 2. Suavização do diâmetro para determinar a tendência dinâmica sem ruídos
            window = min(15, len(diametro) if len(diametro) % 2 != 0 else len(diametro) - 1)
            if window > 3:
                diam_suave = savgol_filter(diametro, window_length=window, polyorder=1)
            else:
                diam_suave = diametro.copy()

            # Calcula o gradiente do diâmetro para segmentação das regiões
            derivada_diam = np.gradient(diam_suave)

            # Limiar de tolerância para detecção do regime estático (ajuste se necessário)
            limiar_estatico = 0.15 

            # 3. Separação dos índices de dados
            idx_cresce = np.where(derivada_diam > limiar_estatico)[0]
            idx_constante = np.where(np.abs(derivada_diam) <= limiar_estatico)[0]
            idx_decresce = np.where(derivada_diam < -limiar_estatico)[0]

            # 4. Configuração da Figura
            fig, ax = plt.subplots(figsize=(9, 6))
            
            # Paleta de cores (Tons de Azul/Ciano para Esquerda, Vermelho/Laranja para Direita)
            cores_esq = {'cresce': '#1f77b4', 'constante': '#17becf', 'decresce': '#aec7e8'}
            cores_dir = {'cresce': '#d62728', 'constante': '#ff7f0e', 'decresce': '#ffbb78'}

            # 5. Ajustes Direcionais Conforme o Regime Dinâmico

            # --- REGIME 1: DIÂMETRO CRESCENTE (Ajuste padrão em X) ---
            if len(idx_cresce) >= 3:
                d_reg = diametro[idx_cresce]
                ae_reg = ang_esq[idx_cresce]
                ad_reg = ang_dir[idx_cresce]
                ax.scatter(d_reg, ae_reg, color=cores_esq['cresce'], alpha=0.4, edgecolors='none')
                ax.scatter(d_reg, ad_reg, color=cores_dir['cresce'], alpha=0.4, edgecolors='none')
                
                d_espaco = np.linspace(d_reg.min(), d_reg.max(), 100)
                p_esq = np.polyfit(d_reg, ae_reg, 1)
                p_dir = np.polyfit(d_reg, ad_reg, 1)
                ax.plot(d_espaco, np.polyval(p_esq, d_espaco), color=cores_esq['cresce'], linestyle='-', linewidth=2, label=r'$\theta_{Esq}$ (Crescente)')
                ax.plot(d_espaco, np.polyval(p_dir, d_espaco), color=cores_dir['cresce'], linestyle='--', linewidth=2, label=r'$\theta_{Dir}$ (Crescente)')

            # --- REGIME 2: DIÂMETRO ESTÁTICO (Ajuste Invertido na Direção Y) ---
            if len(idx_constante) >= 3:
                d_reg = diametro[idx_constante]
                ae_reg = ang_esq[idx_constante]
                ad_reg = ang_dir[idx_constante]
                ax.scatter(d_reg, ae_reg, color=cores_esq['constante'], alpha=0.4, edgecolors='none')
                ax.scatter(d_reg, ad_reg, color=cores_dir['constante'], alpha=0.4, edgecolors='none')
                
                # Ajuste Esquerdo Estático: X = f(Y)
                p_esq_inv = np.polyfit(ae_reg, d_reg, 1)
                ang_espaco_esq = np.linspace(ae_reg.min(), ae_reg.max(), 100)
                diam_fit_esq = np.polyval(p_esq_inv, ang_espaco_esq)
                ax.plot(diam_fit_esq, ang_espaco_esq, color=cores_esq['constante'], linestyle='-', linewidth=2,
                        label=r'$\theta_{Esq}$ (Estático - Y fit)')

                # Ajuste Direito Estático: X = f(Y)
                p_dir_inv = np.polyfit(ad_reg, d_reg, 1)
                ang_espaco_dir = np.linspace(ad_reg.min(), ad_reg.max(), 100)
                diam_fit_dir = np.polyval(p_dir_inv, ang_espaco_dir)
                ax.plot(diam_fit_dir, ang_espaco_dir, color=cores_dir['constante'], linestyle='--', linewidth=2,
                        label=r'$\theta_{Dir}$ (Estático - Y fit)')

            # --- REGIME 3: DIÂMETRO DECRESCENTE (Ajuste padrão em X) ---
            if len(idx_decresce) >= 3:
                d_reg = diametro[idx_decresce]
                ae_reg = ang_esq[idx_decresce]
                ad_reg = ang_dir[idx_decresce]
                ax.scatter(d_reg, ae_reg, color=cores_esq['decresce'], alpha=0.4, edgecolors='none')
                ax.scatter(d_reg, ad_reg, color=cores_dir['decresce'], alpha=0.4, edgecolors='none')
                
                d_espaco = np.linspace(d_reg.min(), d_reg.max(), 100)
                p_esq = np.polyfit(d_reg, ae_reg, 1)
                p_dir = np.polyfit(d_reg, ad_reg, 1)
                ax.plot(d_espaco, np.polyval(p_esq, d_espaco), color=cores_esq['decresce'], linestyle='-', linewidth=2,
                        label=r'$\theta_{Esq}$ (Decrescente)')
                ax.plot(d_espaco, np.polyval(p_dir, d_espaco), color=cores_dir['decresce'], linestyle='--', linewidth=2,
                        label=r'$\theta_{Dir}$ (Decrescente)')

            # 6. Estilização Gráfica Padrão Periódico Científico
            ax.set_xlabel('Droplet Base Diameter (px)', fontsize=12, fontweight='bold', labelpad=8)
            ax.set_ylabel('Contact Angle (°)', fontsize=12, fontweight='bold', labelpad=8)
            ax.set_title('WLS Hysteresis Correlation: Contact Angles vs. Base Diameter', fontsize=13, fontweight='bold', pad=12)
            
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(loc='best', fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#d3d3d3')
            
            plt.tight_layout()    
        elif type == "HIS_ANG_AREA":

            try:
                ang_esq = pd.to_numeric(df[3])
                ang_dir = pd.to_numeric(df[4])
                area    = pd.to_numeric(df[6])
            except Exception as e:
                messagebox.showerror("Erro", "Insufficient data columns for Area.")
                return

            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Média dos ângulos para ver a tendência central
            ang_medio = (ang_esq + ang_dir) / 2
            ax.plot(area, ang_medio, color='black', linestyle='--', label='Média', alpha=0.7)

            ax.set_xlabel("Contact Area (px²)")
            ax.set_ylabel("Contact Angle (°)")
            ax.set_title("Contact Angle vs. Contact Area")
            ax.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def show_data_table(self):
        self.canvas_view.pack_forget()
        self.table_container.pack(fill="both", expand=True)
        
        # Configurar cabeçalho conforme o modo atual
        headers = {
            "WCA": ["Frame", "Ângulo Esq (°)", "Ângulo Dir (°)", "Média (°)"],
            "HIS": ["Frame", "Tempo (s)", "Diâmetro (px)", "Diâm. Base (px)", "Avanc. Esq (px)", "Avanc. Dir (px)", "Âng. Esq (°)", "Âng. Dir (°)", "Histérese (°)", "Área Base (px²)"],
            "DROP": ["Tempo (s)", "Posição (mm)", "Diâmetro (mm)", "Beta", "Altura (mm)", "Área (mm²)", "dD/dt (mm/s)"]
        }
        
        
        cols = headers.get(self.current_mode, ["Dados"])
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor="center")
        
        self.lbl_status.configure(text=f"Tabela de Dados: Modo {self.current_mode}")

    def create_main_menu(self):
    # --- ADICIONE ESTAS LINHAS AQUI ---
        if hasattr(self, 'ajustes_container') and self.ajustes_container:
            self.ajustes_container.destroy()
            self.ajustes_container = None
    # ----------------------------------
        for child in self.menu_container.winfo_children(): child.destroy()
        btn_static = ctk.CTkButton(self.menu_container, text="Static Angle", command=lambda: self.show_sub_menu("WCA"))
        btn_static.pack(pady=10, fill="x")
        Tooltip(btn_static, "Measure the contact angle of a stationary droplet using different fitting methods.")
        btn_hysteresis = ctk.CTkButton(self.menu_container, text="Hysteresis", command=lambda: self.show_sub_menu("HIS"))
        btn_hysteresis.pack(pady=10, fill="x")
        Tooltip(btn_hysteresis, "Analyze advancing and receding angles to determine surface wettability properties.")
        btn_dropimpact = ctk.CTkButton(self.menu_container, text="Drop Impact", command=lambda: self.show_sub_menu("DROP"))
        btn_dropimpact.pack(pady=10, fill="x")
        Tooltip(btn_dropimpact, "Calculate the velocity and spreading factor of a droplet during impact.")
        btn_info = ctk.CTkButton(
            self.menu_container, 
            text="Important Information", 
            fg_color="#A93226",  # Cor de destaque (avermelhado/alerta)
            hover_color="#7B241C",
            command=self.show_important_info
        )
        btn_info.pack(padx=20, pady=20)
        Tooltip(btn_info, "Important information about analyses and files.")
    def show_sub_menu(self, mode):
        self.current_mode = mode
        for child in self.menu_container.winfo_children(): child.destroy()
        titles = {
            "WCA": ["Tangent method", "Polynomial Fit", "Three points"], 
            "HIS": ["Adv-Rec", "Polinomial", "Weighted Least Squares"], 
            "DROP": ["First method", "Second method", "Third method", "Four method"]
        }
        
        ctk.CTkButton(self.menu_container, text="← Return", fg_color="#444", command=self.create_main_menu).pack(pady=(0, 10), fill="x")
        
        for method in titles[mode]:
            if method == "Tangent method":
                btn_normal = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_static(), self.run_analise_normal()])
                btn_normal.pack(pady=5, fill="x")
                Tooltip(btn_normal, "To measure the equilibrium contact angle of a stationary droplet on a horizontal surface.")
            elif method == "Polynomial Fit":
                btn_fitting = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_static_pol(), self.run_analise_avancada()])
                btn_fitting.pack(pady=5, fill="x")
                Tooltip(btn_fitting, r"This method uses a second-degree equation to automatically model the droplet's curvature and calculate the tangent at the contact point")            
            elif method == "Three points":
                btn_points = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_static_pol(), self.run_analise_manual()])
                btn_points.pack(pady=5, fill="x")
                Tooltip(btn_points, r"This manual approach defines the contact angle by connecting three user-selected points to form the base and the slope of the droplet.")
            elif method == "Adv-Rec":
                btn_adv_rec = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_histerese(), self.run_histerese_avancante()])
                btn_adv_rec.pack(pady=5, fill="x")
                Tooltip(btn_adv_rec, r"To measure the Advancing Angle ($\theta_A$) and the Receding Angle ($\theta_R$)")
            elif method == "Polinomial":
                btn_polinomial = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_histerese(), self.run_histerese_polinomial()])
                btn_polinomial.pack(pady=5, fill="x")
                Tooltip(btn_polinomial, r"To measure the Advancing Angle ($\theta_A$) and the Receding Angle ($\theta_R$)")
            elif method == "Weighted Least Squares":
                btn_run_wls = ctk.CTkButton(
                    self.menu_container, 
                    text=method, 
                    # Travamos a referência exata das funções no momento da criação do botão
                    command=lambda app_ref=self: app_ref.run_histerese_wls()
                )
                btn_run_wls.pack(pady=5, fill="x")
                Tooltip(btn_run_wls, "Calculate the contact angle using polynomials with Weighted Least Squares (WLS).")
               
            elif method == "First method":
                btn_first = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_impact(), self.run_analise_velocidade()])
                btn_first.pack(pady=5, fill="x")
                Tooltip(btn_first, "Performs impact detection based on point filtering at the surface level, providing a simplified tracking of the drop's vertical approach")
            elif method == "Second method":
                btn_second = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_impact(), self.run_velocity_analysis()])
                btn_second.pack(pady=5, fill="x")
                Tooltip(btn_second, "Utilizes advanced morphological segmentation and convex hull geometry to accurately measure the maximum spreading factor and expansion dynamics.")
            elif method == "Third method":
                btn_Third = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_impact(), self.run_analise_impacto_tres()])
                btn_Third.pack(pady=5, fill="x")
                Tooltip(btn_Third, "test.")
            elif method == "Four method":
                btn_four = ctk.CTkButton(self.menu_container, text=method, command=lambda: [self.mostrar_ajustes_impact(), self.run_drop_impact_analysis()])
                btn_four.pack(pady=5, fill="x")
                Tooltip(btn_four, "This method analyzes droplet impact by measuring the droplet diameter at the baseline during spreading using image segmentation.")
            else:
                ctk.CTkButton(self.menu_container, text=method, command=self.load_media).pack(pady=5, fill="x")

    def apply_zoom(self, factor):
        self.zoom_factor *= factor
        if self.zoom_factor < 1.0:
            self.zoom_factor = 1.0
            self.roi_zoom = None
            
        if self.img_result_original is not None:
            self.display_frame(self.img_result_original, update_original=False)

    def on_mouse_wheel(self, event):
        if self.img_result_original is not None:
            if event.delta > 0 or event.num == 4: self.apply_zoom(1.1)
            else: self.apply_zoom(0.9)
    def on_mouse_scroll_windows(self, event):
        """Trata o zoom via scroll do mouse especificamente no Windows."""
        if event.delta > 0:
            # Scroll para cima -> Zoom In (Aumenta o fator)
            self.apply_zoom(1.1)
        elif event.delta < 0:
            # Scroll para baixo -> Zoom Out (Diminui o fator)
            self.apply_zoom(0.9)
    def on_click_press(self, event):
        if self.img_result_original is not None:
            self.dragging = True
            self.start_x, self.start_y = event.x, event.y

    def on_mouse_drag(self, event):
        pass 

    def on_click_release(self, event):
        if self.dragging and self.img_result_original is not None:
            self.dragging = False
            end_x, end_y = event.x, event.y
            if abs(end_x - self.start_x) > 10 and abs(end_y - self.start_y) > 10:
                self.roi_zoom = (self.start_x, self.start_y, end_x, end_y)
                self.update_zoom_view(is_roi=True)

    def update_zoom_view(self, is_roi=False):
        # A lógica de recorte foi movida de forma segura para o display_frame.
        # Aqui, apenas avisamos a interface para atualizar com os dados recentes.
        if self.img_result_original is not None:
            self.display_frame(self.img_result_original, update_original=False)

    def display_frame(self, frame, update_original=True):
        if frame is None:
            return

        if self.cap and hasattr(self, 'video_slider_var'):
            current_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._syncing_slider = True
            try:
                self.video_slider_var.set(current_idx)
            finally:
                self._syncing_slider = False

        # Atualiza o frame histórico, MAS NÃO RESETA O ZOOM DURANTE O VÍDEO!
        if update_original:
            self.img_result_original = frame.copy()
            # As variáveis self.zoom_factor e self.roi_zoom não devem ser zeradas aqui.

        # Prevenção de segurança caso sejam chamados antes da inicialização
        if not hasattr(self, 'zoom_factor'): self.zoom_factor = 1.0
        if not hasattr(self, 'roi_zoom'): self.roi_zoom = None

        # ====================================================
        # CORTE DO ZOOM CENTRALIZADO EM TEMPO REAL
        # ====================================================
        h, w = frame.shape[:2]

        if self.roi_zoom is not None:
            tw = self.canvas_view.winfo_width()
            th = self.canvas_view.winfo_height()
            if tw > 0 and th > 0:
                x1, y1, x2, y2 = self.roi_zoom
                rx1 = int(min(x1, x2) * (w / tw))
                ry1 = int(min(y1, y2) * (h / th))
                rx2 = int(max(x1, x2) * (w / tw))
                ry2 = int(max(y1, y2) * (h / th))
                
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(w, rx2), min(h, ry2)
                
                if rx2 > rx1 and ry2 > ry1:
                    frame = frame[ry1:ry2, rx1:rx2]

        elif self.zoom_factor > 1.0:
            new_w, new_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
            x1, y1 = max(0, (w - new_w) // 2), max(0, (h - new_h) // 2)
            frame = frame[y1:y1+new_h, x1:x1+new_w]
        # ====================================================

        self.update_idletasks()
        
        # Redimensiona as variáveis após o recorte
        h, w = frame.shape[:2]
        self.current_vis_img = frame.copy()

        # O restante do seu código (window_w = self.winfo_width()...) continua intacto abaixo:
        window_w = self.winfo_width()
        window_h = self.winfo_height()
        # ...

        # 4. Definir limites fixos para a área da imagem
        # Descontamos o espaço ocupado pelos menus e tabelas (aprox. 300px na vertical)
        # E o espaço lateral (aprox. 250px se houver barra lateral)
        max_allowed_w = window_w - 300 
        max_allowed_h = window_h - 250 

        # Fallback caso a janela ainda não tenha tamanho definido
        if max_allowed_w <= 100: max_allowed_w = 800
        if max_allowed_h <= 100: max_allowed_h = 500

        # 5. Calcular o ratio baseado no espaço REAL que sobra na tela
        ratio = min(max_allowed_w / w, max_allowed_h / h)
        
        # Opcional: Impedir que a imagem seja ampliada além do tamanho original
        # ratio = min(ratio, 1.0) 

        new_size = (int(w * ratio), int(h * ratio))

        # 6. Conversão e exibição
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        
        # Usar CTkImage com o tamanho calculado rigidamente
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=new_size)
        
        self.canvas_view.configure(image=img_tk, text="")
        self.canvas_view.image = img_tk
    
    def on_slider_move(self, value):
        """Atualiza o vídeo dinamicamente baseado na rolagem do Slider."""
        if self.cap and not getattr(self, '_syncing_slider', False):
            frame_destino = int(float(value))
            
            # 1. Sincroniza a variável de controle de frames do aplicativo
            self.current_frame_idx = frame_destino
            
            # 2. Força o OpenCV a pular rigidamente para essa posição
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_destino)
            
            # 3. Lê o frame imediatamente para limpar o buffer e renderizar na tela
            ret, frame = self.cap.read()
            if ret:
                self.img_result_original = frame.copy()
                self.display_frame(frame, update_original=False)
                
                # 4. Mantém o ponteiro fixado no frame correto para a próxima leitura do 'Start'
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_destino)
    def robust_segmentation(self, frame_gray, background):
        """Subtração de background e realce da gota em movimento."""
        diff = cv2.absdiff(background, frame_gray)
        blurred = cv2.medianBlur(diff, 5)
        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def run_analise_impacto_tres(self):
        """Método 3 com seleção manual de frame de referência e baseline."""
        metodo_atual = "DROP_TERCEIRO" 
        if self.ultimo_metodo_selecionado != metodo_atual:
            for item in self.tree.get_children(): self.tree.delete(item)
            self.ultimo_metodo_selecionado = metodo_atual
        
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("Todos", "*.*")])
        if not path: return
        self.video_path = path
        # --- CARREGA PARÂMETROS FÍSICOS ---
        self.params = self.configurar_parametros_velocidade() # Agora usa a janela de config
        
        cap_config = cv2.VideoCapture(path)
        total = int(cap_config.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        cap_config = cv2.VideoCapture(path)
        total = int(cap_config.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        
        # --- 1. Navegação de Frames para determinar o Background ---
        win_frame = "Config: Select Background"
        cv2.namedWindow(win_frame, cv2.WINDOW_NORMAL)
        
        while True:
            cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_ref = cap_config.read()
            if not ret: break
            
            vis = frame_ref.copy()
            cv2.putText(vis, f"Frame: {idx}/{total-1} - A/D navigate, ENTER select.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win_frame, vis)
            
            k = cv2.waitKey(0) & 0xFF
            if k == ord('d'): idx = min(idx + 1, total - 1)
            elif k == ord('a'): idx = max(idx - 1, 0)
            elif k == 13: # ENTER
                self.background_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
                break
            elif k == 27: # ESC
                cv2.destroyWindow(win_frame)
                cap_config.release()
                return
        
        cv2.destroyWindow(win_frame)

        # --- 2. Seleção da Baseline (Linha do Chão) com Navegação de Frames ---
        self.pts_base = []
        win_baseline = "Config: Define Baseline"
        cv2.namedWindow(win_baseline, cv2.WINDOW_NORMAL)
        
        def click_ev(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                # Permite apenas 2 pontos. Se já tiver 2, não faz nada até limpar/reiniciar
                if len(self.pts_base) < 2:
                    self.pts_base.append((x, y))

        cv2.setMouseCallback(win_baseline, click_ev)

        while True:
            # Carrega o frame baseado no 'idx' atual (permite navegar aqui também)
            cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_ref = cap_config.read()
            if not ret: break
            
            vis_base = frame_ref.copy()
            
            # Texto explicativo na tela
            cv2.putText(vis_base, f"Frame: {idx}/{total-1} - A/D navigate. Click 2 points.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis_base, "ENTER: Confirm Baseline | R: Reset Points | ESC: Exit", 
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Desenha os pontos e a linha dinamicamente conforme o usuário clica
            for pt in self.pts_base:
                cv2.circle(vis_base, pt, 3, (0, 0, 255), -1)
            
            if len(self.pts_base) == 2:
                cv2.line(vis_base, self.pts_base[0], self.pts_base[1], (0, 255, 255), 2)
                
            cv2.imshow(win_baseline, vis_base)
            
            k = cv2.waitKey(30) & 0xFF  # Pequeno delay para atualizar a imagem continuamente
            
            # Controles de teclado
            if k == ord('d'): 
                idx = min(idx + 1, total - 1)
            elif k == ord('a'): 
                idx = max(idx - 1, 0)
            elif k == ord('r'): # Tecla extra opcional: reinicia os pontos se errar o clique
                self.pts_base = []
            elif k == 13: # ENTER
                if len(self.pts_base) == 2:
                    break
                else:
                    # Avisa textualmente na tela ou ignora se não tiver 2 pontos
                    continue
            elif k == 27: # ESC
                cv2.destroyAllWindows()
                cap_config.release()
                return

        # Definir o limite Y baseado na média dos pontos clicados
        self.limite_y_base = (self.pts_base[0][1] + self.pts_base[1][1]) / 2
        cv2.destroyAllWindows()
        cap_config.release()

        # --- 3. Inicialização do Processamento ---

        # --- 3. Inicialização do Processamento ---
        self.cap_vel = cv2.VideoCapture(path)
        self.current_mode = "DROP"
# CORREÇÃO: Em vez de self.params = {...}, usamos update para manter densidade/viscosidade
        self.params.update({
            "min_area": 300,
            "fps_real": self.params.get("fps_real", 10000) # Mantém o que veio da janela ou usa 10k
        })
        
        self.data = {"history_pos": [], "diam_pre": 1.0, "tempos": [], "posicoes": [], "areas_queda": []}
        self.frame_id = 0
        self.impacto_detectado = False
        self.vel_active = True
        
        self.lbl_status.configure(text="Baseline defined. Starting analysis...")
        self.processar_frame_impacto_tres()

    def processar_frame_impacto_tres(self):
        """Loop de processamento com altura intrínseca e taxa de espalhamento."""
        if not self.vel_active or self.cap_vel is None: return
        
        p, d = self.params, self.data
        ret, frame = self.cap_vel.read()
        
        if not ret:
            # --- CÁLCULOS FÍSICOS FINAIS ---
            if len(d["posicoes"]) > 2:
                tempos = np.array(d["tempos"])
                pos_m = np.array(d["posicoes"]) / 1000.0
                v_impacto, _ = np.polyfit(tempos, pos_m, 1)
                v_abs = abs(v_impacto)
                d_medio_mm = d["diam_pre"] if d["diam_pre"] > 0 else 1.0
                d_m = d_medio_mm / 1000.0
                
                re = (p["densidade"] * v_abs * d_m) / p["viscosidade"]
                we = (p["densidade"] * (v_abs**2) * d_m) / p["tensao"]
                oh = p["viscosidade"] / np.sqrt(p["densidade"] * p["tensao"] * d_m)
                area_med_px2 = sum(d["areas_queda"]) / len(d["areas_queda"]) if d["areas_queda"] else 0
                
                self.mostrar_resultados_adimensionais(v_abs, d_medio_mm, re, we, oh, area_med_px2)

            self.cap_vel.release(); self.vel_active = False
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.robust_segmentation(gray, self.background_gray)
        mask[int(self.limite_y_base):, :] = 0

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cnts:
            w_img = frame.shape[1]
            c = min(cnts, key=lambda ct: abs(cv2.moments(ct)["m10"]/(cv2.moments(ct)["m00"]+1e-5) - w_img/2))
            
            area_atual_px = cv2.contourArea(c) 
            if area_atual_px > p["min_area"]:
                _, y_topo_box, _, h_box = cv2.boundingRect(c)
                y_base_gota_box = y_topo_box + h_box
                
                # Desenhar a linha do chão para referência visual
                cv2.line(frame, (0, int(self.limite_y_base)), (w_img, int(self.limite_y_base)), (0, 255, 255), 1)

                if not self.impacto_detectado:
                    # --- FASE DE QUEDA ---
                    d["areas_queda"].append(area_atual_px)
                    d["tempos"].append(self.frame_id * p["dt"])
                    d["posicoes"].append(y_topo_box * p["escala"])
                    
                    if len(c) >= 5:
                        ellipse = cv2.fitEllipse(c)
                        d["diam_pre"] = ((ellipse[1][0] + ellipse[1][1]) / 2) * p["escala"]
                        cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
                    
                    if abs(y_base_gota_box - self.limite_y_base) < 5:
                        self.impacto_detectado = True
                else:
                    # --- FASE DE ESPALHAMENTO ---
                    hull = cv2.convexHull(c)
                    pts_suave = hull.reshape(-1, 2)
                    
                    p_esq = tuple(hull[hull[:, :, 0].argmin()][0])
                    p_dir = tuple(hull[hull[:, :, 0].argmax()][0])
                    
                    diam_pixel = p_dir[0] - p_esq[0]
                    diam_mm = diam_pixel * p["escala"]
                    beta = diam_mm / d["diam_pre"] if d["diam_pre"] > 0 else 0
                    area_mm2 = area_atual_px * (p["escala"]**2)
                    tempo_atual = self.frame_id * p["dt"]

# 2. ALTURA (Vertical): Aplicando a MESMA lógica do diâmetro
                    # Encontra o ponto com o menor Y (topo) e o maior Y (base) no contorno
                    p_topo = tuple(hull[hull[:, :, 1].argmin()][0])
                    p_base = tuple(hull[hull[:, :, 1].argmax()][0])
                    
                   # A altura é a diferença entre o Y da base e o Y do topo
                    altura_pixel = p_base[1] - p_topo[1]
                    altura_mm = altura_pixel * p["escala"]

                    # --- TAXA DE VARIAÇÃO E DESENHOS ---
                    last_d = d.get("last_diam_val")
                    dd_dt = (diam_mm - last_d) / p["dt"] if last_d is not None else 0.0
                    d["last_diam_val"] = diam_mm 

                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                    cv2.line(frame, p_esq, p_dir, (0, 0, 255), 2) # Linha do Diâmetro
# Linha da Altura (Magenta) - Exatamente como o diâmetro, mas vertical
                    # Para a linha ficar centralizada visualmente, usamos o X do topo
                    p1_alt = (p_topo[0], p_topo[1])
                    p2_alt = (p_topo[0], p_base[1]) 
                    cv2.line(frame, p1_alt, p2_alt, (255, 0, 255), 2)
                    # Inserção na Treeview
                    self.tree.insert("", "end", values=(
                        f"{tempo_atual:.5f}", "-", f"{diam_mm:.3f}", 
                        f"{beta:.3f}", f"{altura_mm:.3f}", 
                        f"{area_mm2:.3f}", f"{dd_dt:.3f}"
                    ))

        self.display_frame(frame)
        self.frame_id += 1
        self.after(1, self.processar_frame_impacto_tres)
    def run_drop_impact_analysis(self):
        """
        Método 4: Análise de impacto sobre a linha de base (sem ROI).
        Aplica as configurações e o fluxo de parâmetros/tabela do Third Method.
        Mede o diâmetro da gota exatamente sobre a linha do chão utilizando segmentação PCO.
        """
        metodo_atual = "DROP_QUARTO" 
        if self.ultimo_metodo_selecionado != metodo_atual:
            for item in self.tree.get_children(): 
                self.tree.delete(item)
            self.ultimo_metodo_selecionado = metodo_atual
        
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4 *.tif"), ("Todos", "*.*")])
        if not path: return
        self.video_path = path
        # --- CARREGA PARÂMETROS FÍSICOS (Usa a janela nativa do Third Method) ---
        self.params = self.configurar_parametros_velocidade()
        
        cap_config = cv2.VideoCapture(path)
        total = int(cap_config.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: return
        idx = 0
        
        # --- 1. Navegação de Frames para determinar o Background ---
        win_frame = "Config: Select Background"
        cv2.namedWindow(win_frame, cv2.WINDOW_NORMAL)
        
        while True:
            cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_ref = cap_config.read()
            if not ret: break
            
            vis = frame_ref.copy()
            cv2.putText(vis, f"Frame: {idx}/{total-1} - A/D navigate, ENTER select.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win_frame, vis)
            
            k = cv2.waitKey(0) & 0xFF
            if k == ord('d'): idx = min(idx + 1, total - 1)
            elif k == ord('a'): idx = max(idx - 1, 0)
            elif k == 13: # ENTER
                self.background_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
                break
            elif k == 27: # ESC
                cv2.destroyWindow(win_frame)
                cap_config.release()
                return
        
        cv2.destroyWindow(win_frame)

        # --- 2. Seleção da Baseline (Linha do Chão) com Navegação de Frames ---
        self.pts_base = []
        win_baseline = "Config: Define Baseline"
        cv2.namedWindow(win_baseline, cv2.WINDOW_NORMAL)
        
        def click_ev(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                if len(self.pts_base) < 2:
                    self.pts_base.append((x, y))

        cv2.setMouseCallback(win_baseline, click_ev)

        while True:
            cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_ref = cap_config.read()
            if not ret: break
            
            vis_base = frame_ref.copy()
            w_img = vis_base.shape[1]
            
            cv2.putText(vis_base, f"Frame: {idx}/{total-1} - A/D navigate. Click 2 points.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(vis_base, "ENTER: Confirm Baseline | R: Reset Points | ESC: Exit", 
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            for pt in self.pts_base:
                cv2.circle(vis_base, pt, 3, (0, 0, 255), -1)
            
            if len(self.pts_base) == 2:
                # --- CORREÇÃO: Desenha a reta estendida cruzando a tela inteira ---
                x1, y1 = self.pts_base[0]
                x2, y2 = self.pts_base[1]
                if x2 - x1 != 0:
                    m = (y2 - y1) / (x2 - x1)
                    c = y1 - m * x1
                    pt_inicio = (0, int(c))
                    pt_fim = (w_img, int(m * w_img + c))
                else: # Reta vertical (salvaguarda)
                    pt_inicio = (x1, 0)
                    pt_fim = (x1, vis_base.shape[0])
                cv2.line(vis_base, pt_inicio, pt_fim, (0, 255, 255), 2)
                
            cv2.imshow(win_baseline, vis_base)
            
            k = cv2.waitKey(30) & 0xFF
            
            if k == ord('d'): 
                idx = min(idx + 1, total - 1)
            elif k == ord('a'): 
                idx = max(idx - 1, 0)
            elif k == ord('r'): 
                self.pts_base = []
            elif k == 13: # ENTER
                if len(self.pts_base) == 2:
                    break
            elif k == 27: # ESC
                cv2.destroyAllWindows()
                cap_config.release()
                return

        # --- CORREÇÃO: Armazena os coeficientes da reta y = mx + c ---
        x1, y1 = self.pts_base[0]
        x2, y2 = self.pts_base[1]
        if x2 - x1 != 0:
            self.m_base = (y2 - y1) / (x2 - x1)
            self.c_base = y1 - self.m_base * x1
        else:
            self.m_base = 0
            self.c_base = y1 # Fallback para horizontal se os pontos forem iguais no X

        cv2.destroyAllWindows()
        cap_config.release()

        # --- 3. Inicialização do Processamento ---
        self.cap_vel = cv2.VideoCapture(path)
        self.current_mode = "DROP"
        
        self.params.update({
            "min_area": 300,
            "fps_real": self.params.get("fps_real", 10000)
        })
        
        self.data = {
            "history_pos": [], 
            "diam_pre": 1.0, 
            "tempos": [], 
            "posicoes": [], 
            "areas_queda": [],
            "last_diam_val": None,
            "max_w_px": 0
        }
        
        self.frame_id = 0
        self.impacto_detectado = False
        self.vel_active = True
        
        if hasattr(self, 'lbl_status'):
            self.lbl_status.configure(text="Baseline defined. Starting Fourth Method analysis...")
        self.analysis_active = True
        self.vel_active = True
        
        if hasattr(self, 'lbl_status'):
            self.lbl_status.configure(text="Análise de Impacto em execução...")    
        self.processar_frame_impacto_quatro()

    def processar_frame_impacto_quatro(self):
        """Loop de processamento frame a frame sem ROI coletando o diâmetro na base."""
        if not self.vel_active or self.cap_vel is None: return
        if not getattr(self, 'analysis_active', False) or not getattr(self, 'vel_active', False) or self.cap_vel is None: 
            return
        p, d = self.params, self.data
        ret, frame = self.cap_vel.read()
        
        if not ret:
            if len(d["posicoes"]) > 2:
                tempos = np.array(d["tempos"])
                pos_m = np.array(d["posicoes"]) / 1000.0
                v_impacto, _ = np.polyfit(tempos, pos_m, 1)
                v_abs = abs(v_impacto)
                
                d_medio_mm = d["diam_pre"] if d["diam_pre"] > 0 else 1.0
                d_m = d_medio_mm / 1000.0
                
                re = (p["densidade"] * v_abs * d_m) / p["viscosidade"]
                we = (p["densidade"] * (v_abs**2) * d_m) / p["tensao"]
                oh = p["viscosidade"] / np.sqrt(p["densidade"] * p["tensao"] * d_m)
                area_med_px2 = sum(d["areas_queda"]) / len(d["areas_queda"]) if d["areas_queda"] else 0
                
                self.mostrar_resultados_adimensionais(v_abs, d_medio_mm, re, we, oh, area_med_px2)

            self.cap_vel.release()
            self.vel_active = False
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if hasattr(self, 'background_gray') and self.background_gray is not None:
            if gray.shape == self.background_gray.shape:
                diff = cv2.absdiff(gray, self.background_gray)
                clean = cv2.bilateralFilter(diff, 5, 75, 75)
                _, mask = cv2.threshold(clean, 15, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                clean = cv2.bilateralFilter(frame, 5, 75, 75)
                gray_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            clean = cv2.bilateralFilter(frame, 5, 75, 75)
            gray_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # --- CORREÇÃO: Remove objetos abaixo da linha inclinada gerando uma máscara dinâmica ---
        h_img, w_img = frame.shape[:2]
        for x in range(w_img):
            y_limite = int(self.m_base * x + self.c_base)
            if 0 <= y_limite < h_img:
                mask[y_limite:, x] = 0

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- CORREÇÃO: Desenhar a linha inclinada estendida de referência ---
        pt_inicio = (0, int(self.c_base))
        pt_fim = (w_img, int(self.m_base * w_img + self.c_base))
        cv2.line(frame, pt_inicio, pt_fim, (0, 255, 255), 1)

        if cnts:
            c = min(cnts, key=lambda ct: abs(cv2.moments(ct)["m10"]/(cv2.moments(ct)["m00"]+1e-5) - w_img/2))
            area_atual_px = cv2.contourArea(c) 
            
            if area_atual_px > p["min_area"]:
                _, y_topo_box, _, h_box = cv2.boundingRect(c)
                y_base_gota_box = y_topo_box + h_box
                
                hull = cv2.convexHull(c)
                
                # Para saber a distância até a reta inclinada no centro do bounding box
                x_centro_box = _ + (_ + h_box)//2 # Aproximação rápida do centro X
                y_limite_centro = self.m_base * x_centro_box + self.c_base

                # 1. --- FASE ANTES DO TOQUE (QUEDA LIVRE) ---
                if not self.impacto_detectado:
                    d["areas_queda"].append(area_atual_px)
                    d["tempos"].append(self.frame_id * p["dt"])
                    d["posicoes"].append(y_topo_box * p["escala"])
                    
                    if len(c) >= 5:
                        ellipse = cv2.fitEllipse(c)
                        d["diam_pre"] = ((ellipse[1][0] + ellipse[1][1]) / 2) * p["escala"]
                        cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
                    
                    # Se aproximou da reta inclinada, muda o estado
                    if abs(y_base_gota_box - y_limite_centro) < 5:
                        self.impacto_detectado = True
                        
                # 2. --- FASE DE IMPACTO E ESPALHAMENTO ---
                else:
                    # --- CORREÇÃO: Filtra pontos que estão próximos à reta inclinada (distância vertical <= 3px) ---
                    pts_na_base = []
                    for pt in c:
                        px, py = pt[0][0], pt[0][1]
                        y_esperado = self.m_base * px + self.c_base
                        if abs(py - y_esperado) <= 3:
                            pts_na_base.append((px, int(y_esperado)))
                    
                    if len(pts_na_base) >= 2:
                        # Ordena pelo eixo X para pegar as extremidades esquerda e direita
                        pts_na_base = sorted(pts_na_base, key=lambda pt: pt[0])
                        p_esq = pts_na_base[0]
                        p_dir = pts_na_base[-1]
                        
                        # Distância Euclidiana real entre os dois pontos da reta inclinada
                        diam_pixel = np.sqrt((p_dir[0] - p_esq[0])**2 + (p_dir[1] - p_esq[1])**2)
                        
                        diam_mm = diam_pixel * p["escala"]
                        beta = diam_mm / d["diam_pre"] if d["diam_pre"] > 0 else 0
                        area_mm2 = area_atual_px * (p["escala"]**2)
                        tempo_atual = self.frame_id * p["dt"]

                        # Altura Vertical Intrínseca (Distância do topo até a reta base)
                        p_topo = tuple(hull[hull[:, :, 1].argmin()][0])
                        y_base_no_x_topo = self.m_base * p_topo[0] + self.c_base
                        altura_pixel = abs(y_base_no_x_topo - p_topo[1])
                        altura_mm = altura_pixel * p["escala"]

                        last_d = d.get("last_diam_val")
                        dd_dt = (diam_mm - last_d) / p["dt"] if last_d is not None else 0.0
                        d["last_diam_val"] = diam_mm 

                        if diam_pixel > d["max_w_px"]:
                            d["max_w_px"] = diam_pixel

                        # Desenhos dinâmicos ajustados para inclinação
                        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                        cv2.line(frame, p_esq, p_dir, (0, 0, 255), 2)  # Linha de contato real (Vermelha)
                        cv2.line(frame, p_topo, (p_topo[0], int(y_base_no_x_topo)), (255, 0, 255), 2) # Altura perpendicular/vertical (Magenta)
                        
                        self.tree.insert("", "end", values=(
                            f"{tempo_atual:.5f}", 
                            "-", 
                            f"{diam_mm:.3f}", 
                            f"{beta:.3f}", 
                            f"{altura_mm:.3f}", 
                            f"{area_mm2:.3f}", 
                            f"{dd_dt:.3f}"
                        ))
                    else:
                        diam_mm = 0.0
                        beta = 0.0
                        tempo_atual = self.frame_id * p["dt"]
                        d["last_diam_val"] = None

                        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 1)
                        
                        self.tree.insert("", "end", values=(
                            f"{tempo_atual:.5f}", 
                            "-", 
                            "0.000", 
                            "0.000", 
                            "-", 
                            "-", 
                            "-"
                        ))

        
        self.display_frame(frame, update_original=True)
        self.frame_id += 1
        self.after(1, self.processar_frame_impacto_quatro)
    
    def configurar_parametros_velocidade(self):
        """
        Opens a window to configure technical parameters before analysis.
        Labels are now in full English.
        """
        # Mapeamento de chaves técnicas para nomes por extenso em inglês
        display_names = {
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
            "tensao": "Surface Tension (N/m)"
        }

        # Valores padrão iniciais
        default_values = {
            "fps_real": 10000.0, "escala": 0.01, "area_min_queda": 20.0,
            "area_min_pos": 50.0, "dt": 1/10000.0, "frames_antes": 3, 
            "tolerancia": 3.0, "gray_thresh": 5.0, "densidade": 1000.0, 
            "viscosidade": 0.001, "tensao": 0.072
        }

        dialog = tk.Toplevel(self)
        dialog.title("Analysis Parameters Configuration")
        dialog.geometry("450x500")
        dialog.grab_set()

        entries = {}
        for i, (key, value) in enumerate(default_values.items()):
            # Usa o nome por extenso do dicionário display_names
            tk.Label(dialog, text=f"{display_names[key]}:").grid(row=i, column=0, padx=20, pady=5, sticky="e")
            entry = tk.Entry(dialog)
            entry.insert(0, str(value))
            entry.grid(row=i, column=1, padx=20, pady=5)
            entries[key] = entry

        params = {}

        def confirmar():
            try:
                for key in default_values:
                    params[key] = float(entries[key].get())
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values.")

        tk.Button(dialog, text="Confirm and Continue", command=confirmar, height=2, width=20).grid(row=len(default_values), columnspan=2, pady=25)
        
        self.wait_window(dialog)
        
        # Se a janela for fechada sem confirmar, retorna os valores padrão
        return params if params else default_values

    def mostrar_resultados_adimensionais(self, v, d_medio, re, we, oh, area_pre):
        """
        Exibe uma janela com os resultados dos cálculos físicos.
        """
        result_win = tk.Toplevel(self)
        result_win.title("Physical Analysis Results")
        result_win.geometry("350x380")
        
        # --- CORREÇÃO: Aguarda a janela renderizar antes de capturar o foco ---
        result_win.update_idletasks()     # Força o processamento de eventos pendentes na interface
        result_win.wait_visibility()      # Trava a execução até a janela estar fisicamente visível
        result_win.grab_set()             # Agora sim, captura o foco com segurança
        # ------------------------------------------------------------------------

        results = [
            ("Impact Velocity (v):", f"{v:.4f} m/s"),
            ("Mean Pre-impact Diameter (D):", f"{d_medio:.4f} mm"),
            ("Reynolds Number (Re):", f"{re:.2f}"),
            ("Weber Number (We):", f"{we:.2f}"),
            ("Ohnesorge Number (Oh):", f"{oh:.4f}"),
            ("Mean Pre-Impact Area:", f"{area_pre:.2f} mm²\n")
        ]

        for i, (label, val) in enumerate(results):
            tk.Label(result_win, text=label, font=("Arial", 10, "bold")).grid(row=i, column=0, padx=20, pady=10, sticky="e")
            tk.Label(result_win, text=val, font=("Arial", 10)).grid(row=i, column=1, padx=20, pady=10, sticky="w")

        tk.Button(result_win, text="Close", command=result_win.destroy, width=15).grid(row=6, columnspan=2, pady=15)
    def selecionar_chao_manual(self, path_video):
        """
        Abre uma janela para selecionar o frame (A/D) e definir a linha de base inclinada (2 cliques).
        """
        cap = cv2.VideoCapture(path_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_atual_idx = 0
        pontos = [] 
        line_params = {"p1": (0, 300), "p2": (1000, 300)} 
        nome_janela = "A/D: Frame | 2 Clicks: Define Inclined Line | ENTER: Confirm"

        def get_frame(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            return frame if ret else None

        frame_exemplo = get_frame(frame_atual_idx)
        temp_img = frame_exemplo.copy()

        def click_event(event, x, y, flags, param):
            nonlocal temp_img, pontos
            if event == cv2.EVENT_LBUTTONDOWN:
                pontos.append((x, y))
                cv2.circle(temp_img, (x, y), 4, (0, 255, 0), -1)
                if len(pontos) == 2:
                    temp_img = frame_exemplo.copy()
                    cv2.line(temp_img, pontos[0], pontos[1], (0, 0, 255), 2)
                    line_params["p1"], line_params["p2"] = pontos[0], pontos[1]
                    pontos = [] 
                cv2.imshow(nome_janela, temp_img)

        cv2.namedWindow(nome_janela)
        cv2.setMouseCallback(nome_janela, click_event)
        
        while True:
            display_img = temp_img.copy()
            cv2.putText(display_img, f"Frame: {frame_atual_idx}/{total_frames-1}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(nome_janela, display_img)
            key = cv2.waitKey(0) & 0xFF
            if key in [ord('d'), ord('D'), ord('a'), ord('A')]:
                if key in [ord('d'), ord('D')] and frame_atual_idx < total_frames - 1:
                    frame_atual_idx += 1
                elif key in [ord('a'), ord('A')] and frame_atual_idx > 0:
                    frame_atual_idx -= 1
                frame_exemplo = get_frame(frame_atual_idx)
                temp_img = frame_exemplo.copy()
                cv2.line(temp_img, line_params["p1"], line_params["p2"], (0, 0, 255), 2)
            elif key == 13 or key == 27 or cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cap.release()
        cv2.destroyWindow(nome_janela)
        return line_params

    def run_analise_velocidade(self):
        metodo_atual = "DROP_VELOCIDADE" 

        if self.ultimo_metodo_selecionado != metodo_atual:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.ultimo_metodo_selecionado = metodo_atual
        
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("Todos", "*.*")])
        if not path: return
        self.video_path = path
        # Janela de parâmetros em inglês
        self.vel_params = self.configurar_parametros_velocidade()

        self.cap_vel = cv2.VideoCapture(path)
        if not self.cap_vel.isOpened(): return

        lp = self.selecionar_chao_manual(path)
        self.cap_vel.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if (lp["p2"][0] - lp["p1"][0]) != 0:
            incl_a = (lp["p2"][1] - lp["p1"][1]) / (lp["p2"][0] - lp["p1"][0])
            incl_b = lp["p1"][1] - incl_a * lp["p1"][0]
        else:
            incl_a = 0
            incl_b = lp["p1"][1]

        self.current_mode = "DROP"
        self.vel_params.update({"line_a": incl_a, "line_b": incl_b})

        self.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": [], "areas_queda": []}
        self.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0, "last_diam_val": None}
        self.vel_active = True
        self.lbl_status.configure(text=f"Analyzing Speed")
        self.processar_frame_velocidade()

    def segmenta_gota_velocidade(self, diff_frame):
        p = self.vel_params
        blur = cv2.GaussianBlur(diff_frame, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        h, w = edges.shape
        for x in range(w):
            y_limit = int(p["line_a"] * x + p["line_b"])
            if y_limit < h:
                edges[y_limit:, x] = 0
        return edges

    def processar_frame_velocidade(self):
        if not self.vel_active or self.cap_vel is None: return
        ret, frame = self.cap_vel.read()
        
        if not ret:
            # --- CÁLCULOS FÍSICOS AO FINAL DO VÍDEO ---
            p, d = self.vel_params, self.vel_data
            if len(d["posicoes"]) > 2:
                # v = derivada da posição (m/s). Escala em mm converte para metros (/1000)
                # Ajuste linear: pos_m = v * tempo + b
                tempos = np.array(d["tempos"])
                pos_m = (np.array(d["posicoes"])) / 1000.0 
                v_impacto, _ = np.polyfit(tempos, pos_m, 1)
                
                d_medio_mm = np.mean(d["pre_impacto"]) if d["pre_impacto"] else 1.0
                d_m = d_medio_mm / 1000.0
                
                # Reynolds: (rho * v * D) / mu
                re = (p["densidade"] * abs(v_impacto) * d_m) / p["viscosidade"]
                # Weber: (rho * v^2 * D) / sigma
                we = (p["densidade"] * (v_impacto**2) * d_m) / p["tensao"]
                # Ohnesorge: mu / sqrt(rho * sigma * D)
                oh = p["viscosidade"] / np.sqrt(p["densidade"] * p["tensao"] * d_m)
                # --- NOVA LÓGICA DE ÁREA MÉDIA EM PX2 ---
                if d.get("areas_queda"):
                    area_media_px2 = sum(d["areas_queda"]) / len(d["areas_queda"])
                else:
                    area_media_px2 = 0.0
                
                self.mostrar_resultados_adimensionais(abs(v_impacto), d_medio_mm, re, we, oh, area_media_px2)

            self.cap_vel.release()
            self.vel_active = False
            self.lbl_status.configure(text="Drop Impact analysis completed.")
            return

        p, s, d = self.vel_params, self.vel_state, self.vel_data
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = gray.copy()
        gray_masked[gray_masked < p["gray_thresh"]] = 0
        
        if s["background"] is None:
            s["background"] = gray_masked.copy()
            s["frame_id"] += 1
            self.after(1, self.processar_frame_velocidade)
            return
            
        diff = cv2.absdiff(gray_masked, s["background"])
        y_medio = int(p["line_a"] * (frame.shape[1]/2) + p["line_b"])

        if not s["impacto"]:
            roi_limit = max(50, y_medio - 50)
            roi = diff[0:roi_limit, :]
            blur_roi = cv2.GaussianBlur(roi, (5, 5), 0)
            _, mask_queda = cv2.threshold(blur_roi, self.threshold_impact, 255, cv2.THRESH_BINARY) # self.threshold_impact=30
            cnts, _ = cv2.findContours(mask_queda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                area_px = cv2.contourArea(c)
                if cv2.contourArea(c) > p["area_min_queda"]:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    diam_mm = (2 * radius) * p["escala"]
                    
                    d["tempos"].append(s["frame_id"] * p["dt"])
                    d["posicoes"].append(y * p["escala"])
                    d["pre_impacto"].append(diam_mm)
                    d["areas_queda"].append(area_px)
                    
                    if len(d["pre_impacto"]) > p["frames_antes"]:
                        s["diametro_pre"] = d["pre_impacto"][-int(p["frames_antes"])]
                    
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    y_base_no_x = p["line_a"] * x + p["line_b"]
                    if y + radius >= y_base_no_x: s["impacto"] = True
        else:
            mask_pos = self.segmenta_gota_velocidade(diff)
            cnts_pos, _ = cv2.findContours(mask_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            candidatos = [c for c in cnts_pos if cv2.contourArea(c) > p["area_min_pos"]]
            
            if candidatos:
                c_gota = max(candidatos, key=cv2.contourArea)
                pts = c_gota.reshape(-1, 2)
                pts_suave = pts if len(pts) <= 21 else np.column_stack((savgol_filter(pts[:, 0], 21, 3), savgol_filter(pts[:, 1], 21, 3))).astype(np.int32)
                
                y_na_reta = p["line_a"] * pts_suave[:, 0] + p["line_b"]
                contatos = pts_suave[np.abs(pts_suave[:, 1] - y_na_reta) <= p["tolerancia"]]
                
                if len(contatos) > 2:
                    idx_esq, idx_dir = np.argmin(contatos[:, 0]), np.argmax(contatos[:, 0])
                    p_esq, p_dir = contatos[idx_esq], contatos[idx_dir]
                    dist_px = np.sqrt((p_dir[0]-p_esq[0])**2 + (p_dir[1]-p_esq[1])**2)
                    diam_mm = dist_px * p["escala"]
                    beta = diam_mm / s["diametro_pre"]
                    tempo_atual = s["frame_id"] * p["dt"]
                    # --- NOVAS MEDIÇÕES ---
                    # 1. Área Projetada (mm²)
                    area_px = cv2.contourArea(c_gota)
                    area_mm2 = area_px * (p["escala"]**2)
                    
                    a, b = p["line_a"], p["line_b"]
                    
                    # 1. Encontrar o ponto médio da linha de contato (centro do diâmetro)
                    # p_esq e p_dir já foram definidos no código anterior
                    x_centro = (p_esq[0] + p_dir[0]) / 2
                    y_centro = (p_esq[1] + p_dir[1]) / 2
                    ponto_base_centro = (int(x_centro), int(y_centro))

                    # 2. Encontrar o ponto no contorno da gota (pts_suave) que está mais próximo da coordenada X do centro
                    # Isso nos dá o topo da gota exatamente acima do centro do diâmetro
                    idx_topo_centro = np.argmin(np.abs(pts_suave[:, 0] - x_centro))
                    ponto_topo = pts_suave[idx_topo_centro]

                    # 3. Calcular a altura perpendicular nesse ponto central
                    # Distância entre o ponto_topo e o ponto_base_centro
                    dist_px_altura = np.sqrt((ponto_topo[0] - x_centro)**2 + (ponto_topo[1] - y_centro)**2)
                    altura_mm = dist_px_altura * p["escala"]
                    
                    # ----------------------------------------------------
                    
                    # --- DESENHO DA LINHA DA ALTURA NO CENTRO ---
                    # Desenha a linha do ponto do topo até o centro da base projetada
                    cv2.line(frame, tuple(ponto_topo), ponto_base_centro, (255, 0, 255), 2)
                    
                    # Opcional: marcador no ponto do topo para conferência
                    cv2.circle(frame, tuple(ponto_topo), 3, (255, 0, 255), -1)
                    
                    # 3. Taxa de variação do diâmetro (mm/s) -> dD/dt
                    last_d = s.get("last_diam_val")
                    dd_dt = (diam_mm - last_d) / p["dt"] if last_d is not None else 0.0
                    s["last_diam_val"] = diam_mm # Atualiza para o próximo frame
                    
                    # Inserção na tabela com os novos parâmetros
                    self.tree.insert("", "end", values=(
                        f"{tempo_atual:.4f}", 
                        "-", 
                        f"{diam_mm:.2f}", 
                        f"{beta:.2f}",
                        f"{altura_mm:.2f}",
                        f"{area_mm2:.2f}",
                        f"{dd_dt:.2f}"
                    ))
                    # ----------------------
                    #self.tree.insert("", "end", values=(f"{tempo_atual:.4f}", "-", f"{diam_mm:.2f}", f"{beta:.2f}"))
                    cv2.polylines(frame, [pts_suave.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                    cv2.line(frame, tuple(p_esq), tuple(p_dir), (0, 0, 255), 2)
        
        h, w = frame.shape[:2]
        p1_draw = (0, int(p["line_b"]))
        p2_draw = (w, int(p["line_a"] * w + p["line_b"]))
        cv2.line(frame, p1_draw, p2_draw, (255, 0, 0), 1)
        
        s["frame_id"] += 1
        self.display_frame(frame)
        self.after(1, self.processar_frame_velocidade)
    def run_velocity_analysis(self):
        # Define o método atual
        metodo_atual = "DROP_VELOCITY" 

        # Verifica se o método mudou
        if self.ultimo_metodo_selecionado != metodo_atual:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.ultimo_metodo_selecionado = metodo_atual
        
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("Todos", "*.*")])
        if not path: return
        self.video_path = path
        # --- Configuração via Janela e Seleção de Chão ---
        self.vel_params = self.configurar_parametros_velocidade()
        self.cap_vel = cv2.VideoCapture(path)
        if not self.cap_vel.isOpened(): return

        lp = self.selecionar_chao_manual(path)
        if (lp["p2"][0] - lp["p1"][0]) != 0:
            incl_a = (lp["p2"][1] - lp["p1"][1]) / (lp["p2"][0] - lp["p1"][0])
            incl_b = lp["p1"][1] - incl_a * lp["p1"][0]
        else:
            incl_a, incl_b = 0, lp["p1"][1]

        self.vel_params.update({"line_a": incl_a, "line_b": incl_b})
        
        # Estado inicial e dados
        self.current_mode = "DROP"
        self.vel_data = {
            "tempos": [], 
            "posicoes": [], 
            "diametros": [], 
            "pre_impacto": [],
            "areas_queda": []
        }
        self.vel_state = {
            "frame_id": 0, 
            "impacto": False, 
            "background": None, 
            "diametro_pre": 1.0,
            "last_diam_val": None
        }
        self.vel_active = True
        self.lbl_status.configure(text="Analyzing Speed and Impact...")
        self.processar_frame_velocidade_2()

    def segmenta_gota_velocidade_2(self, diff_frame):
        p = self.vel_params
        blur = cv2.GaussianBlur(diff_frame, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        kernel = np.ones((3, 3), np.uint8)
        # Erode primeiro para apagar respingos pequenos
        #edges = cv2.erode(edges, kernel, iterations=1)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Zera a área abaixo do limite do chão (usando a linha manual como referência)
        h_img, w_img = edges.shape
        y_corte = int(p["line_b"]) # Simplificação baseada na linha manual
        edges[y_corte:, :] = 0
        return edges

    def processar_frame_velocidade_2(self):
        if not self.vel_active or self.cap_vel is None: return
        ret, frame = self.cap_vel.read()
        
        if not ret:
            # --- CÁLCULOS FÍSICOS FINAIS (ADIMENSIONAIS) ---
            p, d = self.vel_params, self.vel_data
            
            if len(d["posicoes"]) > 2:
                tempos = np.array(d["tempos"])
                pos_m = (np.array(d["posicoes"])) / 1000.0 # Converte posição para metros
                
                # Velocidade via regressão linear (m/s)
                v_impacto, _ = np.polyfit(tempos, pos_m, 1)
                v_abs = abs(v_impacto)
                
                d_medio_mm = np.mean(d["pre_impacto"]) if d["pre_impacto"] else 1.0
                d_m = d_medio_mm / 1000.0
                
                # Números Adimensionais
                re = (p["densidade"] * v_abs * d_m) / p["viscosidade"]
                we = (p["densidade"] * (v_abs**2) * d_m) / p["tensao"]
                oh = p["viscosidade"] / np.sqrt(p["densidade"] * p["tensao"] * d_m)
                # --- CÁLCULO DA ÁREA MÉDIA ---

                if d["areas_queda"]: # Use o nome correto que você definiu no self.vel_data
                    area_media_px2 = sum(d["areas_queda"]) / len(d["areas_queda"])
                    # Se quiser em mm², multiplique pela escala ao quadrado:
                    #area_media_mm2 = area_media_px2 * (p["escala"]**2)
                else:
                    area_media_px2 = 0.0
                    #area_media_mm2 = 0.0
                self.mostrar_resultados_adimensionais(v_abs, d_medio_mm, re, we, oh, area_media_px2) #area_media_mm2
                
                

            self.cap_vel.release()
            self.vel_active = False
            self.lbl_status.configure(text="Análise de impacto concluída.")
            return

        p, s, d = self.vel_params, self.vel_state, self.vel_data
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if s["background"] is None:
            s["background"] = gray.copy()
            s["frame_id"] += 1
            self.after(1, self.processar_frame_velocidade_2)
            return

        diff = cv2.absdiff(gray, s["background"])
        
        if not s["impacto"]:
            # --- LÓGICA ANTES DO IMPACTO (QUEDA) ---
            roi_h = int(p["line_b"]) - 10 if "line_b" in p else 250
            roi = diff[0:roi_h, :]
            blur_roi = cv2.GaussianBlur(roi, (5, 5), 0)
            _, mask_queda = cv2.threshold(blur_roi, self.threshold_impact, 255, cv2.THRESH_BINARY) #self.threshold_impact =30
            
            cnts, _ = cv2.findContours(mask_queda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                # --- DEFINIÇÃO DA VARIÁVEL ---
                area_px = cv2.contourArea(c)
                if cv2.contourArea(c) > p["area_min_queda"]:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    diam_mm = (2 * radius) * p["escala"]
                    d["areas_queda"].append(area_px)
                    d["tempos"].append(s["frame_id"] * p["dt"])
                    d["posicoes"].append(y * p["escala"])
                    d["pre_impacto"].append(diam_mm)
                    
                    if len(d["pre_impacto"]) > p["frames_antes"]:
                        s["diametro_pre"] = d["pre_impacto"][-int(p["frames_antes"])]
                    
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    
                    # Checa impacto contra a linha manual
                    y_base = p["line_a"] * x + p["line_b"]
                    if y + radius >= y_base:
                        s["impacto"] = True
        else:
            # --- LÓGICA APÓS IMPACTO (ESPALHAMENTO) ---
            mask_pos = self.segmenta_gota_velocidade_2(diff)
            cnts_pos, _ = cv2.findContours(mask_pos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            candidatos = [c for c in cnts_pos if cv2.contourArea(c) > p["area_min_pos"]]
            
            if candidatos:
                c_maior = max(candidatos, key=cv2.contourArea)
                c_gota = cv2.convexHull(c_maior)
                
                # Medição via Bounding Rect (Mantido conforme solicitado)
                rx, ry, rw, rh = cv2.boundingRect(c_gota)
                diam_mm = rw * p["escala"]
                altura_mm = rh * p["escala"] # Implementação da Altura
                beta = diam_mm / s["diametro_pre"]
                
                # Área e Taxa de variação (dD/dt)
                area_mm2 = cv2.contourArea(c_gota) * (p["escala"]**2)
                last_d = s.get("last_diam_val")
                dd_dt = (diam_mm - last_d) / p["dt"] if last_d is not None else 0.0
                s["last_diam_val"] = diam_mm
                
                tempo_atual = s["frame_id"] * p["dt"]
                
                # Atualiza Interface com as novas variáveis
                self.tree.insert("", "end", values=(
                    f"{tempo_atual:.4f}", 
                    "-", 
                    f"{diam_mm:.2f}", 
                    f"{beta:.2f}", 
                    f"{altura_mm:.2f}", 
                    f"{area_mm2:.2f}", 
                    f"{dd_dt:.2f}"
                ))
                
                # Visualização
                cv2.drawContours(frame, [c_gota], -1, (0, 255, 0), 2)
                y_meio = int(ry + rh / 2)
                cv2.line(frame, (rx, y_meio), (rx + rw, y_meio), (0, 0, 255), 2) # Diâmetro
                cv2.line(frame, (int(rx + rw/2), ry), (int(rx + rw/2), ry + rh), (255, 0, 0), 2) # Altura
                
                cv2.putText(frame, f"Beta: {beta:.2f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        s["frame_id"] += 1
        self.display_frame(frame)
        self.after(1, self.processar_frame_velocidade_2)
    def run_analise_normal(self):
# Define o método atual (exemplo para a função run_analise_normal)

        metodo_atual = "WCA_NORMAL" 

        # Verifica se o método mudou
        if self.ultimo_metodo_selecionado != metodo_atual:
            # Se for um método diferente, limpa a tabela
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Atualiza o registro para o novo método
            self.ultimo_metodo_selecionado = metodo_atual
        
        # Se o método for o mesmo, ele ignora o 'if' e os dados novos serão 
        # acumulados com os anteriores via self.tree.insert
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg"), ("All", "*.*")])
        if not path: return
        self.video_path = path
        # --- EXTRAÇÃO DO NOME DO ARQUIVO ---
        nome_arquivo = os.path.basename(path) # Pega apenas 'foto.jpg' em vez do caminho todo
        img = cv2.imread(path)
        if img is None: return
        self.current_mode = "WCA"
        orig = img.copy()
        # --- LOOP DA JANELA DE ROI COM OPÇÃO DE RETORNO ---
        while True:
            # Abre a janela de seleção
            r = cv2.selectROI("Select ROI - [Enter] Confirm | [ESC] Cancel", img, showCrosshair=True)
            cv2.destroyWindow("Select ROI - [Enter] Confirm | [ESC] Cancel")
            
            x_roi, y_roi, w_roi, h_roi = r
            
            # Se o usuário apertar ESC ou fechar a janela sem selecionar, o 'w' ou 'h' será 0
            if w_roi == 0 or h_roi == 0:
                # Pergunta ao usuário se quer tentar selecionar novamente ou cancelar
                resposta = messagebox.askyesno("Invalid ROI", "No region was selected.\nDo you want to try selecting the ROI again?")
                if resposta:  # Se sim, o loop continua e abre a janela de novo
                    continue
                else:         # Se não, cancela a operação e retorna
                    return
            
            # Se chegou aqui, a ROI é válida. Vamos confirmar visualmente?
            # Se quiser dar a opção de refazer direto na imagem, fazemos um preview rápido:
            roi_preview = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi].copy()
            cv2.putText(roi_preview, "Confirm? [Y]-Yes | [R]-Redo ROI", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Confirm ROI", roi_preview)
            key = cv2.waitKey(0)
            cv2.destroyWindow("Confirm ROI")
            
            if key == ord('r') or key == ord('R'):
                continue # Volta para o início do 'while True' e abre o selectROI novamente
            else:
                break # Sai do loop e continua o código normalmente

        # --- FIM DO LOOP DE RETORNO ---
        roi = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        pontos_base = []
        roi_vis = roi.copy()
        cv2.imshow("Select Base", roi_vis)
        cv2.setMouseCallback("Select Base", selecionar_ponto_callback, param=[pontos_base, roi_vis, "Select Base"])
        while len(pontos_base) < 2:
            if cv2.waitKey(1) == 27: cv2.destroyAllWindows(); return
        cv2.destroyWindow("Select Base")
        p1, p2 = pontos_base
        y_base = max(p1[1], p2[1])
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 30, 150)
        #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours: return
        contorno = max(contours, key=cv2.contourArea).reshape(-1, 2)
        pontos = pontos_proximos_base(contorno, y_base, self.tangent_static) #self.tangent_static=altura=10 #ajustar aqui o ponto próximos para calcular a tangente
        # =========================
        # PASSO 3 — DEBUG VISUAL
         # =========================
        roi_debug = roi.copy()

        for (x, y) in pontos:
            cv2.circle(roi_debug, (x, y), 1, (255, 0, 0), -1)

        cv2.imshow("Selected Points", roi_debug)
        cv2.waitKey(0)
        cv2.destroyWindow("Selected Points")

# =========================
        esq, direi = separar_lados(pontos)
        esq, direi = esq[:40], direi[:40]
        try:
            ang_esq, slope_esq = calcular_angulo_dropy(esq, "esq")
            ang_dir, slope_dir = calcular_angulo_dropy(direi, "dir")
            ang_medio = (ang_esq + ang_dir) / 2
            cv2.drawContours(orig, [contorno + [x_roi, y_roi]], -1, (0, 255, 0), 1)
            # --- ADICIONE ESTA LINHA AQUI (A BASE VERMELHA) ---
            cv2.line(orig, (x_roi + p1[0], y_roi + p1[1]), (x_roi + p2[0], y_roi + p2[1]), (0, 0, 255), 2)
# --------------------------------------------------
            for p, slope in zip([p1, p2], [slope_esq, slope_dir]):
                dx = 40
                dy = int(slope * dx)
                cv2.line(orig, (x_roi+int(p[0]-dx), y_roi+int(p[1]-dy)), (x_roi+int(p[0]+dx), y_roi+int(p[1]+dy)), (0, 255, 255), 1)
            desenhar_arco_angulo(orig, (x_roi+p1[0], y_roi+p1[1]), ang_esq, "esq")
            desenhar_arco_angulo(orig, (x_roi+p2[0], y_roi+p2[1]), ang_dir, "dir")
            cv2.putText(orig, f"Left: {ang_esq:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(orig, f"Right: {ang_dir:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(orig, f"Average: {ang_medio:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            self.display_frame(orig)
            self.lbl_status.configure(text=f"Analysis Completed: Average {ang_medio:.2f}°")
            self.tree.insert("", "end", values=(nome_arquivo, f"{ang_esq:.2f}", f"{ang_dir:.2f}", f"{ang_medio:.2f}"))
        except Exception as e: messagebox.showerror("Erro", str(e))
    def desenhar_curva_ajuste(self, img, coef, pts_y, offset_x, offset_y, cor=(255, 255, 0)):
        """Desenha a parábola resultante do ajuste polinomial x = ay² + by + c"""
        # Criamos um intervalo de Y para desenhar a curva (do ponto de contato para cima)
        y_min = int(min(pts_y))
        y_max = int(max(pts_y))
        
        curva_pts = []
        for y_val in range(y_min, y_max + 1):
            # Calcula x usando a fórmula do polinômio: x = ay² + by + c
            x_val = coef[0] * (y_val**2) + coef[1] * y_val + coef[2]
            curva_pts.append([int(x_val + offset_x), int(y_val + offset_y)])
        
        if len(curva_pts) > 1:
            pts_array = np.array(curva_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_array], False, cor, 1)
    def desenhar_tangente(self, img, ponto_base, slope_dx_dy, lado, cor=(255, 0, 255)):
        x0, y0 = ponto_base
        length = 50 # Tamanho da linha visual
        
        # O slope que calculamos no polyfit é dx/dy (horizontal/vertical)
        # Para desenhar no OpenCV, precisamos converter isso em coordenadas (x, y)
        if lado == "esq":
            # No lado esquerdo, queremos que a linha suba e vá para a esquerda
            dy = -length 
            dx = dy * slope_dx_dy
        else:
            # No lado direito, queremos que a linha suba e vá para a direita
            dy = -length
            dx = dy * slope_dx_dy

        p2 = (int(x0 + dx), int(y0 + dy))
        cv2.line(img, (int(x0), int(y0)), p2, cor, 1)

    def desenhar_arco_angulo(self, img, centro, angulo_valor, lado, cor=(0, 255, 0)):
        """Desenha um arco representando o ângulo de contato."""
        raio = 30  # Tamanho do arco
        espessura = 1
        
        # No OpenCV, os ângulos para a elipse começam no eixo X positivo (0°)
        # e seguem o sentido horário.
        if lado == "esq":
            # Lado esquerdo: o arco vai de 0° até -angulo (sentido anti-horário)
            start_angle = 0
            end_angle = -angulo_valor
        else:
            # Lado direito: o arco vai de 180° até 180 + angulo
            start_angle = 180
            end_angle = 180 + angulo_valor

        cv2.ellipse(img, centro, (raio, raio), 0, start_angle, end_angle, cor, espessura)
    def run_analise_avancada(self):
        metodo_atual = "WCA_AVANCADO"

        if self.ultimo_metodo_selecionado != metodo_atual:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.ultimo_metodo_selecionado = metodo_atual

        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.png *.jpeg"), ("All", "*.*")]
        )
        if not path:
            return
        self.video_path = path
        nome_arquivo = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            return

        orig = img.copy()

        # -----------------------------
        # ROI
        # -----------------------------
        r = cv2.selectROI("Selecione ROI", img, showCrosshair=True)
        cv2.destroyWindow("Selecione ROI")
        x_roi, y_roi, w_roi, h_roi = r
        if w_roi == 0 or h_roi == 0:
            return

        roi = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

        # -----------------------------
        # BASE
        # -----------------------------
        pontos_base = []
        roi_vis = roi.copy()
        cv2.imshow("Selecione Base", roi_vis)
        cv2.setMouseCallback("Selecione Base", selecionar_ponto_callback,
                             param=[pontos_base, roi_vis, "Selecione Base"])

        while len(pontos_base) < 2:
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                return

        cv2.destroyWindow("Selecione Base")

        p1, p2 = pontos_base
        y_base = max(p1[1], p2[1])

        # -----------------------------
        # PRÉ-PROCESSAMENTO (AGORA ADAPTATIVO)
        # -----------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calcula limiares automáticos baseados na mediana da imagem
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)

        # -----------------------------
        # CONTORNO (CORRIGIDO CONTRA REFLEXOS)
        # -----------------------------
        # RETR_EXTERNAL teoricamente ignora o que está dentro, 
        # mas reflexos muito fortes conectados à borda podem burlar isso.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return

        # Nova estratégia: Filtra contornos que tenham tamanho mínimo 
        # e escolhe aquele com a maior amplitude no eixo X (largura real da gota)
        contornos_validos = []
        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) > 50: # descarta pequenos ruídos
                largura = np.max(pts[:, 0]) - np.min(pts[:, 0])
                contornos_validos.append((largura, pts))

        if not contornos_validos:
            return

        # Seleciona o contorno com maior largura horizontal (a silhueta da gota)
        contorno = max(contornos_validos, key=lambda item: item[0])[1]

        # -----------------------------
        # FILTRAR REGIÃO PRÓXIMA DA BASE (DINÂMICO)
        # -----------------------------
        # Começa com o valor configurado pelo sistema
        tolerancia_altura = self.tangent_static_pol
        minimo_pontos_obrigatorio = max(10, self.tangent_static_base_points)
        
        # LOOP DINÂMICO: Se não achar pontos, ele alarga a janela de busca até 3 vezes mais
        for tentativa in range(3):
            lista_pontos = [p for p in contorno if abs(p[1] - y_base) < tolerancia_altura]
            
            # Se achamos pontos suficientes, podemos sair do loop
            if len(lista_pontos) >= minimo_pontos_obrigatorio:
                break
                
            # Se não achou, aumenta a tolerância vertical em 50% para a próxima tentativa
            tolerancia_altura = int(tolerancia_altura * 1.5)

        # Validação final de segurança
        if len(lista_pontos) < minimo_pontos_obrigatorio:
            messagebox.showerror(
                "Erro de Detecção", 
                f"Poucos pontos próximos à base ({len(lista_pontos)} encontrados).\n\n"
                "Dicas:\n"
                "1. Melhore o contraste/iluminação da foto.\n"
                "2. Selecione a base exatamente onde a gota toca a superfície.\n"
                "3. Aumente o valor de 'tangent_static_pol' nas configurações."
            )
            return

        # Conversão segura para o array NumPy 2D
        pontos = np.array(lista_pontos)
        
        # O resto do código (x_med = np.mean...) segue igual daqui para baixo
        x_med = np.mean(pontos[:, 0])
        esq = pontos[pontos[:, 0] < x_med]
        dirr = pontos[pontos[:, 0] >= x_med]

        esq = esq[np.argsort(esq[:, 1])]
        dirr = dirr[np.argsort(dirr[:, 1])]

        def suavizar(p):
            if len(p) < 11: return p
            x = savgol_filter(p[:, 0], 11, 3)
            y = savgol_filter(p[:, 1], 11, 3)
            return np.column_stack((x, y))

        esq = suavizar(esq)
        dirr = suavizar(dirr)

        # -----------------------------
        # AJUSTE LOCAL (POLINÔMIO)
        # -----------------------------
        def calcular_angulo(pontos, lado):
            n = min(30, len(pontos))
            pts = pontos[np.argsort(pontos[:, 1])][:n]
            x = pts[:, 0]
            y = pts[:, 1]

            coef = np.polyfit(y, x, 2)
            y_contato = np.max(y)
            x_contato = coef[0] * (y_contato**2) + coef[1] * y_contato + coef[2]
            dx_dy = 2 * coef[0] * y_contato + coef[1]

            if dx_dy != 0:
                slope_v = 1 / abs(dx_dy)
                ang_base = np.degrees(np.arctan(slope_v))
            else:
                ang_base = 90.0

            if lado == "esq":
                ang = 180 - ang_base if dx_dy > 0 else ang_base
            else:
                ang = 180 - ang_base if dx_dy < 0 else ang_base
            return ang, dx_dy, (x_contato, y_contato), coef, y

        ang_esq, slope_esq, pt_esq, coef_esq, y_esq = calcular_angulo(esq, "esq")
        ang_dir, slope_dir, pt_dir, coef_dir, y_dir = calcular_angulo(dirr, "dir")
        ang_medio = (ang_esq + ang_dir) / 2

        # -----------------------------
        # DESENHO DAS TANGENTES E RESULTADOS
        # -----------------------------
        cv2.drawContours(orig, [contorno + [x_roi, y_roi]], -1, (0, 255, 0), 1)
        cv2.line(orig, (x_roi+p1[0], y_roi+p1[1]), (x_roi+p2[0], y_roi+p2[1]), (0, 0, 255), 1)
        
        self.desenhar_curva_ajuste(orig, coef_esq, y_esq, x_roi, y_roi, (255, 255, 0))
        self.desenhar_curva_ajuste(orig, coef_dir, y_dir, x_roi, y_roi, (255, 255, 0))
        
        pt_esq_global = (pt_esq[0] + x_roi, pt_esq[1] + y_roi)
        pt_dir_global = (pt_dir[0] + x_roi, pt_dir[1] + y_roi)
        
        ponto_base_esq = (int(pt_esq[0] + x_roi), int(pt_esq[1] + y_roi))
        ponto_base_dir = (int(pt_dir[0] + x_roi), int(pt_dir[1] + y_roi))

        self.desenhar_arco_angulo(orig, ponto_base_esq, ang_esq, "esq")
        self.desenhar_arco_angulo(orig, ponto_base_dir, ang_dir, "dir")  

        self.desenhar_tangente(orig, ponto_base_esq, slope_esq, "esq")
        self.desenhar_tangente(orig, ponto_base_dir, slope_dir, "dir")
 
        cv2.putText(orig, f"Left: {ang_esq:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        cv2.putText(orig, f"Right: {ang_dir:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
        cv2.putText(orig, f"Average: {ang_medio:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # -----------------------------
        # NOVO: DEBUG VISUAL DO PIPELINE
        # -----------------------------
        # 1. Cria uma versão colorida das bordas (Canny) para empilhar lado a lado
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Desenha a linha de corte da base nas bordas para conferência visual
        cv2.line(edges_bgr, (p1[0], int(y_base)), (p2[0], int(y_base)), (0, 0, 255), 2)
        
        # 2. Recorta a região correspondente na imagem final 'orig' para comparar lado a lado
        roi_final_desenhada = orig[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
        
        # 3. Junta as duas visões da ROI (Bordas Detectadas vs Resultado do Ajuste)
        debug_horizontal = np.hstack((edges_bgr, roi_final_desenhada))
        
        # Adiciona instruções na tela de debug
        cv2.putText(debug_horizontal, "[ENTER]: Confirmar | [ESC]: Cancelar", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 4. Mostra a janela de debug e aguarda ação do usuário
        cv2.imshow("Debug Visual - Pipeline de Analise", debug_horizontal)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("Debug Visual - Pipeline de Analise")

        # Se o usuário apertar ESC (27), cancela e não salva na tabela
        if key == 27:
            return

        # -----------------------------
        # FINALIZAÇÃO
        # -----------------------------
        self.display_frame(orig)

        self.tree.insert("", "end", values=(nome_arquivo, f"{ang_esq:.2f}", f"{ang_dir:.2f}", f"{ang_medio:.2f}", "N/A"))
    
    def selecionar_tres_pontos(self, img_roi):
        """
        Permite selecionar 3 pontos com linhas elásticas em tempo real.
        Retorna uma lista com os 3 pontos [(x1,y1), (x2,y2), (x3,y3)]
        """
        pontos = []
        temp_img = img_roi.copy()
        window_name = "Select 3 Points (Base 1, Base 2, Angle)"
        
        # Estado para rastrear a posição do mouse para a linha elástica
        mouse_pos = [0, 0]

        def callback_clique(event, x, y, flags, param):
            nonlocal mouse_pos
            if event == cv2.EVENT_LBUTTONDOWN:
                pontos.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE:
                mouse_pos[0], mouse_pos[1] = x, y

        cv2.imshow(window_name, temp_img)
        cv2.setMouseCallback(window_name, callback_clique)

        while len(pontos) < 3:
            viz_img = temp_img.copy()
            
            if len(pontos) > 0:
                # Desenha círculos nos pontos já confirmados
                for p in pontos:
                    cv2.circle(viz_img, p, 1, (0, 0, 255), -1)
                
                # Lógica das linhas elásticas (amarelo)
                if len(pontos) == 1:
                    # Linha do 1º ponto até o cursor do mouse
                    cv2.line(viz_img, pontos[0], (mouse_pos[0], mouse_pos[1]), (0, 255, 255), 1)
                
                elif len(pontos) == 2:
                    # Linha da base já fixa (vermelha)
                    cv2.line(viz_img, pontos[0], pontos[1], (0, 0, 255), 2)
                    # Linha do 2º ponto (vértice) até o cursor do mouse
                    cv2.line(viz_img, pontos[1], (mouse_pos[0], mouse_pos[1]), (0, 255, 255), 1)

            cv2.imshow(window_name, viz_img)
            # Tecla ESC para cancelar a operação
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyWindow(window_name)
        return pontos if len(pontos) == 3 else None
    
    def run_analise_manual(self):
        """
        Executa a análise baseada na seleção manual de 3 pontos.
        """
        # 1. Primeiro, abre o seletor de arquivos e carrega a imagem
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.png *.jpeg"), ("All", "*.*")]
        )
        if not path:
            return
        self.video_path = path
        nome_arquivo = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            return

        orig = img.copy()

        # 2. Seleção da ROI (Necessário para definir a variável 'roi')
        r = cv2.selectROI("Select the Droplet Region", img, showCrosshair=True)
        cv2.destroyWindow("Select the Droplet Region")
        
        x_roi, y_roi, w_roi, h_roi = r
        if w_roi == 0 or h_roi == 0:
            return

        # DEFINIÇÃO DA VARIÁVEL ROI (O que estava faltando)
        roi = img[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

        # 3. Chama o método de seleção de pontos passando a roi definida acima
        pts = self.selecionar_tres_pontos(roi)
        
        if pts:
            p1, p2, p3 = pts # P2 é o vértice comum onde o ângulo será medido
            
            # Cálculo de vetores a partir do vértice (P2)
            v_base = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v_face = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Normalização e cálculo do ângulo via Produto Escalar
            norm_b = np.linalg.norm(v_base)
            norm_f = np.linalg.norm(v_face)
            
            if norm_b > 0 and norm_f > 0:
                cos_theta = np.dot(v_base, v_face) / (norm_b * norm_f)
                angulo_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angulo_deg = np.degrees(angulo_rad)
                
                # --- Lógica de Identificação do Lado ---
                # Se o vértice (p2) tem X menor que o ponto da base (p1), é o lado esquerdo
                # Se o vértice (p2) tem X maior que o ponto da base (p1), é o lado direito
                ang_esq = "N/A"
                ang_dir = "N/A"
                ang_medio = "N/A"

                if p2[0] < p1[0]:
                    ang_esq = angulo_deg
                    label_lado = "Esq"
                else:
                    ang_dir = angulo_deg
                    label_lado = "Dir"
                # Inserção na Tabela (Treeview)
                # Nota: ang_medio aqui é setado como N/A ou o valor individual, 
                # pois a medida manual é feita um lado por vez.
                self.tree.insert("", "end", values=(
                    nome_arquivo, 
                    f"{ang_esq:.2f}" if isinstance(ang_esq, float) else ang_esq, 
                    f"{ang_dir:.2f}" if isinstance(ang_dir, float) else ang_dir, 
                    "Manual"
                ))
                # Ajuste de coordenadas para desenho na imagem original
                p1_g = (p1[0] + x_roi, p1[1] + y_roi)
                p2_g = (p2[0] + x_roi, p2[1] + y_roi)
                p3_g = (p3[0] + x_roi, p3[1] + y_roi)
                
                # Desenho das linhas finais
                cv2.line(orig, p1_g, p2_g, (0, 0, 255), 2)      # Base
                cv2.line(orig, p2_g, p3_g, (255, 0, 255), 2)    # Tangente
                
                # Inserção do texto
                texto = f"Ang: {angulo_deg:.2f} deg"
                cv2.putText(orig, texto, (p2_g[0] - 20, p2_g[1] - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Atualiza a interface
                self.display_frame(orig)
    
    def run_histerese_polinomial(self):
        """
        Executa a análise de histerese com ajuste polinomial, 
        usando navegação por frames para configurar o início.
        """
        # 1. Configuração inicial do método e tabela
        metodo_atual = "HIS_POLINOMIAL"
        if self.ultimo_metodo_selecionado != metodo_atual:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.ultimo_metodo_selecionado = metodo_atual

        # 2. Seleção do arquivo
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("All", "*.*")])
        if not path: return
        self.video_path = path
        self.current_mode = "HIS"
        self.cap = cv2.VideoCapture(path) # Usando self.cap para manter o padrão
        self.histerese_pts_base = []
        self.histerese_frame_base = None
        
        idx = 0
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. Navegação de Frames (A/D)
        win_frame = "Config: Frame"
        cv2.namedWindow(win_frame, cv2.WINDOW_NORMAL)
        
        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret: break
            
            vis = frame.copy()
            cv2.putText(vis, f"Frame: {idx}/{total-1} - A/D navegar, ENTER selecionar.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win_frame, vis)
            
            k = cv2.waitKey(0) & 0xFF
            if k == ord('d'): idx = min(idx + 1, total - 1)
            elif k == ord('a'): idx = max(idx - 1, 0)
            elif k == 13: # ENTER
                self.histerese_frame_base = frame.copy()
                break
            elif k == 27: # ESC para cancelar
                cv2.destroyWindow(win_frame)
                return
                
        cv2.destroyWindow(win_frame)

        # 4. Seleção da Base por Cliques
        def click_ev(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                self.histerese_pts_base.append((x, y))
                cv2.circle(self.histerese_frame_base, (x, y), 1, (0, 0, 255), -1)
                cv2.imshow("Config: Base", self.histerese_frame_base)

        win_base = "Config: Base"
        cv2.namedWindow(win_base, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_base, click_ev)
        cv2.imshow(win_base, self.histerese_frame_base)

        while len(self.histerese_pts_base) < 2:
            if cv2.waitKey(1) == 27: 
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

        # 5. Finalização da Configuração e Início do Loop
        # Definimos a base como a maior coordenada Y selecionada
        self.histerese_y_base = max(self.histerese_pts_base[0][1], self.histerese_pts_base[1][1])
        
        # Reseta o vídeo para o frame 0 (ou para o idx selecionado, se preferir)
        #self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       # self.analysis_active = True
        #self.lbl_status.configure(text="Processando Histerese Polinomial...")
        
        # Chama o processamento frame a frame
       # self.processar_frame_histerese_polinomial()
        # ... (final do método run_histerese_polinomial antes de chamar o processamento)
        # --- INSERIR AQUI: CAPTURA DO BACKGROUND ---
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Volta ao início para pegar o fundo
        ret_bg, frame_bg = self.cap.read()
        if ret_bg:
            # Prepara o fundo: cinza e desfoque
            bg_gray = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2GRAY)
            self.background_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
        # ------------------------------------------

        # Reseta o vídeo para o frame 0 novamente para começar a análise real
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.analysis_active = True
        self.lbl_status.configure(text="Processing Polynomial Hysteresis...")
        
        # Cria a janela para o vídeo binário (Opcional: WINDOW_NORMAL permite redimensionar)
        cv2.namedWindow("Analysis: Binary Video", cv2.WINDOW_NORMAL)
        # --- MODIFICAÇÃO: Ativa a flag para iniciar a análise automaticamente ---
        self.analysis_active = True
        if hasattr(self, 'lbl_status'):
            self.lbl_status.configure(text="WLS analysis in progress.")
        # Chama o processamento frame a frame
        self.processar_frame_histerese_polinomial()

    def processar_frame_histerese_polinomial(self):
        """Processamento frame a frame com subtração de fundo."""
        if not self.analysis_active or self.cap is None:
            cv2.destroyWindow("Analysis: Binary Video")
            return
        """Loop dinâmico que força a reconfiguração das colunas a cada inserção."""
        # --- MODIFICAÇÃO: Trava de Pause ---
        if not getattr(self, 'analysis_active', False):
            return
        ret, frame = self.cap.read()
        if not ret:
            self.analysis_active = False
            self.lbl_status.configure(text="Analysis completed.")
            cv2.destroyWindow("Analysis: Binary Video")
            return

        # --- INSERIR/SUBSTITUIR AQUI: LÓGICA DE SUBTRAÇÃO ---
        # 1. Converte o frame atual
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Subtração: Calcula a diferença real entre o fundo salvo e o frame atual
        # Isso remove tudo o que for estático (sujeira, base, agulha parada)
        diff = cv2.absdiff(self.background_gray, gray)
        
        # 3. Binarização da diferença
        # Note: Usamos THRESH_BINARY pois a diferença gera pixels claros onde há mudança
        _, thresh = cv2.threshold(diff, self.threshold_histerese, 255, cv2.THRESH_BINARY)
        
        # 1. Definir o "Kernel" (o tamanho da expansão)
        # Um kernel 3x3 ou 5x5 costuma ser ideal.
        kernel = np.ones((3, 3), np.uint8)

        # 2. Aplicar a Dilatação
        # iterations=1 define quantas vezes ele vai expandir.
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Exibe o vídeo binário e força a atualização da janela
        cv2.imshow("Analysis: Binary Video", thresh)
        cv2.waitKey(1) 
        # ---------------------------------------------------

        # -----------------------------------------------------

        # 2. Detecção de Contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # ... (resto do seu código original)
        
        if contours:
            # Pega o maior contorno (a gota)
            cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
            
            # 3. Filtragem de pontos próximos à linha de base
            # h_y_base foi definida no método de configuração (run_histerese_polinomial)
            pts_perto_base = np.array([p for p in cnt if abs(p[1] - self.histerese_y_base) < self.tangent_static_pol])

            if len(pts_perto_base) > self.tangent_static_base_points:
                # Divide a gota ao meio para processar esquerda e direita
                x_med = np.mean(pts_perto_base[:, 0])
                esq_pts = pts_perto_base[pts_perto_base[:, 0] < x_med]
                dir_pts = pts_perto_base[pts_perto_base[:, 0] >= x_med]

                # Função interna para o ajuste polinomial x = ay² + by + c
                def calc_poly(pts, lado):
                    if len(pts) < 5: return 0.0, 0.0, (0,0), None, None
                    
                    # Ordena por Y e pega os pontos mais baixos (próximos à base)
                    sort_idx = np.argsort(pts[:, 1])
                    pts_ajuste = pts[sort_idx][:30] 
                    px, py = pts_ajuste[:, 0], pts_ajuste[:, 1]
                    
                    # Ajuste de segunda ordem
                    coef = np.polyfit(py, px, 2)
                    y_contato = np.max(py)
                    x_contato = np.polyval(coef, y_contato)
                    
                    # Derivada para inclinação da tangente
                    dx_dy = 2 * coef[0] * y_contato + coef[1]
                    slope = 1 / abs(dx_dy) if dx_dy != 0 else 999
                    ang = np.degrees(np.arctan(slope))
                    
                    # Ajuste do quadrante do ângulo
                    if lado == "esq":
                        ang = 180 - ang if dx_dy > 0 else ang
                    else:
                        ang = 180 - ang if dx_dy < 0 else ang
                        
                    return ang, dx_dy, (int(x_contato), int(y_contato)), coef, py

                # Cálculos
                a_esq, s_esq, p_esq, c_esq, y_range_e = calc_poly(esq_pts, "esq")
                a_dir, s_dir, p_dir, c_dir, y_range_d = calc_poly(dir_pts, "dir")

                # 4. Desenho e Visualização (utilizando suas funções existentes)
                if c_esq is not None:
                    self.desenhar_curva_ajuste(frame, c_esq, y_range_e, 0, 0, (255, 255, 0))
                    self.desenhar_tangente(frame, p_esq, s_esq, "esq")
                if c_dir is not None:
                    self.desenhar_curva_ajuste(frame, c_dir, y_range_d, 0, 0, (255, 255, 0))
                    self.desenhar_tangente(frame, p_dir, s_dir, "dir")

                # 5. Registro de Dados
                f_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                tempo = f_id / fps if fps > 0 else 0
                diam = abs(p_dir[0] - p_esq[0])
                area = cv2.contourArea(cnt)
                
                self.tree.insert("", "end", values=(
                    f_id, 
                    f"{tempo:.2f}", 
                    f"{diam:.1f}", 
                    f"{a_esq:.2f}", 
                    f"{a_dir:.2f}", 
                    f"{abs(a_esq-a_dir):.2f}", 
                    f"{area:.0f}"
                ))

        # Atualiza a interface (canvas do Tkinter)
        self.display_frame(frame)
        self.after(100, self.processar_frame_histerese_polinomial)

    def run_histerese_wls(self):
        """Inicializa a configuração de mídia e a seleção de base para o método WLS."""
        metodo_atual = "WCA_WLS_DINAMICO" 

        # 1. Limpa os dados residuais da tabela
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        self.ultimo_metodo_selecionado = metodo_atual
        
        # 2. Caixa de diálogo nativa para seleção do arquivo
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("All", "*.*")])
        if not path: 
            return
        self.video_path = path    
        self.current_mode = "HIS"
        self.cap_histerese = cv2.VideoCapture(path)
        self.histerese_pts_base = []
        self.histerese_frame_base = None
        self.histerese_background = None
        
        # Inicialização das variáveis para os pontos fixos iniciais de avanço
        self.p_esq_inicial = None
        self.p_dir_inicial = None
        
        idx = 0
        total = int(self.cap_histerese.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 3. Janela de Navegação Avançar/Recuar por Teclado
        cv2.namedWindow("Config WLS: Frame")
        while True:
            self.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap_histerese.read()
            if not ret: 
                break
            vis = frame.copy()
            cv2.putText(vis, f"WLS Frame: {idx}/{total-1} - A/D to navigate, ENTER to select.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Config WLS: Frame", vis)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('d'): 
                idx = min(idx+1, total-1)
            elif k == ord('a'): 
                idx = max(idx-1, 0)
            elif k == 13: 
                self.histerese_frame_base = frame.copy()
                break
                
        cv2.destroyWindow("Config WLS: Frame")
        
        # 4. Callback para capturar os cliques do mouse na base
        def click_ev(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                self.histerese_pts_base.append((x,y))
                cv2.circle(self.histerese_frame_base, (x,y), 1, (0,0,255), -1)
                cv2.imshow("Config WLS: Base", self.histerese_frame_base)
               
        cv2.namedWindow("Config WLS: Base")
        cv2.setMouseCallback("Config WLS: Base", click_ev)
        cv2.imshow("Config WLS: Base", self.histerese_frame_base)
        
        while len(self.histerese_pts_base) < 2:
            if cv2.waitKey(1) == 27: 
                cv2.destroyAllWindows()
                return 
                
        cv2.destroyAllWindows()
        
        # Define a coordenada Y estável da base
        self.histerese_y_base = max(self.histerese_pts_base[0][1], self.histerese_pts_base[1][1])
        self.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Abre o painel lateral gráfico da interface e dispara o loop contínuo
        self.mostrar_ajustes_histerese()
        # --- MODIFICAÇÃO: Ativa a flag para iniciar a análise automaticamente ---
        self.analysis_active = True
        if hasattr(self, 'lbl_status'):
            self.lbl_status.configure(text="Análise WLS em execução...")
        self.processar_proximo_frame_wls()

    def processar_proximo_frame_wls(self):
        """Loop dinâmico que força a reconfiguração das colunas a cada inserção."""
        # --- MODIFICAÇÃO: Trava de Pause ---
        if not getattr(self, 'analysis_active', False):
            return
        
        ret, frame = self.cap_histerese.read()
        if not ret: 
            self.cap_histerese.release()
            self.lbl_status.configure(text="WLS Hysteresis Analysis Completed.")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fps = self.cap_histerese.get(cv2.CAP_PROP_FPS) or 30
        
        if self.histerese_background is None: 
            self.histerese_background = gray.copy()
        else:
            y_b = self.histerese_y_base
            gray[y_b:, :] = 255  
            
            diff = cv2.absdiff(gray, self.histerese_background)
            blur = cv2.GaussianBlur(diff[:y_b, :], (5,5), 0)
            
            thresh_val = getattr(self, 'threshold_histerese', 45)
            _, bin_img = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
            
            cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 500:
                    n_tangente = getattr(self, 'n_pontos_tangente_val', 30)
                    min_pts = getattr(self, 'min_pontos_fit', 5)
                    
                    res = angulo_interno_base_wls(cnt, y_b, margem_base=15, n_pontos_tangente=n_tangente, min_pontos=min_pts)
                    ang_esq, ang_dir, p_esq, p_dir, viz_esq, viz_dir = res
                    
                    diam_px, p1_d, p2_d = calcular_diametro(cnt, frame)
                    raio = diam_px / 2
                    area_base = math.pi * (raio**2)
                    frame_idx = int(self.cap_histerese.get(cv2.CAP_PROP_POS_FRAMES))
                    tempo_seg = frame_idx / fps

                    # --- CÁLCULOS DAS NOVAS VARIÁVEIS ---
                    # Captura o primeiro ponto de contato fixo se ele ainda não foi definido
                    if p_esq is not None and self.p_esq_inicial is None:
                        self.p_esq_inicial = p_esq[0]
                    if p_dir is not None and self.p_dir_inicial is None:
                        self.p_dir_inicial = p_dir[0]

                    # Medição do diâmetro a partir do ponto de contato com a linha de base
                    if p_esq is not None and p_dir is not None:
                        diametro_base_px = abs(p_dir[0] - p_esq[0])
                    else:
                        diametro_base_px = 0.0

                    # Medição independente do avanço horizontal a partir do primeiro ponto fixo
                    avanço_esq = abs(p_esq[0] - self.p_esq_inicial) if (p_esq is not None and self.p_esq_inicial is not None) else 0.0
                    avanço_dir = abs(p_dir[0] - self.p_dir_inicial) if (p_dir is not None and self.p_dir_inicial is not None) else 0.0
                    # ----------------------------------

                    # Renderizações na imagem
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)
                    cv2.line(frame, (0, int(y_b)), (frame.shape[1], int(y_b)), (255, 0, 0), 2)

                    if p_esq is not None and not np.isnan(ang_esq):
                        cv2.circle(frame, p_esq, 4, (0, 255, 0), -1)
                        rad_esq = math.radians(ang_esq)
                        cv2.line(frame, p_esq, (p_esq[0] + int(50 * math.cos(rad_esq)), p_esq[1] - int(50 * math.sin(rad_esq))), (0, 0, 255), 2)

                    if p_dir is not None and not np.isnan(ang_dir):
                        cv2.circle(frame, p_dir, 4, (0, 255, 0), -1)
                        rad_dir = math.radians(ang_dir)
                        cv2.line(frame, p_dir, (p_dir[0] - int(50 * math.cos(rad_dir)), p_dir[1] - int(50 * math.sin(rad_dir))), (0, 0, 255), 2)
                    
                    histerese_inst = abs(ang_esq - ang_dir) if (not np.isnan(ang_esq) and not np.isnan(ang_dir)) else 0.0
                    
                    # --- FORÇAR RECONFIGURAÇÃO DA TREEVIEW DIRETAMENTE NA INSERÇÃO ---
                    # Adicionadas as 3 novas colunas na tupla de mapeamento estrutural
                    colunas_wls = ("frame", "tempo", "diametro", "diametro_base", "avanço_esq", "avanço_dir", "esq", "dir", "histerese", "area")
                    
                    if self.tree["columns"] != colunas_wls:
                        self.tree.configure(columns=colunas_wls)
                        self.tree.heading("#0", text="", anchor="center")
                        self.tree.heading("frame", text="Frame")
                        self.tree.heading("tempo", text="Tempo (s)")
                        self.tree.heading("diametro", text="Diâmetro (px)")
                        self.tree.heading("diametro_base", text="Diâm. Base (px)")
                        self.tree.heading("avanço_esq", text="Avanc. Esq (px)")
                        self.tree.heading("avanço_dir", text="Avanc. Dir (px)")
                        self.tree.heading("esq", text="Âng. Esq (°)")
                        self.tree.heading("dir", text="Âng. Dir (°)")
                        self.tree.heading("histerese", text="Histérese (°)")
                        self.tree.heading("area", text="Área Base (px²)")
                        
                        self.tree.column("#0", width=0, stretch=tk.NO)
                        for col in colunas_wls:
                            self.tree.column(col, width=90, anchor="center")

                    # Insere todos os dados na tabela, incluindo as novas colunas formatadas
                    self.tree.insert("", "end", values=(
                        frame_idx, 
                        f"{tempo_seg:.3f}",
                        f"{diam_px:.2f}",
                        f"{diametro_base_px:.2f}",
                        f"{avanço_esq:.2f}",
                        f"{avanço_dir:.2f}",
                        f"{ang_esq:.2f}" if not np.isnan(ang_esq) else "N/A", 
                        f"{ang_dir:.2f}" if not np.isnan(ang_dir) else "N/A", 
                        f"{histerese_inst:.2f}",
                        f"{area_base:.2f}"
                    ))

        self.display_frame(frame)
        self.after(10, self.processar_proximo_frame_wls)
    def run_histerese_avancante(self):
# Define o método atual (exemplo para a função run_analise_normal)
        metodo_atual = "HIS_AVANCANTE" 

        # Verifica se o método mudou
        if self.ultimo_metodo_selecionado != metodo_atual:
            # Se for um método diferente, limpa a tabela
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Atualiza o registro para o novo método
            self.ultimo_metodo_selecionado = metodo_atual
        
        # Se o método for o mesmo, ele ignora o 'if' e os dados novos serão 
        # acumulados com os anteriores via self.tree.insert
        path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("All", "*.*")])
        if not path: return
        self.video_path = path
        self.current_mode = "HIS"
        self.cap_histerese = cv2.VideoCapture(path)
        self.histerese_pts_base = []
        self.histerese_frame_base = None
        self.histerese_background = None
        idx = 0
        total = int(self.cap_histerese.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow("Config: Frame")
        while True:
            self.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap_histerese.read()
            if not ret: break
            vis = frame.copy()
            cv2.putText(vis, f"Frame: {idx}/{total-1} - A/D to navigate, ENTER to select.", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Config: Frame", vis)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('d'): idx = min(idx+1, total-1)
            elif k == ord('a'): idx = max(idx-1, 0)
            elif k == 13: self.histerese_frame_base = frame.copy(); break
        cv2.destroyWindow("Config: Frame")
        def click_ev(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                self.histerese_pts_base.append((x,y))
                cv2.circle(self.histerese_frame_base, (x,y), 1, (0,0,255), -1)
                cv2.imshow("Config: Base", self.histerese_frame_base)
               
        cv2.namedWindow("Config: Base")
        cv2.setMouseCallback("Config: Base", click_ev)
        cv2.imshow("Config: Base", self.histerese_frame_base)
        while len(self.histerese_pts_base) < 2:
            if cv2.waitKey(1) == 27: break
        cv2.destroyAllWindows()
        self.histerese_y_base = max(self.histerese_pts_base[0][1], self.histerese_pts_base[1][1])
        # Ordena os pontos da base da esquerda para a direita para o cálculo do avanço
        self.histerese_pts_base = sorted(self.histerese_pts_base, key=lambda p: p[0])
        self.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # --- MODIFICAÇÃO: Ativa a flag para iniciar a análise automaticamente ---
        self.analysis_active = True
        if hasattr(self, 'lbl_status'):
            self.lbl_status.configure(text="Análise WLS em execução...")
        self.processar_proximo_frame_histerese()
   

    def processar_proximo_frame_histerese(self):
        """Loop dinâmico que força a reconfiguração das colunas a cada inserção."""
        # --- MODIFICAÇÃO: Trava de Pause ---
        if not getattr(self, 'analysis_active', False):
            return
        ret, frame = self.cap_histerese.read()
        if not ret: 
            self.cap_histerese.release()
            self.lbl_status.configure(text="Hysteresis Analysis Completed.")
            return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fps = self.cap_histerese.get(cv2.CAP_PROP_FPS) or 30 # Fallback para 30 fps
        if self.histerese_background is None: 
            self.histerese_background = gray.copy()
        else:
            y_b = self.histerese_y_base
            # Máscara para ignorar o que está abaixo da base (reflexos/suporte)
            gray[y_b:, :] = 255
            
            diff = cv2.absdiff(gray, self.histerese_background)
            blur = cv2.GaussianBlur(diff[:y_b, :], (5,5), 0)
            _, bin_img = cv2.threshold(blur, self.threshold_histerese, 255, cv2.THRESH_BINARY)
            
            cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 500:
                    # Suavização das coordenadas do contorno
                   #  pts = cnt.reshape(-1, 2).astype(float)
                   #  if len(pts) > 15:
                    #     pts[:, 0] = savgol_filter(pts[:, 0], 11, 3) # Janela 11, Polinômio 3
                     #    pts[:, 1] = savgol_filter(pts[:, 1], 11, 3)
                      #   cnt = pts.reshape(-1, 1, 2).astype(np.int32)
                    # --- CÁLCULO DOS ÂNGULOS ---
                    # 1. Calcula os ângulos e vizinhanças usando a função auxiliar
                    res = angulo_interno_base(cnt, y_b, n_pontos_tangente=self.n_pontos_tangente_val, min_pontos=self.min_pontos_fit)
      #esse é para aplicar um método de polinomeio de 4 oredem             
                    #res = angulo_interno_base_polinomial(cnt, y_b, n_pontos_tangente=self.n_pontos_tangente_val, min_pontos=self.min_pontos_fit)
                    ang_esq, ang_dir, p_esq, p_dir, viz_esq, viz_dir = res
                    diam_px, p1_d, p2_d = calcular_diametro(cnt, frame)
                    # Cálculo da Área da Base (em pixels quadrados)
                    raio = diam_px / 2
                    area_base = math.pi * (raio**2)
                    frame_idx = int(self.cap_histerese.get(cv2.CAP_PROP_POS_FRAMES))
                    tempo_seg = frame_idx / fps

                    # --- CÁLCULO DO DIÂMETRO DA BASE E DESLOCAMENTOS ---
                    x_base_esq = self.histerese_pts_base[0][0]
                    x_base_dir = self.histerese_pts_base[1][0]
                    
                    # Medição independente baseada na referência de contato fixa inicial
                    if p_esq is not None:
                        avanço_esq = x_base_esq - p_esq[0]
                    else:
                        avanço_esq = 0.0

                    if p_dir is not None:
                        avanço_dir = p_dir[0] - x_base_dir
                    else:
                        avanço_dir = 0.0

                    # CORREÇÃO: O diâmetro da base agora é medido dinamicamente entre os dois pontos de contato atuais
                    if p_esq is not None and p_dir is not None:
                        diametro_base_px = abs(p_dir[0] - p_esq[0])
                    else:
                        # Fallback seguro caso um dos lados falhe na detecção deste frame específico
                        diametro_base_px = abs(x_base_dir - x_base_esq)
                    # 2. Desenha a silhueta e as tangentes
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 1)
                    if p_esq and p_dir:
                        frame = desenhar_tangentes(frame, p_esq, p_dir, viz_esq, viz_dir)
                    
                    # 3. Calcula a média ou histerese e insere na tabela
                    if not np.isnan(ang_esq) and not np.isnan(ang_dir):
                        histerese_inst = abs(ang_esq - ang_dir)
                        frame_idx = int(self.cap_histerese.get(cv2.CAP_PROP_POS_FRAMES))
                        
                        # Insere na Treeview (Tabela)
                        self.tree.insert("", "end", values=(
                            frame_idx, 
                            f"{tempo_seg:.3f}",
                            f"{diam_px:.2f}",
                            f"{diametro_base_px:.2f}",
                            f"{avanço_esq:.2f}",
                            f"{avanço_dir:.2f}",
                            f"{ang_esq:.2f}", 
                            f"{ang_dir:.2f}", 
                            f"{histerese_inst:.2f}",
                            f"{area_base:.2f}"
                        ))

        self.display_frame(frame)
        # Controla a velocidade do loop (10ms de espera)
        self.after(10, self.processar_proximo_frame_histerese)

    def calcular_angulo_quadratico(self, pontos, lado):
        """
        Calcula a tangente no ponto de contato usando ajuste polinomial de grau 2.
        Reduz a flutuação ao considerar a curvatura da gota.
        """
        if len(pontos) < 15:  # Necessário um mínimo de pontos para estabilidade
            return 0.0
        
        try:
            x = pontos[:, 0].astype(float)
            y = pontos[:, 1].astype(float)
            
            # Ajuste: y = ax² + bx + c
            coef = np.polyfit(x, y, 2)
            polinômio = np.poly1d(coef)
            derivada = np.polyder(polinômio)
            
            # Ponto de contato é o X mais externo na base
            x_contato = np.min(x) if lado == "esq" else np.max(x)
            m = derivada(x_contato) # Coeficiente angular da tangente
            
            angulo_rad = math.atan(abs(m))
            angulo_deg = math.degrees(angulo_rad)
            
            # Lógica de quadrante para ângulos avançantes/recuantes
            if lado == "esq":
                return (180 - angulo_deg) if m > 0 else angulo_deg
            else:
                return (180 - angulo_deg) if m < 0 else angulo_deg
        except Exception as e:
            print(f"Erro no cálculo: {e}")
            return 0.0

   
    def load_media(self):
        # Ao não definir initialdir, o sistema abre a última pasta usada ou 'Documentos'
        path = filedialog.askopenfilename(
            title="Selecionar Mídia",
            filetypes=[
                ("Vídeos", "*.avi *.mp4 *.mkv *.mov"), 
                ("Imagens", "*.jpg *.png *.jpeg *.bmp")
            ]
        )
        
        if not path: 
            return
            
        
        # ... código existente ...
        self.stop_video()
        self.img_result_original = None
        
        # ADICIONE ESTAS DUAS LINHAS:
        self.zoom_factor = 1.0 
        self.roi_zoom = None
        # Para qualquer vídeo anterior e limpa referências
        self.stop_video()
        self.img_result_original = None
        
        # O OpenCV lida com caminhos absolutos de qualquer diretório
        if path.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            self.cap = None
            frame = cv2.imread(path)
            if frame is not None:
                self.display_frame(frame)
                self.lbl_status.configure(text=f"Imagem: {os.path.basename(path)}")
        else:
            self.cap = cv2.VideoCapture(path)
            if self.cap:
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if hasattr(self, 'video_slider'):
                    # Configura o slider para ter exatamente o tamanho do vídeo carregado
                    self.video_slider.configure(to=total_frames - 1)
            if self.cap.isOpened():
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.current_frame_idx = 0
                self.show_frame()
                self.lbl_status.configure(text=f"Vídeo: {os.path.basename(path)}")
            else:
                messagebox.showerror("Erro", "Não foi possível abrir o arquivo de vídeo.")

    def show_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                if self.analysis_active:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts: cv2.drawContours(frame, [max(cnts, key=cv2.contourArea)], -1, (0, 255, 0), 2)
                self.display_frame(frame)

    def toggle_video(self):
        if not self.cap: return
        # --- ADICIONE ESTA LINHA ABAIXO ---
            # Sincroniza o OpenCV com a posição atual da bolinha do Slider antes de dar o Play
        if hasattr(self, 'video_slider_var'):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.video_slider_var.get()))
            # ----------------------------------
        self.video_playing = not self.video_playing
        self.btn_play.configure(text="PAUSE" if self.video_playing else "START")
        if self.video_playing: self.run_video_loop()

    def run_video_loop(self):
        if self.video_playing and self.cap:
            if self.current_frame_idx < self.total_frames - 1:
                self.current_frame_idx += 1; self.show_frame(); self.after_id = self.after(30, self.run_video_loop)
            else: self.video_playing = False; self.btn_play.configure(text="START")

    def stop_video(self):
        self.video_playing = False
        if self.after_id: self.after_cancel(self.after_id)
        self.btn_play.configure(text="START")

    def prev_frame(self):
        if self.current_frame_idx > 0: self.current_frame_idx -= 1; self.show_frame()

    def next_frame(self):
        if self.current_frame_idx < self.total_frames - 1: self.current_frame_idx += 1; self.show_frame()

    def start_analysis(self):
        """Resumes analysis after being paused by the user."""
        if not self.analysis_active:
            self.analysis_active = True
            self.vel_active = True
            self.lbl_status.configure(text="Analysis resumed...")
            
            mode = getattr(self, "current_mode", "")
            
            if mode == "HIS":
                metodo = getattr(self, "ultimo_metodo_selecionado", "")
                if metodo == "WCA_WLS_DINAMICO":
                    self.processar_proximo_frame_wls()
                elif metodo == "HIS_POLINOMIAL":
                    self.processar_frame_histerese_polinomial()
                else:
                    self.processar_proximo_frame_histerese()
            elif mode == "DROP":
                metodo = getattr(self, "ultimo_metodo_selecionado", "")
                if metodo == "DROP_QUARTO":
                    self.processar_frame_impacto_quatro()
                else:
                    # Caso haja outros loops de drop impact na sua aplicação
                    if hasattr(self, "processar_proximo_frame_drop"):
                        self.processar_proximo_frame_drop()

    def pause_analysis(self):
        """Pause any running analysis by clearing the flags."""
        self.analysis_active = False
        self.vel_active = False
        self.lbl_status.configure(text="Analysis PAUSED.")
    def reset_all(self):
        """Completely reset the program's state and clear the collected data."""
        # 1. Para qualquer execução ou loop de vídeo ativo
        self.video_playing = False
        self.analysis_active = False
        self.vel_active = False
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

        # 2. Libera o arquivo de vídeo atual da memória
        if self.cap:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'cap_vel') and self.cap_vel:
            self.cap_vel.release()
            self.cap_vel = None

        # 3. Limpa a tabela de dados (Treeview)
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 4. Reseta as variáveis de controle e armazenamento de dados
        self.current_frame_idx = 0
        self.total_frames = 0
        self.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": []}
        self.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0}
        
        # 5. Limpar a Visualização (Imagem/Canvas)
        # Removemos a referência da imagem original e limpamos o widget
        self.img_result_original = None 
        
        # Se for um Label:
        self.canvas_view.configure(image='', text="Awaiting media...")
        self.canvas_view.image = None # Importante para evitar que o Garbage Collector falhe
        
        # Se você usa um Canvas do Tkinter em vez de Label, use:
        # self.canvas_view.delete("all")

        # 6. Atualizar Status e Feedback
        self.lbl_status.configure(text="System reset. Ready for new measurements.")
        messagebox.showinfo("Reset", "All data and images were successfully cleaned.")
        # 5. Reseta elementos da interface (Labels e Canvas)
        #self.lbl_status.configure(text="System Reset. Please import a new file.")
        # Limpa o canvas de vídeo (opcional, dependendo de como você implementou display_frame)
        # self.canvas.delete("all") 
        
        print("All data and video handles have been reset.")
    def restart_video(self):
        """
        Reinicia o vídeo para o primeiro frame.
        Caso uma análise esteja ativa, pausada ou já CONCLUÍDA, limpa a tabela,
        reabre/reseta os arquivos de vídeo e reinicia a medição automaticamente do zero.
        """
        # Captura o modo e o método atual antes de resetar as flags
        modo_atual = getattr(self, "current_mode", None)
        metodo_atual = getattr(self, "ultimo_metodo_selecionado", None)

        # Verifica se há dados na tabela ou se um modo de análise está mapeado
        tem_dados = len(self.tree.get_children()) > 0
        modo_valido = modo_atual in ["HIS", "DROP"]

        # Recupera o caminho do vídeo armazenado na instância (certifique-se de salvar self.video_path ao carregar)
        path_video = getattr(self, "video_path", None)

        if tem_dados or modo_valido:
            # Força a ativação das flags para que a análise continue rodando sozinha após o reset
            self.analysis_active = True
            self.vel_active = True
            self.video_playing = False  # Evita conflitos com player simples

            # Limpa todas as linhas da tabela (Treeview)
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Reseta o dicionário de armazenamento de métricas do Drop Impact
            if hasattr(self, 'data') and isinstance(self.data, dict):
                self.data = {
                    "history_pos": [], 
                    "diam_pre": 1.0, 
                    "tempos": [], 
                    "posicoes": [], 
                    "areas_queda": [],
                    "last_diam_val": None,
                    "max_w_px": 0
                }
            
            # Reseta variáveis de controle de fluxo de ambos os modos
            self.p_esq_inicial = None
            self.p_dir_inicial = None
            self.frame_id = 0
            self.impacto_detectado = False

            if hasattr(self, 'lbl_status'):
                self.lbl_status.configure(text="Análise reiniciada automaticamente do início...")
        else:
            # Se for apenas um vídeo importado para visualização simples
            self.video_playing = False
            self.analysis_active = False
            self.vel_active = False
            if hasattr(self, 'lbl_status'):
                self.lbl_status.configure(text="Vídeo reiniciado para o início.")
            if hasattr(self, 'btn_play'):
                self.btn_play.configure(text="START")

        # --- RESETS DE PONTEIRO E REABERTURA DE VÍDEOS CASO TENHAM TERMINADO (.release) ---
        self.current_frame_idx = 0
        if hasattr(self, 'video_slider_var'):
            self._syncing_slider = True
            try:
                self.video_slider_var.set(0)
            finally:
                self._syncing_slider = False

        # 1. Player principal/visualização simples (self.cap)
        if self.cap:
            if not self.cap.isOpened() and path_video:
                self.cap = cv2.VideoCapture(path_video)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        # 2. Módulo específico de Drop Impact (self.cap_vel)
        if hasattr(self, 'cap_vel'):
            if (self.cap_vel is None or not self.cap_vel.isOpened()) and path_video:
                self.cap_vel = cv2.VideoCapture(path_video)
            elif self.cap_vel is not None:
                self.cap_vel.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        # 3. Módulo específico de Histerese (self.cap_histerese)
        if hasattr(self, 'cap_histerese'):
            if (self.cap_histerese is None or not self.cap_histerese.isOpened()) and path_video:
                self.cap_histerese = cv2.VideoCapture(path_video)
            elif self.cap_histerese is not None:
                self.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # --- REDISPARA O LOOP AUTOMÁTICO BASEADO NO MÉTODO SELECIONADO ---
        if tem_dados or modo_valido:
            if modo_atual == "HIS":
                if metodo_atual == "WCA_WLS_DINAMICO":
                    self.processar_proximo_frame_wls()
                elif metodo_atual == "HIS_POLINOMIAL":
                    # Ajuste extra caso o método polinomial use a propriedade genérica 'self.cap'
                    if self.cap and not self.cap.isOpened() and path_video:
                        self.cap = cv2.VideoCapture(path_video)
                    self.processar_frame_histerese_polinomial()
                else:
                    self.processar_proximo_frame_histerese()
            elif modo_atual == "DROP":
                if metodo_atual == "DROP_QUARTO":
                    self.processar_frame_impacto_quatro()
                else:
                    if hasattr(self, "processar_proximo_frame_drop"):
                        self.processar_proximo_frame_drop()
        else:
            # Caso padrão: apenas atualiza a tela com o frame inicial estático
            self.show_frame()
    def on_mouse_move(self, event):
        if self.img_result_original is None:
            return

        # --- CÁLCULO DE COORDENADAS ABSOLUTAS (Para evitar o pisca-pisca) ---
        # Calculamos a posição do mouse em relação ao Canvas, não ao 'event'
        # Isso evita que o x,y mude quando o mouse entra na label do zoom
        mx = self.canvas_view.winfo_pointerx() - self.canvas_view.winfo_rootx()
        my = self.canvas_view.winfo_pointery() - self.canvas_view.winfo_rooty()

        img = self.img_result_original
        h_orig, w_orig = img.shape[:2]
        tw = self.canvas_view.winfo_width()
        th = self.canvas_view.winfo_height()

        # Mapeamento para a imagem original usando as coordenadas calculadas
        raw_x = int(mx * (w_orig / tw))
        raw_y = int(my * (h_orig / th))

        # Recorte (ROI)
        z_size = int(self.zoom_window_size / self.magnification)
        x1 = max(0, raw_x - z_size // 2)
        y1 = max(0, raw_y - z_size // 2)
        x2 = min(w_orig, x1 + z_size)
        y2 = min(h_orig, y1 + z_size)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0: return
        
        crop_resiz = cv2.resize(crop, (self.zoom_window_size, self.zoom_window_size))
        crop_rgb = cv2.cvtColor(crop_resiz, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(crop_rgb))

        if self.zoom_label is None:
            # Criamos a label vinculada ao canvas
            self.zoom_label = tk.Label(self.canvas_view, image=img_tk, bd=1, relief="solid")
            self.zoom_label.image = img_tk
            
            # --- SOLUÇÃO PARA NÃO SUMIR ---
            # Fazemos a própria label ignorar os eventos de mouse, 
            # ou repassar o movimento para esta mesma função
            self.zoom_label.bind("<Motion>", self.on_mouse_move)
        else:
            self.zoom_label.configure(image=img_tk)
            self.zoom_label.image = img_tk

        # Posicionamento centralizado
        offset = self.zoom_window_size // 2
        new_x = mx - offset
        new_y = my - offset

        # Move a janela
        self.zoom_label.place(x=new_x, y=new_y)
        
        # Garante que ela fique no topo mas não capture o foco
        self.zoom_label.lift()

    def hide_zoom_window(self, event=None):
        """Esconde a lupa apenas se o mouse realmente sair da área do Canvas."""
        if self.zoom_label:
            # Verifica se o mouse ainda está dentro dos limites do canvas_view
            mx = self.canvas_view.winfo_pointerx() - self.canvas_view.winfo_rootx()
            my = self.canvas_view.winfo_pointery() - self.canvas_view.winfo_rooty()
            tw = self.canvas_view.winfo_width()
            th = self.canvas_view.winfo_height()
            
            # Se o mouse ainda estiver dentro da área, não destrói a lupa
            if 0 <= mx <= tw and 0 <= my <= th:
                return
                
            self.zoom_label.destroy()
            self.zoom_label = None
    def save_data_table(self):
        # 1. Verificar se a tabela tem dados
        items = self.tree.get_children()
        if not items:
            messagebox.showwarning("Warning", "The table is empty. There is no data to save.")
            return

        # 2. Obter os cabeçalhos das colunas
        columns = self.tree["columns"]
        
        # 3. Extrair os dados da Treeview
        data = []
        for item in items:
            data.append(self.tree.item(item)["values"])

        # 4. Criar o DataFrame
        df = pd.DataFrame(data, columns=columns)

        # 5. Abrir caixa de diálogo para escolher formato e local
        # Removi o 'defaultextension' fixo para permitir que a escolha do usuário dite a extensão
        file_path = filedialog.asksaveasfilename(
            filetypes=[
                ("Excel (XLSX)", "*.xlsx"),
                ("CSV (Separado por vírgula)", "*.csv"),
                ("Texto (TAB)", "*.txt"),
                ("Todos os arquivos", "*.*")
            ],
            title="Save Analysis Data"
        )

        if not file_path:
            return

        # 6. Salvar conforme a extensão escolhida (usando lower() para segurança)
        try:
            path_lower = file_path.lower()
            
            if path_lower.endswith(".xlsx"):
                # Requer: pip install openpyxl
                df.to_excel(file_path, index=False, engine='openpyxl')
                
            elif path_lower.endswith(".txt"):
                df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')
                
            else:
                # Caso seja .csv ou qualquer outra extensão não mapeada
                if not path_lower.endswith(".csv"):
                    file_path += ".csv"
                df.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')
            
            messagebox.showinfo("Success", f"Data successfully saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error saving", f"The file could not be saved:\n{str(e)}")

if __name__ == "__main__":
    app = TrinityApp()
    app.mainloop()
