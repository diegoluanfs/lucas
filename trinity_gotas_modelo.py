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
from trinity.analysis.plot_service import render_legacy_plot
from trinity.analysis.edge_detection import detect_edges
from trinity.analysis.drop_impact import (
    process_fourth_impact_frame,
    process_third_impact_frame,
    process_velocity_frame,
    process_velocity_frame_v2,
    robust_segmentation as segment_drop_background,
    segment_velocity_drop,
    segment_velocity_drop_v2,
)
from trinity.analysis.info_overlay import create_info_overlay
from trinity.analysis.static_angle import run_advanced_static_angle, run_circle_fit, run_normal_static_angle
from trinity.analysis.hysteresis import (
    fit_polynomial_contact,
    process_advancing_hysteresis_frame,
    process_polynomial_frame,
    process_wls_frame,
    run_advancing_hysteresis,
    run_polynomial_hysteresis,
    run_wls_hysteresis,
)
from trinity.ui.analysis_menu import (
    create_main_menu,
    mostrar_ajustes_histerese,
    mostrar_ajustes_impact,
    mostrar_ajustes_static,
    mostrar_ajustes_static_pol,
    show_analysis_options,
    show_sub_menu,
    update_min_pontos_fit,
    update_tangente_value,
    update_thresh_value,
    update_thresh_value_impact,
    update_thresh_value_static,
    update_thresh_value_static_base_points,
    update_thresh_value_static_pol,
)
from trinity.ui.app_shell import build_main_ui
from trinity.ui.tooltip import Tooltip
from trinity.ui.drop_impact_selector import select_manual_floor_line
from trinity.ui.drop_impact_parameters import configure_drop_impact_parameters
from trinity.ui.drop_impact_results import show_dimensionless_results
from trinity.ui.static_angle_selector import select_three_points
from trinity.ui.app_controls import (
    hide_zoom_window,
    on_mouse_move,
    pause_analysis,
    reset_all,
    restart_video,
    save_data_table,
    start_analysis,
)
from trinity.ui.view_controls import (
    apply_zoom,
    display_frame,
    on_click_press,
    on_click_release,
    on_mouse_drag,
    on_mouse_scroll_windows,
    on_mouse_wheel,
    on_slider_move,
    robust_segmentation,
    update_zoom_view,
)
from trinity.ui.dashboard import (
    reset_analysis,
    show_data_table,
    show_home_view,
)
from trinity.ui.visual_tools import (
    edge_detection,
    fitting,
    save_current_image,
    show_important_info,
)
from trinity.ui.drop_impact_workflow import (
    processar_frame_impacto_quatro,
    processar_frame_impacto_tres,
    run_analise_impacto_tres,
    run_analise_velocidade,
    run_drop_impact_analysis,
    run_velocity_analysis,
    segmenta_gota_velocidade,
    segmenta_gota_velocidade_2,
)
from trinity.utils.contact_angles import (
    angulo_interno_base,
    angulo_interno_base_polinomial,
    angulo_interno_base_wls,
    calcular_angulo_quadratico,
    calcular_angulo_wls,
)
from trinity.utils.drawing import (
    desenhar_arco_angulo,
    desenhar_arco_angulo_ajuste,
    desenhar_curva_ajuste,
    desenhar_tangentes,
)
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
        self.title("TRINITY - Automated Wettability and Contact Angle Analysis")
        self.geometry("1400x900")
        self.minsize(1100, 700)

        self.current_mode = "WCA"
        self.ultimo_metodo_selecionado = None
        self.analysis_active = False
        self.vel_active = False
        self.video_playing = False
        self.after_id = None
        self.img_result_original = None
        self.current_vis_img = None
        self.current_contour = None
        self.cap = None
        self.cap_vel = None
        self.cap_histerese = None
        self.video_path = ""
        self.show_edges = False
        self.zoom_factor = 1.0
        self.roi_zoom = None
        self.dragging = False
        self.start_x = 0
        self.start_y = 0
        self.zoom_label = None
        self.zoom_window_size = 180
        self.magnification = 2
        self.total_frames = 0
        self.current_frame_idx = 0
        self.threshold_histerese = 60
        self.n_pontos_tangente_val = 20
        self.min_pontos_fit = 10
        self.threshold_impact = 100
        self.tangent_static = 20
        self.tangent_static_pol = 20
        self.tangent_static_base_points = 15
        self.ajustes_container = None

        build_main_ui(self)
    
    def edge_detection(self):
        edge_detection(self)

    def show_important_info(self):
        show_important_info(self)

    def save_current_image(self):
        save_current_image(self)

    def fitting(self):
        fitting(self)

    def show_home_view(self):
        show_home_view(self)

    def show_analyze_options(self):
        show_analysis_options(self)

    def reset_analysis(self):
        reset_analysis(self)
    def mostrar_ajustes_histerese(self):
        mostrar_ajustes_histerese(self)

    def update_thresh_value(self, val):
        update_thresh_value(self, val)

    def update_tangente_value(self, val):
        update_tangente_value(self, val)

    def update_min_pontos_fit(self, val):
        update_min_pontos_fit(self, val)

    def mostrar_ajustes_impact(self):
        mostrar_ajustes_impact(self)

    def update_thresh_value_impact(self, val):
        update_thresh_value_impact(self, val)

    def mostrar_ajustes_static(self):
        mostrar_ajustes_static(self)

    def update_thresh_value_static(self, val):
        update_thresh_value_static(self, val)

    def mostrar_ajustes_static_pol(self):
        mostrar_ajustes_static_pol(self)

    def update_thresh_value_static_pol(self, val):
        update_thresh_value_static_pol(self, val)

    def update_thresh_value_static_base_points(self, val):
        update_thresh_value_static_base_points(self, val)

    def plot_graph(self, type):
        render_legacy_plot(self, type)

    def show_data_table(self):
        show_data_table(self)

    def create_main_menu(self):
        create_main_menu(self)

    def show_sub_menu(self, mode):
        show_sub_menu(self, mode)

    def apply_zoom(self, factor):
        apply_zoom(self, factor)

    def on_mouse_wheel(self, event):
        on_mouse_wheel(self, event)

    def on_mouse_scroll_windows(self, event):
        on_mouse_scroll_windows(self, event)

    def on_click_press(self, event):
        on_click_press(self, event)

    def on_mouse_drag(self, event):
        on_mouse_drag(self, event)

    def on_click_release(self, event):
        on_click_release(self, event)

    def update_zoom_view(self, is_roi=False):
        update_zoom_view(self, is_roi)

    def display_frame(self, frame, update_original=True):
        display_frame(self, frame, update_original)

    def on_slider_move(self, value):
        on_slider_move(self, value)

    def robust_segmentation(self, frame_gray, background):
        robust_segmentation(self, frame_gray, background)

    def run_analise_impacto_tres(self):
        run_analise_impacto_tres(self)

    def processar_frame_impacto_tres(self):
        processar_frame_impacto_tres(self)

    def run_drop_impact_analysis(self):
        run_drop_impact_analysis(self)

    def processar_frame_impacto_quatro(self):
        processar_frame_impacto_quatro(self)

    def configurar_parametros_velocidade(self):
        return configure_drop_impact_parameters(self)

    def mostrar_resultados_adimensionais(self, v, d_medio, re, we, oh, area_pre):
        return show_dimensionless_results(self, v, d_medio, re, we, oh, area_pre)

    def selecionar_chao_manual(self, path_video):
        return select_manual_floor_line(path_video)

    def run_analise_velocidade(self):
        run_analise_velocidade(self)

    def segmenta_gota_velocidade(self, diff_frame):
        return segmenta_gota_velocidade(self, diff_frame)

    def processar_frame_velocidade(self):
        return process_velocity_frame(self)

    def run_velocity_analysis(self):
        run_velocity_analysis(self)

    def segmenta_gota_velocidade_2(self, diff_frame):
        return segmenta_gota_velocidade_2(self, diff_frame)

    def processar_frame_velocidade_2(self):
        return process_velocity_frame_v2(self)
    def run_analise_normal(self):
        return run_normal_static_angle(self)
    def run_analise_avancada(self):
        return run_advanced_static_angle(self)

    def selecionar_tres_pontos(self, img_roi):
        """
        Permite selecionar 3 pontos com linhas elásticas em tempo real.
        Retorna uma lista com os 3 pontos [(x1,y1), (x2,y2), (x3,y3)]
        """
        return select_three_points(img_roi)
    
    def run_analise_manual(self):
        run_manual_static_angle(self)

    def run_histerese_polinomial(self):
        return run_polynomial_hysteresis(self)

    def processar_frame_histerese_polinomial(self):
        return process_polynomial_frame(self)

    def run_histerese_wls(self):
        return run_wls_hysteresis(self)

    def processar_proximo_frame_wls(self):
        return process_wls_frame(self)

    def run_histerese_avancante(self):
        return run_advancing_hysteresis(self)

    def processar_proximo_frame_histerese(self):
        return process_advancing_hysteresis_frame(self)

    def calcular_angulo_quadratico(self, pontos, lado):
        return calcular_angulo_quadratico(pontos, lado)

    def load_media(self):
        load_media_file(self)

    def show_frame(self):
        show_current_frame(self)

    def toggle_video(self):
        toggle_video_player(self)

    def run_video_loop(self):
        run_video_loop(self)

    def stop_video(self):
        stop_video_player(self)

    def prev_frame(self):
        prev_frame(self)

    def next_frame(self):
        next_frame(self)

    def start_analysis(self):
        start_analysis(self)

    def pause_analysis(self):
        pause_analysis(self)

    def reset_all(self):
        reset_all(self)

    def restart_video(self):
        restart_video(self)

    def on_mouse_move(self, event):
        on_mouse_move(self, event)

    def hide_zoom_window(self, event=None):
        hide_zoom_window(self, event)

    def save_data_table(self):
        save_data_table(self)

if __name__ == "__main__":
    app = TrinityApp()
    app.mainloop()
