import cv2
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


def start_analysis(app):
    """Resume a paused analysis based on the current mode and selected method."""
    if not app.analysis_active:
        app.analysis_active = True
        app.vel_active = True
        app.lbl_status.configure(text="Analysis resumed...")

        mode = getattr(app, "current_mode", "")

        if mode == "HIS":
            metodo = getattr(app, "ultimo_metodo_selecionado", "")
            if metodo == "WCA_WLS_DINAMICO":
                app.processar_proximo_frame_wls()
            elif metodo == "HIS_POLINOMIAL":
                app.processar_frame_histerese_polinomial()
            else:
                app.processar_proximo_frame_histerese()
        elif mode == "DROP":
            metodo = getattr(app, "ultimo_metodo_selecionado", "")
            if metodo == "DROP_QUARTO":
                app.processar_frame_impacto_quatro()
            else:
                if hasattr(app, "processar_proximo_frame_drop"):
                    app.processar_proximo_frame_drop()


def pause_analysis(app):
    """Pause the active analysis loop."""
    app.analysis_active = False
    app.vel_active = False
    app.lbl_status.configure(text="Analysis PAUSED.")


def reset_all(app):
    """Reset all program state and clear collected data."""
    app.video_playing = False
    app.analysis_active = False
    app.vel_active = False
    if app.after_id:
        app.after_cancel(app.after_id)
        app.after_id = None

    if app.cap:
        app.cap.release()
        app.cap = None
    if hasattr(app, 'cap_vel') and app.cap_vel:
        app.cap_vel.release()
        app.cap_vel = None

    for item in app.tree.get_children():
        app.tree.delete(item)

    app.current_frame_idx = 0
    app.total_frames = 0
    app.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": []}
    app.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0}
    app.img_result_original = None

    app.canvas_view.configure(image='', text="Awaiting media...")
    app.canvas_view.image = None

    app.lbl_status.configure(text="System reset. Ready for new measurements.")
    messagebox.showinfo("Reset", "All data and images were successfully cleaned.")


def restart_video(app):
    """Restart the current video or analysis from frame zero."""
    modo_atual = getattr(app, "current_mode", None)
    metodo_atual = getattr(app, "ultimo_metodo_selecionado", None)
    tem_dados = len(app.tree.get_children()) > 0
    modo_valido = modo_atual in ["HIS", "DROP"]
    path_video = getattr(app, "video_path", None)

    if tem_dados or modo_valido:
        app.analysis_active = True
        app.vel_active = True
        app.video_playing = False

        for item in app.tree.get_children():
            app.tree.delete(item)

        if hasattr(app, 'data') and isinstance(app.data, dict):
            app.data = {
                "history_pos": [],
                "diam_pre": 1.0,
                "tempos": [],
                "posicoes": [],
                "areas_queda": [],
                "last_diam_val": None,
                "max_w_px": 0,
            }

        app.p_esq_inicial = None
        app.p_dir_inicial = None
        app.frame_id = 0
        app.impacto_detectado = False

        if hasattr(app, 'lbl_status'):
            app.lbl_status.configure(text="Análise reiniciada automaticamente do início...")
    else:
        app.video_playing = False
        app.analysis_active = False
        app.vel_active = False
        if hasattr(app, 'lbl_status'):
            app.lbl_status.configure(text="Vídeo reiniciado para o início.")
        if hasattr(app, 'btn_play'):
            app.btn_play.configure(text="START")

    app.current_frame_idx = 0
    if hasattr(app, 'video_slider_var'):
        app._syncing_slider = True
        try:
            app.video_slider_var.set(0)
        finally:
            app._syncing_slider = False

    if app.cap:
        if not app.cap.isOpened() and path_video:
            app.cap = cv2.VideoCapture(path_video)
        else:
            app.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if hasattr(app, 'cap_vel'):
        if (app.cap_vel is None or not app.cap_vel.isOpened()) and path_video:
            app.cap_vel = cv2.VideoCapture(path_video)
        elif app.cap_vel is not None:
            app.cap_vel.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if hasattr(app, 'cap_histerese'):
        if (app.cap_histerese is None or not app.cap_histerese.isOpened()) and path_video:
            app.cap_histerese = cv2.VideoCapture(path_video)
        elif app.cap_histerese is not None:
            app.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if tem_dados or modo_valido:
        if modo_atual == "HIS":
            if metodo_atual == "WCA_WLS_DINAMICO":
                app.processar_proximo_frame_wls()
            elif metodo_atual == "HIS_POLINOMIAL":
                if app.cap and not app.cap.isOpened() and path_video:
                    app.cap = cv2.VideoCapture(path_video)
                app.processar_frame_histerese_polinomial()
            else:
                app.processar_proximo_frame_histerese()
        elif modo_atual == "DROP":
            if metodo_atual == "DROP_QUARTO":
                app.processar_frame_impacto_quatro()
            else:
                if hasattr(app, "processar_proximo_frame_drop"):
                    app.processar_proximo_frame_drop()
    else:
        app.show_frame()


def on_mouse_move(app, event):
    if app.img_result_original is None:
        return

    mx = app.canvas_view.winfo_pointerx() - app.canvas_view.winfo_rootx()
    my = app.canvas_view.winfo_pointery() - app.canvas_view.winfo_rooty()

    img = app.img_result_original
    h_orig, w_orig = img.shape[:2]
    tw = app.canvas_view.winfo_width()
    th = app.canvas_view.winfo_height()

    raw_x = int(mx * (w_orig / tw))
    raw_y = int(my * (h_orig / th))

    z_size = int(app.zoom_window_size / app.magnification)
    x1 = max(0, raw_x - z_size // 2)
    y1 = max(0, raw_y - z_size // 2)
    x2 = min(w_orig, x1 + z_size)
    y2 = min(h_orig, y1 + z_size)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return

    crop_resiz = cv2.resize(crop, (app.zoom_window_size, app.zoom_window_size))
    crop_rgb = cv2.cvtColor(crop_resiz, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(crop_rgb))

    if app.zoom_label is None:
        app.zoom_label = tk.Label(app.canvas_view, image=img_tk, bd=1, relief="solid")
        app.zoom_label.image = img_tk
        app.zoom_label.bind("<Motion>", app.on_mouse_move)
    else:
        app.zoom_label.configure(image=img_tk)
        app.zoom_label.image = img_tk

    offset = app.zoom_window_size // 2
    new_x = mx - offset
    new_y = my - offset
    app.zoom_label.place(x=new_x, y=new_y)
    app.zoom_label.lift()


def hide_zoom_window(app, event=None):
    """Hide the magnifier when the pointer leaves the canvas."""
    if app.zoom_label:
        mx = app.canvas_view.winfo_pointerx() - app.canvas_view.winfo_rootx()
        my = app.canvas_view.winfo_pointery() - app.canvas_view.winfo_rooty()
        tw = app.canvas_view.winfo_width()
        th = app.canvas_view.winfo_height()

        if 0 <= mx <= tw and 0 <= my <= th:
            return

        app.zoom_label.destroy()
        app.zoom_label = None


def save_data_table(app):
    """Export the current tree data to CSV, TSV or XLSX."""
    items = app.tree.get_children()
    if not items:
        messagebox.showwarning("Warning", "The table is empty. There is no data to save.")
        return

    columns = app.tree["columns"]
    data = [app.tree.item(item)["values"] for item in items]
    df = pd.DataFrame(data, columns=columns)

    file_path = filedialog.asksaveasfilename(
        filetypes=[
            ("Excel (XLSX)", "*.xlsx"),
            ("CSV (Separado por vírgula)", "*.csv"),
            ("Texto (TAB)", "*.txt"),
            ("Todos os arquivos", "*.*"),
        ],
        title="Save Analysis Data",
    )

    if not file_path:
        return

    try:
        path_lower = file_path.lower()

        if path_lower.endswith(".xlsx"):
            df.to_excel(file_path, index=False, engine='openpyxl')
        elif path_lower.endswith(".txt"):
            df.to_csv(file_path, index=False, sep='\t', encoding='utf-8')
        else:
            if not path_lower.endswith(".csv"):
                file_path += ".csv"
            df.to_csv(file_path, index=False, sep=';', encoding='utf-8-sig')

        messagebox.showinfo("Success", f"Data successfully saved to:\n{file_path}")
    except Exception as e:
        messagebox.showerror("Error saving", f"The file could not be saved:\n{str(e)}")
