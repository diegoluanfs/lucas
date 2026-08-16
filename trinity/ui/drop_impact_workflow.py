import cv2
from tkinter import filedialog

from trinity.analysis.drop_impact import (
    process_fourth_impact_frame,
    process_third_impact_frame,
    process_velocity_frame,
    process_velocity_frame_v2,
    segment_velocity_drop,
    segment_velocity_drop_v2,
)
from trinity.ui.drop_impact_parameters import configure_drop_impact_parameters
from trinity.ui.drop_impact_selector import select_manual_floor_line


def processar_frame_impacto_tres(app):
    """Process a third-method impact analysis frame through the shared analysis helper."""
    return process_third_impact_frame(app)


def run_analise_impacto_tres(app):
    """Method 3: select background and baseline, then run frame processing."""
    metodo_atual = "DROP_TERCEIRO"
    if app.ultimo_metodo_selecionado != metodo_atual:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = metodo_atual

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("Todos", "*.*")])
    if not path:
        return

    app.video_path = path
    app.params = app.configurar_parametros_velocidade()

    cap_config = cv2.VideoCapture(path)
    total = int(cap_config.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0

    win_frame = "Config: Select Background"
    cv2.namedWindow(win_frame, cv2.WINDOW_NORMAL)

    while True:
        cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_ref = cap_config.read()
        if not ret:
            break

        vis = frame_ref.copy()
        cv2.putText(vis, f"Frame: {idx}/{total-1} - A/D navigate, ENTER select.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win_frame, vis)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('d'):
            idx = min(idx + 1, total - 1)
        elif k == ord('a'):
            idx = max(idx - 1, 0)
        elif k == 13:
            app.background_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            break
        elif k == 27:
            cv2.destroyWindow(win_frame)
            cap_config.release()
            return

    cv2.destroyWindow(win_frame)

    app.pts_base = []
    win_baseline = "Config: Define Baseline"
    cv2.namedWindow(win_baseline, cv2.WINDOW_NORMAL)

    def click_ev(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(app.pts_base) < 2:
            app.pts_base.append((x, y))

    cv2.setMouseCallback(win_baseline, click_ev)

    while True:
        cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_ref = cap_config.read()
        if not ret:
            break

        vis_base = frame_ref.copy()
        cv2.putText(vis_base, f"Frame: {idx}/{total-1} - A/D navigate. Click 2 points.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_base, "ENTER: Confirm Baseline | R: Reset Points | ESC: Exit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for pt in app.pts_base:
            cv2.circle(vis_base, pt, 3, (0, 0, 255), -1)

        if len(app.pts_base) == 2:
            cv2.line(vis_base, app.pts_base[0], app.pts_base[1], (0, 255, 255), 2)

        cv2.imshow(win_baseline, vis_base)
        k = cv2.waitKey(30) & 0xFF

        if k == ord('d'):
            idx = min(idx + 1, total - 1)
        elif k == ord('a'):
            idx = max(idx - 1, 0)
        elif k == ord('r'):
            app.pts_base = []
        elif k == 13:
            if len(app.pts_base) == 2:
                break
        elif k == 27:
            cv2.destroyAllWindows()
            cap_config.release()
            return

    app.limite_y_base = (app.pts_base[0][1] + app.pts_base[1][1]) / 2
    cv2.destroyAllWindows()
    cap_config.release()

    app.cap_vel = cv2.VideoCapture(path)
    app.current_mode = "DROP"
    app.params.update({"min_area": 300, "fps_real": app.params.get("fps_real", 10000)})

    app.data = {"history_pos": [], "diam_pre": 1.0, "tempos": [], "posicoes": [], "areas_queda": []}
    app.frame_id = 0
    app.impacto_detectado = False
    app.vel_active = True

    app.lbl_status.configure(text="Baseline defined. Starting analysis...")
    processar_frame_impacto_tres(app)


def processar_frame_impacto_quatro(app):
    """Process a fourth-method impact frame through the shared analysis helper."""
    return process_fourth_impact_frame(app)


def run_drop_impact_analysis(app):
    """Method 4: impact analysis measured on the base line using the shared workflow."""
    metodo_atual = "DROP_QUARTO"
    if app.ultimo_metodo_selecionado != metodo_atual:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = metodo_atual

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4 *.tif"), ("Todos", "*.*")])
    if not path:
        return

    app.video_path = path
    app.params = app.configurar_parametros_velocidade()

    cap_config = cv2.VideoCapture(path)
    total = int(cap_config.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return
    idx = 0

    win_frame = "Config: Select Background"
    cv2.namedWindow(win_frame, cv2.WINDOW_NORMAL)

    while True:
        cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_ref = cap_config.read()
        if not ret:
            break

        vis = frame_ref.copy()
        cv2.putText(vis, f"Frame: {idx}/{total-1} - A/D navigate, ENTER select.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win_frame, vis)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('d'):
            idx = min(idx + 1, total - 1)
        elif k == ord('a'):
            idx = max(idx - 1, 0)
        elif k == 13:
            app.background_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            break
        elif k == 27:
            cv2.destroyWindow(win_frame)
            cap_config.release()
            return

    cv2.destroyWindow(win_frame)

    app.pts_base = []
    win_baseline = "Config: Define Baseline"
    cv2.namedWindow(win_baseline, cv2.WINDOW_NORMAL)

    def click_ev(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(app.pts_base) < 2:
            app.pts_base.append((x, y))

    cv2.setMouseCallback(win_baseline, click_ev)

    while True:
        cap_config.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_ref = cap_config.read()
        if not ret:
            break

        vis_base = frame_ref.copy()
        w_img = vis_base.shape[1]

        cv2.putText(vis_base, f"Frame: {idx}/{total-1} - A/D navigate. Click 2 points.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_base, "ENTER: Confirm Baseline | R: Reset Points | ESC: Exit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for pt in app.pts_base:
            cv2.circle(vis_base, pt, 3, (0, 0, 255), -1)

        if len(app.pts_base) == 2:
            x1, y1 = app.pts_base[0]
            x2, y2 = app.pts_base[1]
            if x2 - x1 != 0:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                pt_inicio = (0, int(c))
                pt_fim = (w_img, int(m * w_img + c))
            else:
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
            app.pts_base = []
        elif k == 13 and len(app.pts_base) == 2:
            break
        elif k == 27:
            cv2.destroyAllWindows()
            cap_config.release()
            return

    x1, y1 = app.pts_base[0]
    x2, y2 = app.pts_base[1]
    if x2 - x1 != 0:
        app.m_base = (y2 - y1) / (x2 - x1)
        app.c_base = y1 - app.m_base * x1
    else:
        app.m_base = 0
        app.c_base = y1

    cv2.destroyAllWindows()
    cap_config.release()

    app.cap_vel = cv2.VideoCapture(path)
    app.current_mode = "DROP"
    app.params.update({"min_area": 300, "fps_real": app.params.get("fps_real", 10000)})
    app.data = {"history_pos": [], "diam_pre": 1.0, "tempos": [], "posicoes": [], "areas_queda": [], "last_diam_val": None, "max_w_px": 0}
    app.frame_id = 0
    app.impacto_detectado = False
    app.vel_active = True

    if hasattr(app, 'lbl_status'):
        app.lbl_status.configure(text="Baseline defined. Starting Fourth Method analysis...")
    app.analysis_active = True
    app.vel_active = True

    if hasattr(app, 'lbl_status'):
        app.lbl_status.configure(text="Análise de Impacto em execução...")
    processar_frame_impacto_quatro(app)


def run_analise_velocidade(app):
    """Run the first drop-impact speed workflow through the extracted helper."""
    metodo_atual = "DROP_VELOCIDADE"
    if app.ultimo_metodo_selecionado != metodo_atual:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = metodo_atual

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("Todos", "*.*")])
    if not path:
        return

    app.video_path = path
    app.vel_params = app.configurar_parametros_velocidade()
    app.cap_vel = cv2.VideoCapture(path)
    if not app.cap_vel.isOpened():
        return

    lp = app.selecionar_chao_manual(path)
    app.cap_vel.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if (lp["p2"][0] - lp["p1"][0]) != 0:
        incl_a = (lp["p2"][1] - lp["p1"][1]) / (lp["p2"][0] - lp["p1"][0])
        incl_b = lp["p1"][1] - incl_a * lp["p1"][0]
    else:
        incl_a = 0
        incl_b = lp["p1"][1]

    app.current_mode = "DROP"
    app.vel_params.update({"line_a": incl_a, "line_b": incl_b})
    app.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": [], "areas_queda": []}
    app.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0, "last_diam_val": None}
    app.vel_active = True
    app.lbl_status.configure(text="Analyzing Speed")
    process_velocity_frame(app)


def run_velocity_analysis(app):
    """Run the second drop-impact speed workflow through the extracted helper."""
    metodo_atual = "DROP_VELOCITY"
    if app.ultimo_metodo_selecionado != metodo_atual:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = metodo_atual

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("Todos", "*.*")])
    if not path:
        return

    app.video_path = path
    app.vel_params = app.configurar_parametros_velocidade()
    app.cap_vel = cv2.VideoCapture(path)
    if not app.cap_vel.isOpened():
        return

    lp = app.selecionar_chao_manual(path)
    if (lp["p2"][0] - lp["p1"][0]) != 0:
        incl_a = (lp["p2"][1] - lp["p1"][1]) / (lp["p2"][0] - lp["p1"][0])
        incl_b = lp["p1"][1] - incl_a * lp["p1"][0]
    else:
        incl_a, incl_b = 0, lp["p1"][1]

    app.vel_params.update({"line_a": incl_a, "line_b": incl_b})
    app.current_mode = "DROP"
    app.vel_data = {"tempos": [], "posicoes": [], "diametros": [], "pre_impacto": [], "areas_queda": []}
    app.vel_state = {"frame_id": 0, "impacto": False, "background": None, "diametro_pre": 1.0, "last_diam_val": None}
    app.vel_active = True
    app.lbl_status.configure(text="Analyzing Speed and Impact...")
    process_velocity_frame_v2(app)


def segmenta_gota_velocidade(app, diff_frame):
    return segment_velocity_drop(diff_frame, app.vel_params["line_a"], app.vel_params["line_b"])


def segmenta_gota_velocidade_2(app, diff_frame):
    return segment_velocity_drop_v2(diff_frame, app.vel_params["line_b"])
