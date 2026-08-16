import numpy as np
import cv2
import math
import tkinter as tk
from tkinter import filedialog

from trinity.utils.drawing import desenhar_curva_ajuste, desenhar_tangente_ajuste, desenhar_tangentes
from trinity.utils.contact_angles import angulo_interno_base, angulo_interno_base_wls
from trinity.utils.geometry import calcular_diametro


def run_polynomial_hysteresis(app):
    """Start the polynomial hysteresis workflow through the extracted analysis module."""
    method = "HIS_POLINOMIAL"
    if app.ultimo_metodo_selecionado != method:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = method

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("All", "*.*")])
    if not path:
        return

    app.video_path = path
    app.current_mode = "HIS"
    app.cap = cv2.VideoCapture(path)
    app.histerese_pts_base = []
    app.histerese_frame_base = None

    idx = 0
    total = int(app.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win_frame = "Config: Frame"
    cv2.namedWindow(win_frame, cv2.WINDOW_NORMAL)

    while True:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = app.cap.read()
        if not ret:
            break

        vis = frame.copy()
        cv2.putText(
            vis,
            f"Frame: {idx}/{total - 1} - A/D navegar, ENTER selecionar.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(win_frame, vis)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('d'):
            idx = min(idx + 1, total - 1)
        elif k == ord('a'):
            idx = max(idx - 1, 0)
        elif k == 13:
            app.histerese_frame_base = frame.copy()
            break
        elif k == 27:
            cv2.destroyWindow(win_frame)
            return

    cv2.destroyWindow(win_frame)

    def click_ev(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            app.histerese_pts_base.append((x, y))
            cv2.circle(app.histerese_frame_base, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow("Config: Base", app.histerese_frame_base)

    win_base = "Config: Base"
    cv2.namedWindow(win_base, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_base, click_ev)
    cv2.imshow(win_base, app.histerese_frame_base)

    while len(app.histerese_pts_base) < 2:
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    app.histerese_y_base = max(app.histerese_pts_base[0][1], app.histerese_pts_base[1][1])

    app.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret_bg, frame_bg = app.cap.read()
    if ret_bg:
        bg_gray = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2GRAY)
        app.background_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)

    app.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    app.analysis_active = True
    if hasattr(app, 'lbl_status'):
        app.lbl_status.configure(text="Processing Polynomial Hysteresis...")

    cv2.namedWindow("Analysis: Binary Video", cv2.WINDOW_NORMAL)
    app.processar_frame_histerese_polinomial()


def run_wls_hysteresis(app):
    """Start the WLS hysteresis workflow through the extracted analysis module."""
    method = "WCA_WLS_DINAMICO"
    for item in app.tree.get_children():
        app.tree.delete(item)
    app.ultimo_metodo_selecionado = method

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("All", "*.*")])
    if not path:
        return

    app.video_path = path
    app.current_mode = "HIS"
    app.cap_histerese = cv2.VideoCapture(path)
    app.histerese_pts_base = []
    app.histerese_frame_base = None
    app.histerese_background = None
    app.p_esq_inicial = None
    app.p_dir_inicial = None

    idx = 0
    total = int(app.cap_histerese.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow("Config WLS: Frame")
    while True:
        app.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = app.cap_histerese.read()
        if not ret:
            break
        vis = frame.copy()
        cv2.putText(
            vis,
            f"WLS Frame: {idx}/{total - 1} - A/D to navigate, ENTER to select.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Config WLS: Frame", vis)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('d'):
            idx = min(idx + 1, total - 1)
        elif k == ord('a'):
            idx = max(idx - 1, 0)
        elif k == 13:
            app.histerese_frame_base = frame.copy()
            break

    cv2.destroyWindow("Config WLS: Frame")

    def click_ev(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            app.histerese_pts_base.append((x, y))
            cv2.circle(app.histerese_frame_base, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow("Config WLS: Base", app.histerese_frame_base)

    cv2.namedWindow("Config WLS: Base")
    cv2.setMouseCallback("Config WLS: Base", click_ev)
    cv2.imshow("Config WLS: Base", app.histerese_frame_base)

    while len(app.histerese_pts_base) < 2:
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    app.histerese_y_base = max(app.histerese_pts_base[0][1], app.histerese_pts_base[1][1])
    app.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, 0)
    app.mostrar_ajustes_histerese()
    app.analysis_active = True
    if hasattr(app, 'lbl_status'):
        app.lbl_status.configure(text="Análise WLS em execução...")
    app.processar_proximo_frame_wls()


def run_advancing_hysteresis(app):
    """Start the advancing/receding hysteresis workflow through the extracted analysis module."""
    method = "HIS_AVANCANTE"
    if app.ultimo_metodo_selecionado != method:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = method

    path = filedialog.askopenfilename(filetypes=[("Vídeos", "*.avi *.mp4"), ("All", "*.*")])
    if not path:
        return

    app.video_path = path
    app.current_mode = "HIS"
    app.cap_histerese = cv2.VideoCapture(path)
    app.histerese_pts_base = []
    app.histerese_frame_base = None
    app.histerese_background = None
    idx = 0
    total = int(app.cap_histerese.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow("Config: Frame")
    while True:
        app.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = app.cap_histerese.read()
        if not ret:
            break
        vis = frame.copy()
        cv2.putText(
            vis,
            f"Frame: {idx}/{total - 1} - A/D to navigate, ENTER to select.",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Config: Frame", vis)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('d'):
            idx = min(idx + 1, total - 1)
        elif k == ord('a'):
            idx = max(idx - 1, 0)
        elif k == 13:
            app.histerese_frame_base = frame.copy()
            break

    cv2.destroyWindow("Config: Frame")

    def click_ev(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            app.histerese_pts_base.append((x, y))
            cv2.circle(app.histerese_frame_base, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow("Config: Base", app.histerese_frame_base)

    cv2.namedWindow("Config: Base")
    cv2.setMouseCallback("Config: Base", click_ev)
    cv2.imshow("Config: Base", app.histerese_frame_base)

    while len(app.histerese_pts_base) < 2:
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    app.histerese_y_base = max(app.histerese_pts_base[0][1], app.histerese_pts_base[1][1])
    app.histerese_pts_base = sorted(app.histerese_pts_base, key=lambda p: p[0])
    app.cap_histerese.set(cv2.CAP_PROP_POS_FRAMES, 0)
    app.analysis_active = True
    if hasattr(app, 'lbl_status'):
        app.lbl_status.configure(text="Análise WLS em execução...")
    app.processar_proximo_frame_histerese()


def fit_polynomial_contact(near_base_points, side):
    """Fit x as a quadratic function of y near one contact side."""
    if len(near_base_points) < 5:
        return 0.0, 0.0, (0, 0), None, None

    sort_index = np.argsort(near_base_points[:, 1])
    fitted_points = near_base_points[sort_index][:30]
    x_values, y_values = fitted_points[:, 0], fitted_points[:, 1]

    coefficients = np.polyfit(y_values, x_values, 2)
    contact_y = np.max(y_values)
    contact_x = np.polyval(coefficients, contact_y)
    dx_dy = 2 * coefficients[0] * contact_y + coefficients[1]
    slope = 1 / abs(dx_dy) if dx_dy != 0 else 999
    angle = np.degrees(np.arctan(slope))

    if side == "esq":
        angle = 180 - angle if dx_dy > 0 else angle
    else:
        angle = 180 - angle if dx_dy < 0 else angle

    return angle, dx_dy, (int(contact_x), int(contact_y)), coefficients, y_values


def process_polynomial_frame(app):
    """Process one frame of the polynomial hysteresis analysis."""
    if not app.analysis_active or app.cap is None:
        cv2.destroyWindow("Analysis: Binary Video")
        return

    ret, frame = app.cap.read()
    if not ret:
        app.analysis_active = False
        app.lbl_status.configure(text="Analysis completed.")
        cv2.destroyWindow("Analysis: Binary Video")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(app.background_gray, gray)
    _, threshold = cv2.threshold(diff, app.threshold_histerese, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Analysis: Binary Video", threshold)
    cv2.waitKey(1)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        near_base = np.array(
            [point for point in contour if abs(point[1] - app.histerese_y_base) < app.tangent_static_pol]
        )

        if len(near_base) > app.tangent_static_base_points:
            x_middle = np.mean(near_base[:, 0])
            left_points = near_base[near_base[:, 0] < x_middle]
            right_points = near_base[near_base[:, 0] >= x_middle]

            left = fit_polynomial_contact(left_points, "esq")
            right = fit_polynomial_contact(right_points, "dir")
            left_angle, left_slope, left_point, left_coef, left_y = left
            right_angle, right_slope, right_point, right_coef, right_y = right

            if left_coef is not None:
                desenhar_curva_ajuste(frame, left_coef, left_y, 0, 0, (255, 255, 0))
                desenhar_tangente_ajuste(frame, left_point, left_slope, "esq")
            if right_coef is not None:
                desenhar_curva_ajuste(frame, right_coef, right_y, 0, 0, (255, 255, 0))
                desenhar_tangente_ajuste(frame, right_point, right_slope, "dir")

            frame_id = int(app.cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = app.cap.get(cv2.CAP_PROP_FPS)
            elapsed = frame_id / fps if fps > 0 else 0
            diameter = abs(right_point[0] - left_point[0])
            area = cv2.contourArea(contour)
            app.tree.insert(
                "",
                "end",
                values=(
                    frame_id,
                    f"{elapsed:.2f}",
                    f"{diameter:.1f}",
                    f"{left_angle:.2f}",
                    f"{right_angle:.2f}",
                    f"{abs(left_angle - right_angle):.2f}",
                    f"{area:.0f}",
                ),
            )

    app.display_frame(frame)
    app.after(100, app.processar_frame_histerese_polinomial)


def process_wls_frame(app):
    """Process one frame of the dynamic WLS hysteresis analysis."""
    if not getattr(app, "analysis_active", False):
        return

    ret, frame = app.cap_histerese.read()
    if not ret:
        app.cap_histerese.release()
        app.lbl_status.configure(text="WLS Hysteresis Analysis Completed.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fps = app.cap_histerese.get(cv2.CAP_PROP_FPS) or 30

    if app.histerese_background is None:
        app.histerese_background = gray.copy()
    else:
        base_y = app.histerese_y_base
        gray[base_y:, :] = 255
        diff = cv2.absdiff(gray, app.histerese_background)
        blur = cv2.GaussianBlur(diff[:base_y, :], (5, 5), 0)
        threshold = getattr(app, "threshold_histerese", 45)
        _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 500:
                tangent_points = getattr(app, "n_pontos_tangente_val", 30)
                minimum_points = getattr(app, "min_pontos_fit", 5)
                result = angulo_interno_base_wls(
                    contour,
                    base_y,
                    margem_base=15,
                    n_pontos_tangente=tangent_points,
                    min_pontos=minimum_points,
                )
                angle_left, angle_right, point_left, point_right, _, _ = result

                diameter_px, _, _ = calcular_diametro(contour, frame)
                base_radius = diameter_px / 2
                base_area = math.pi * (base_radius**2)
                frame_index = int(app.cap_histerese.get(cv2.CAP_PROP_POS_FRAMES))
                elapsed = frame_index / fps

                if point_left is not None and app.p_esq_inicial is None:
                    app.p_esq_inicial = point_left[0]
                if point_right is not None and app.p_dir_inicial is None:
                    app.p_dir_inicial = point_right[0]

                base_diameter = abs(point_right[0] - point_left[0]) if point_left is not None and point_right is not None else 0.0
                advance_left = abs(point_left[0] - app.p_esq_inicial) if point_left is not None and app.p_esq_inicial is not None else 0.0
                advance_right = abs(point_right[0] - app.p_dir_inicial) if point_right is not None and app.p_dir_inicial is not None else 0.0

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                cv2.line(frame, (0, int(base_y)), (frame.shape[1], int(base_y)), (255, 0, 0), 2)
                if point_left is not None and not np.isnan(angle_left):
                    cv2.circle(frame, point_left, 4, (0, 255, 0), -1)
                    radians = math.radians(angle_left)
                    cv2.line(frame, point_left, (point_left[0] + int(50 * math.cos(radians)), point_left[1] - int(50 * math.sin(radians))), (0, 0, 255), 2)
                if point_right is not None and not np.isnan(angle_right):
                    cv2.circle(frame, point_right, 4, (0, 255, 0), -1)
                    radians = math.radians(angle_right)
                    cv2.line(frame, point_right, (point_right[0] - int(50 * math.cos(radians)), point_right[1] - int(50 * math.sin(radians))), (0, 0, 255), 2)

                hysteresis = abs(angle_left - angle_right) if not np.isnan(angle_left) and not np.isnan(angle_right) else 0.0
                columns = ("frame", "tempo", "diametro", "diametro_base", "avanço_esq", "avanço_dir", "esq", "dir", "histerese", "area")
                if app.tree["columns"] != columns:
                    app.tree.configure(columns=columns)
                    app.tree.heading("#0", text="", anchor="center")
                    headings = {
                        "frame": "Frame", "tempo": "Tempo (s)", "diametro": "Diâmetro (px)",
                        "diametro_base": "Diâm. Base (px)", "avanço_esq": "Avanc. Esq (px)",
                        "avanço_dir": "Avanc. Dir (px)", "esq": "Âng. Esq (°)", "dir": "Âng. Dir (°)",
                        "histerese": "Histérese (°)", "area": "Área Base (px²)",
                    }
                    for column, heading in headings.items():
                        app.tree.heading(column, text=heading)
                        app.tree.column(column, width=90, anchor="center")
                    app.tree.column("#0", width=0, stretch=tk.NO)

                app.tree.insert("", "end", values=(
                    frame_index, f"{elapsed:.3f}", f"{diameter_px:.2f}", f"{base_diameter:.2f}",
                    f"{advance_left:.2f}", f"{advance_right:.2f}",
                    f"{angle_left:.2f}" if not np.isnan(angle_left) else "N/A",
                    f"{angle_right:.2f}" if not np.isnan(angle_right) else "N/A",
                    f"{hysteresis:.2f}", f"{base_area:.2f}",
                ))

    app.display_frame(frame)
    app.after(10, app.processar_proximo_frame_wls)


def process_advancing_hysteresis_frame(app):
    """Process one frame of the advancing/receding hysteresis analysis."""
    if not getattr(app, "analysis_active", False):
        return

    ret, frame = app.cap_histerese.read()
    if not ret:
        app.cap_histerese.release()
        app.lbl_status.configure(text="Hysteresis Analysis Completed.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fps = app.cap_histerese.get(cv2.CAP_PROP_FPS) or 30
    if app.histerese_background is None:
        app.histerese_background = gray.copy()
    else:
        base_y = app.histerese_y_base
        gray[base_y:, :] = 255
        diff = cv2.absdiff(gray, app.histerese_background)
        blur = cv2.GaussianBlur(diff[:base_y, :], (5, 5), 0)
        _, binary = cv2.threshold(blur, app.threshold_histerese, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 500:
                result = angulo_interno_base(
                    contour,
                    base_y,
                    n_pontos_tangente=app.n_pontos_tangente_val,
                    min_pontos=app.min_pontos_fit,
                )
                angle_left, angle_right, point_left, point_right, view_left, view_right = result
                diameter_px, _, _ = calcular_diametro(contour, frame)
                area_base = math.pi * (diameter_px / 2) ** 2
                frame_index = int(app.cap_histerese.get(cv2.CAP_PROP_POS_FRAMES))
                elapsed = frame_index / fps
                base_left = app.histerese_pts_base[0][0]
                base_right = app.histerese_pts_base[1][0]
                advance_left = base_left - point_left[0] if point_left is not None else 0.0
                advance_right = point_right[0] - base_right if point_right is not None else 0.0
                base_diameter = abs(point_right[0] - point_left[0]) if point_left is not None and point_right is not None else abs(base_right - base_left)

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                if point_left and point_right:
                    frame = desenhar_tangentes(frame, point_left, point_right, view_left, view_right)

                if not np.isnan(angle_left) and not np.isnan(angle_right):
                    hysteresis = abs(angle_left - angle_right)
                    app.tree.insert("", "end", values=(
                        frame_index, f"{elapsed:.3f}", f"{diameter_px:.2f}", f"{base_diameter:.2f}",
                        f"{advance_left:.2f}", f"{advance_right:.2f}", f"{angle_left:.2f}",
                        f"{angle_right:.2f}", f"{hysteresis:.2f}", f"{area_base:.2f}",
                    ))

    app.display_frame(frame)
    app.after(10, app.processar_proximo_frame_histerese)
