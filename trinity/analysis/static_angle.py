import os
import tkinter as tk

import cv2
import numpy as np
import scipy.optimize
from scipy.signal import savgol_filter
from tkinter import filedialog, messagebox

from trinity.utils.drawing import (
    desenhar_arco_angulo,
    desenhar_arco_angulo_ajuste,
    desenhar_curva_ajuste,
    desenhar_tangente_ajuste,
)
from trinity.utils.geometry import (
    calcular_angulo_dropy,
    pontos_proximos_base,
    selecionar_ponto_callback,
    separar_lados,
)


def run_advanced_static_angle(app):
    """Run the advanced static angle workflow using adaptive preprocessing and polynomial fitting."""
    method = "WCA_AVANCADO"
    if app.ultimo_metodo_selecionado != method:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = method

    path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg"), ("All", "*.*")])
    if not path:
        return

    app.video_path = path
    nome_arquivo = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return

    orig = img.copy()
    r = cv2.selectROI("Selecione ROI", img, showCrosshair=True)
    cv2.destroyWindow("Selecione ROI")
    x_roi, y_roi, w_roi, h_roi = r
    if w_roi == 0 or h_roi == 0:
        return

    roi = img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
    pontos_base = []
    roi_vis = roi.copy()
    cv2.imshow("Selecione Base", roi_vis)
    cv2.setMouseCallback("Selecione Base", selecionar_ponto_callback, param=[pontos_base, roi_vis, "Selecione Base"])

    while len(pontos_base) < 2:
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Selecione Base")

    p1, p2 = pontos_base
    y_base = max(p1[1], p2[1])

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return

    contornos_validos = []
    for c in contours:
        pts = c.reshape(-1, 2)
        if len(pts) > 50:
            largura = np.max(pts[:, 0]) - np.min(pts[:, 0])
            contornos_validos.append((largura, pts))

    if not contornos_validos:
        return

    contorno = max(contornos_validos, key=lambda item: item[0])[1]

    tolerancia_altura = app.tangent_static_pol
    minimo_pontos_obrigatorio = max(10, app.tangent_static_base_points)
    lista_pontos = []

    for tentativa in range(3):
        lista_pontos = [p for p in contorno if abs(p[1] - y_base) < tolerancia_altura]
        if len(lista_pontos) >= minimo_pontos_obrigatorio:
            break
        tolerancia_altura = int(tolerancia_altura * 1.5)

    if len(lista_pontos) < minimo_pontos_obrigatorio:
        messagebox.showerror(
            "Erro de Detecção",
            f"Poucos pontos próximos à base ({len(lista_pontos)} encontrados).\n\n"
            "Dicas:\n"
            "1. Melhore o contraste/iluminação da foto.\n"
            "2. Selecione a base exatamente onde a gota toca a superfície.\n"
            "3. Aumente o valor de 'tangent_static_pol' nas configurações.",
        )
        return

    pontos = np.array(lista_pontos)
    x_med = np.mean(pontos[:, 0])
    esq = pontos[pontos[:, 0] < x_med]
    dirr = pontos[pontos[:, 0] >= x_med]
    esq = esq[np.argsort(esq[:, 1])]
    dirr = dirr[np.argsort(dirr[:, 1])]

    def suavizar(p):
        if len(p) < 11:
            return p
        x = savgol_filter(p[:, 0], 11, 3)
        y = savgol_filter(p[:, 1], 11, 3)
        return np.column_stack((x, y))

    esq = suavizar(esq)
    dirr = suavizar(dirr)

    def calcular_angulo(pontos_local, lado):
        n = min(30, len(pontos_local))
        pts = pontos_local[np.argsort(pontos_local[:, 1])][:n]
        x = pts[:, 0]
        y = pts[:, 1]

        coef = np.polyfit(y, x, 2)
        y_contato = np.max(y)
        x_contato = coef[0] * (y_contato ** 2) + coef[1] * y_contato + coef[2]
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

    cv2.drawContours(orig, [contorno + [x_roi, y_roi]], -1, (0, 255, 0), 1)
    cv2.line(orig, (x_roi + p1[0], y_roi + p1[1]), (x_roi + p2[0], y_roi + p2[1]), (0, 0, 255), 1)

    desenhar_curva_ajuste(orig, coef_esq, y_esq, x_roi, y_roi, (255, 255, 0))
    desenhar_curva_ajuste(orig, coef_dir, y_dir, x_roi, y_roi, (255, 255, 0))

    ponto_base_esq = (int(pt_esq[0] + x_roi), int(pt_esq[1] + y_roi))
    ponto_base_dir = (int(pt_dir[0] + x_roi), int(pt_dir[1] + y_roi))

    desenhar_arco_angulo_ajuste(orig, ponto_base_esq, ang_esq, "esq")
    desenhar_arco_angulo_ajuste(orig, ponto_base_dir, ang_dir, "dir")
    desenhar_tangente_ajuste(orig, ponto_base_esq, slope_esq, "esq")
    desenhar_tangente_ajuste(orig, ponto_base_dir, slope_dir, "dir")

    cv2.putText(orig, f"Left: {ang_esq:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(orig, f"Right: {ang_dir:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(orig, f"Average: {ang_medio:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.line(edges_bgr, (p1[0], int(y_base)), (p2[0], int(y_base)), (0, 0, 255), 2)
    roi_final_desenhada = orig[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
    debug_horizontal = np.hstack((edges_bgr, roi_final_desenhada))
    cv2.putText(debug_horizontal, "[ENTER]: Confirmar | [ESC]: Cancelar", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Debug Visual - Pipeline de Analise", debug_horizontal)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Debug Visual - Pipeline de Analise")

    if key == 27:
        return

    app.display_frame(orig)
    app.tree.insert("", "end", values=(nome_arquivo, f"{ang_esq:.2f}", f"{ang_dir:.2f}", f"{ang_medio:.2f}", "N/A"))


def run_circle_fit(app):
    """Run the circular fit workflow for a droplet ROI."""
    if app.img_result_original is None:
        from tkinter import messagebox
        messagebox.showwarning("Warning", "Upload an image first!")
        return

    roi = cv2.selectROI("Select the Drop (Press ENTER)", app.img_result_original, fromCenter=False)
    cv2.destroyWindow("Select the Drop (Press ENTER)")
    x_roi, y_roi, w_roi, h_roi = roi
    if w_roi == 0 or h_roi == 0:
        return

    roi_img = app.img_result_original[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        app.lbl_status.configure(text="Error: No outline in ROI.")
        return

    cnt = max(contours, key=cv2.contourArea)
    points = cnt.reshape(-1, 2)

    x_data = (points[:, 0] + x_roi).astype(float)
    y_data = (points[:, 1] + y_roi).astype(float)

    x_m, y_m = np.mean(x_data), np.mean(y_data)

    def calc_R(xc, yc):
        return np.sqrt((x_data - xc) ** 2 + (y_data - yc) ** 2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, _ = scipy.optimize.leastsq(f_2, center_estimate)
    xc, yc = center_2
    R = calc_R(*center_2).mean()

    y_base = np.max(y_data)
    x_min, x_max = np.min(x_data), np.max(x_data)

    xbas1, ybas1 = x_min - xc, y_base - yc
    xbas2, ybas2 = x_max - xc, y_base - yc
    dx, dy = xbas2 - xbas1, ybas2 - ybas1
    dr = np.sqrt(dx ** 2 + dy ** 2)
    D = xbas1 * ybas2 - xbas2 * ybas1

    discriminant = R ** 2 * dr ** 2 - D ** 2
    if discriminant < 0:
        app.lbl_status.configure(text="Error: Invalid intersection in ROI.")
        return

    delta_root = np.sqrt(discriminant)
    sign_dy = np.sign(dy) if dy != 0 else 1

    pt1_x = (D * dy + sign_dy * dx * delta_root) / dr ** 2 + xc
    pt2_x = (D * dy - sign_dy * dx * delta_root) / dr ** 2 + xc
    pt1_y = (-D * dx + abs(dy) * delta_root) / dr ** 2 + yc
    pt2_y = (-D * dx - abs(dy) * delta_root) / dr ** 2 + yc

    theta1 = abs(np.degrees(-np.pi / 2 + np.arctan2((yc - pt1_y), (xc - pt1_x))))
    theta2 = abs(np.degrees(np.pi / 2 + np.arctan2((yc - pt2_y), (xc - pt2_x))))

    vis_img = app.img_result_original.copy()
    cv2.rectangle(vis_img, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 255, 0), 1)
    cv2.circle(vis_img, (int(xc), int(yc)), int(R), (255, 0, 0), 2)
    cv2.line(vis_img, (int(x_min), int(y_base)), (int(x_max), int(y_base)), (0, 255, 255), 2)
    cv2.circle(vis_img, (int(pt1_x), int(pt1_y)), 6, (0, 0, 255), -1)
    cv2.circle(vis_img, (int(pt2_x), int(pt2_y)), 6, (0, 0, 255), -1)

    app.display_frame(vis_img, update_original=False)
    app.lbl_status.configure(text=f"WCA (ROI): L={theta1:.2f}° | R={theta2:.2f}°")
    app.tree.insert("", "end", values=(len(app.tree.get_children()) + 1, f"{theta1:.2f}", f"{theta2:.2f}", "Circle Fit (ROI)"))


def run_normal_static_angle(app):
    """Run the standard WCA analysis using ROI and two baseline points."""
    method = "WCA_NORMAL"
    if app.ultimo_metodo_selecionado != method:
        for item in app.tree.get_children():
            app.tree.delete(item)
        app.ultimo_metodo_selecionado = method

    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg"), ("All", "*.*")])
    if not path:
        return
    app.video_path = path
    filename = os.path.basename(path)
    image = cv2.imread(path)
    if image is None:
        return
    app.current_mode = "WCA"
    original = image.copy()

    while True:
        roi_box = cv2.selectROI("Select ROI - [Enter] Confirm | [ESC] Cancel", image, showCrosshair=True)
        cv2.destroyWindow("Select ROI - [Enter] Confirm | [ESC] Cancel")
        x_roi, y_roi, width, height = roi_box
        if width == 0 or height == 0:
            if messagebox.askyesno("Invalid ROI", "No region was selected.\nDo you want to try selecting the ROI again?"):
                continue
            return

        preview = image[y_roi:y_roi + height, x_roi:x_roi + width].copy()
        cv2.putText(preview, "Confirm? [Y]-Yes | [R]-Redo ROI", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Confirm ROI", preview)
        key = cv2.waitKey(0)
        cv2.destroyWindow("Confirm ROI")
        if key not in (ord("r"), ord("R")):
            break

    roi = image[y_roi:y_roi + height, x_roi:x_roi + width]
    base_points = []
    roi_preview = roi.copy()
    cv2.imshow("Select Base", roi_preview)
    cv2.setMouseCallback("Select Base", selecionar_ponto_callback, param=[base_points, roi_preview, "Select Base"])
    while len(base_points) < 2:
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Select Base")

    point_left, point_right = base_points
    base_y = max(point_left[1], point_right[1])
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    points = pontos_proximos_base(contour, base_y, app.tangent_static)
    debug_image = roi.copy()
    for x, y in points:
        cv2.circle(debug_image, (x, y), 1, (255, 0, 0), -1)
    cv2.imshow("Selected Points", debug_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Selected Points")

    left_points, right_points = separar_lados(points)
    left_points, right_points = left_points[:40], right_points[:40]
    try:
        left_angle, left_slope = calcular_angulo_dropy(left_points, "esq")
        right_angle, right_slope = calcular_angulo_dropy(right_points, "dir")
        mean_angle = (left_angle + right_angle) / 2
        cv2.drawContours(original, [contour + [x_roi, y_roi]], -1, (0, 255, 0), 1)
        cv2.line(original, (x_roi + point_left[0], y_roi + point_left[1]), (x_roi + point_right[0], y_roi + point_right[1]), (0, 0, 255), 2)
        for point, slope in zip((point_left, point_right), (left_slope, right_slope)):
            dx = 40
            dy = int(slope * dx)
            cv2.line(original, (x_roi + int(point[0] - dx), y_roi + int(point[1] - dy)), (x_roi + int(point[0] + dx), y_roi + int(point[1] + dy)), (0, 255, 255), 1)
        desenhar_arco_angulo(original, (x_roi + point_left[0], y_roi + point_left[1]), left_angle, "esq")
        desenhar_arco_angulo(original, (x_roi + point_right[0], y_roi + point_right[1]), right_angle, "dir")
        cv2.putText(original, f"Left: {left_angle:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(original, f"Right: {right_angle:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(original, f"Average: {mean_angle:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        app.display_frame(original)
        app.lbl_status.configure(text=f"Analysis Completed: Average {mean_angle:.2f} deg")
        app.tree.insert("", "end", values=(filename, f"{left_angle:.2f}", f"{right_angle:.2f}", f"{mean_angle:.2f}"))
    except Exception as error:
        messagebox.showerror("Error", str(error))
