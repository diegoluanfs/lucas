import os

import cv2
import numpy as np
from tkinter import filedialog

from trinity.ui.static_angle_selector import select_three_points


def run_manual_static_angle(app):
    """Run the manual three-point contact-angle workflow."""
    path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg"), ("All", "*.*")])
    if not path:
        return

    app.video_path = path
    nome_arquivo = os.path.basename(path)
    img = cv2.imread(path)
    if img is None:
        return

    orig = img.copy()

    r = cv2.selectROI("Select the Droplet Region", img, showCrosshair=True)
    cv2.destroyWindow("Select the Droplet Region")

    x_roi, y_roi, w_roi, h_roi = r
    if w_roi == 0 or h_roi == 0:
        return

    roi = img[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
    pts = select_three_points(roi)

    if pts:
        p1, p2, p3 = pts

        v_base = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v_face = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        norm_b = np.linalg.norm(v_base)
        norm_f = np.linalg.norm(v_face)

        if norm_b > 0 and norm_f > 0:
            cos_theta = np.dot(v_base, v_face) / (norm_b * norm_f)
            angulo_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angulo_deg = np.degrees(angulo_rad)

            ang_esq = "N/A"
            ang_dir = "N/A"

            if p2[0] < p1[0]:
                ang_esq = angulo_deg
                label_lado = "Esq"
            else:
                ang_dir = angulo_deg
                label_lado = "Dir"

            app.tree.insert("", "end", values=(
                nome_arquivo,
                f"{ang_esq:.2f}" if isinstance(ang_esq, float) else ang_esq,
                f"{ang_dir:.2f}" if isinstance(ang_dir, float) else ang_dir,
                "Manual",
            ))

            p1_g = (p1[0] + x_roi, p1[1] + y_roi)
            p2_g = (p2[0] + x_roi, p2[1] + y_roi)
            p3_g = (p3[0] + x_roi, p3[1] + y_roi)

            cv2.line(orig, p1_g, p2_g, (0, 0, 255), 2)
            cv2.line(orig, p2_g, p3_g, (255, 0, 255), 2)

            texto = f"Ang: {angulo_deg:.2f} deg"
            cv2.putText(orig, texto, (p2_g[0] - 20, p2_g[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            app.display_frame(orig)
