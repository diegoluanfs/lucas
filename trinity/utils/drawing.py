import cv2
import numpy as np


def desenhar_tangentes(frame, p_esq, p_dir, viz_esq, viz_dir):
    h, w = frame.shape[:2]

    if viz_esq is not None and p_esq is not None:
        a, b = np.polyfit(viz_esq[:, 0], viz_esq[:, 1], 1)

        x0, x1 = max(int(p_esq[0] - 20), 0), min(int(p_esq[0] + 20), w - 1)

        y0, y1 = a * x0 + b, a * x1 + b

        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

        cv2.circle(frame, p_esq, 1, (0, 0, 255), -1)

    if viz_dir is not None and p_dir is not None:
        a, b = np.polyfit(viz_dir[:, 0], viz_dir[:, 1], 1)

        x0, x1 = max(int(p_dir[0] - 20), 0), min(int(p_dir[0] + 20), w - 1)

        y0, y1 = a * x0 + b, a * x1 + b

        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 1)

        cv2.circle(frame, p_dir, 1, (255, 0, 255), -1)

    return frame


def desenhar_arco_angulo(img, ponto_base, angulo, lado="esq", raio=15, cor=(255, 0, 0)):
    x0, y0 = ponto_base
    if lado == "esq":
        start, end = 360 - angulo, 360
    else:
        start, end = 180, 180 + angulo
    cv2.ellipse(img, (int(x0), int(y0)), (raio, raio), 0, start, end, cor, 2)
