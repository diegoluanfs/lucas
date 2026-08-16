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


def desenhar_curva_ajuste(img, coef, pts_y, offset_x, offset_y, cor=(255, 255, 0)):
    """Draw the fitted parabola x = ay^2 + by + c on an image."""
    y_min = int(min(pts_y))
    y_max = int(max(pts_y))

    curva_pts = []
    for y_val in range(y_min, y_max + 1):
        x_val = coef[0] * (y_val**2) + coef[1] * y_val + coef[2]
        curva_pts.append([int(x_val + offset_x), int(y_val + offset_y)])

    if len(curva_pts) > 1:
        pts_array = np.array(curva_pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_array], False, cor, 1)


def desenhar_tangente_ajuste(img, ponto_base, slope_dx_dy, lado, cor=(255, 0, 255)):
    """Draw the fitted tangent segment from a contact point."""
    x0, y0 = ponto_base
    length = 50
    dy = -length
    dx = dy * slope_dx_dy
    p2 = (int(x0 + dx), int(y0 + dy))
    cv2.line(img, (int(x0), int(y0)), p2, cor, 1)


def desenhar_arco_angulo_ajuste(img, centro, angulo_valor, lado, cor=(0, 255, 0)):
    """Draw the contact-angle arc used by the fitted static analysis."""
    raio = 30
    if lado == "esq":
        start_angle = 0
        end_angle = -angulo_valor
    else:
        start_angle = 180
        end_angle = 180 + angulo_valor

    cv2.ellipse(img, centro, (raio, raio), 0, start_angle, end_angle, cor, 1)
