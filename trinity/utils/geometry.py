import math

import cv2
import numpy as np


def selecionar_ponto_callback(event, x, y, flags, param):
    pontos, img, nome_janela = param
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos.append((x, y))
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow(nome_janela, img)


def pontos_proximos_base(contorno, y_base, altura=40):
    pontos_validos = []

    for (x, y) in contorno:
        # Only points ABOVE the base and within the range.
        if (y_base - altura) <= y <= y_base:
            pontos_validos.append((x, y))

    return np.array(pontos_validos)


def separar_lados(pontos):
    # If 'points' is empty (0 lines), it avoids the index error.
    if pontos.size == 0:
        return np.array([]), np.array([])

    centro = np.mean(pontos[:, 0])
    esq = pontos[pontos[:, 0] < centro]
    dir_ = pontos[pontos[:, 0] >= centro]
    return esq, dir_


def calcular_angulo_dropy(pontos, lado):
    x = pontos[:, 0]
    y = pontos[:, 1]
    coef = np.polyfit(x, y, 2)
    p = np.poly1d(coef)
    dp = np.polyder(p)

    idx = np.argmax(y)
    x0 = x[idx]
    slope = dp(x0)
    ang_base = math.degrees(math.atan(abs(slope)))

    if lado == "esq":
        ang = (180 - ang_base) if slope > 0 else ang_base
    else:
        ang = (180 - ang_base) if slope < 0 else ang_base
    return ang, slope


def calcular_diametro(contorno, frame):
    coords = contorno.reshape(-1, 2)

    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()

    diametro = x_max - x_min

    y_med = int(coords[:, 1].mean())

    p1 = (x_min, y_med)
    p2 = (x_max, y_med)

    cv2.line(frame, p1, p2, (0, 0, 255), 2)

    return diametro, p1, p2
