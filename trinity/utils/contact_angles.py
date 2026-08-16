import math

import numpy as np


def angulo_interno_base(contorno, y_base, margem_base=5, n_pontos_tangente=15, min_pontos=4):
    coords = contorno.reshape(-1, 2)

    faixa = coords[np.abs(coords[:, 1] - y_base) <= margem_base]

    if len(faixa) < min_pontos:
        return np.nan, np.nan, None, None, None, None

    x_min = faixa[:, 0].min()
    x_max = faixa[:, 0].max()

    esq = faixa[faixa[:, 0] <= x_min + n_pontos_tangente]
    dir_ = faixa[faixa[:, 0] >= x_max - n_pontos_tangente]

    try:
        a_esq, _ = np.polyfit(esq[:, 0], esq[:, 1], 1)

        ang_esq = 180 - math.degrees(math.atan(a_esq))

        p_esq = (int(esq[:, 0].mean()), int(esq[:, 1].mean()))

        viz_esq = esq

    except Exception:
        ang_esq = np.nan
        p_esq = None
        viz_esq = None

    try:
        a_dir, _ = np.polyfit(dir_[:, 0], dir_[:, 1], 1)

        ang_dir = 180 + math.degrees(math.atan(a_dir))

        p_dir = (int(dir_[:, 0].mean()), int(dir_[:, 1].mean()))

        viz_dir = dir_

    except Exception:
        ang_dir = np.nan
        p_dir = None
        viz_dir = None

    return ang_esq, ang_dir, p_esq, p_dir, viz_esq, viz_dir


def angulo_interno_base_wls(contorno, y_base, margem_base=15, n_pontos_tangente=30, min_pontos=5):
    """
    Calculation of the contact angle using
    Weighted Least Squares (WLS).

    Corrected and more robust version.
    """

    pts = contorno.reshape(-1, 2).astype(float)

    janela_vertical = pts[np.abs(pts[:, 1] - y_base) <= margem_base]

    if len(janela_vertical) < min_pontos:
        janela_vertical = pts[np.abs(pts[:, 1] - y_base) <= (margem_base + 15)]

        if len(janela_vertical) < min_pontos:
            return np.nan, np.nan, None, None, None, None

    x_min = np.min(janela_vertical[:, 0])
    x_max = np.max(janela_vertical[:, 0])

    pts_esq = pts[(pts[:, 0] <= (x_min + n_pontos_tangente)) & (np.abs(pts[:, 1] - y_base) <= (margem_base + 20))]

    pts_dir = pts[(pts[:, 0] >= (x_max - n_pontos_tangente)) & (np.abs(pts[:, 1] - y_base) <= (margem_base + 20))]

    if len(pts_esq) > n_pontos_tangente:
        dist_esq = np.sqrt((pts_esq[:, 0] - x_min) ** 2 + (pts_esq[:, 1] - y_base) ** 2)

        idx = np.argsort(dist_esq)
        pts_esq = pts_esq[idx[:n_pontos_tangente]]

    if len(pts_dir) > n_pontos_tangente:
        dist_dir = np.sqrt((pts_dir[:, 0] - x_max) ** 2 + (pts_dir[:, 1] - y_base) ** 2)

        idx = np.argsort(dist_dir)
        pts_dir = pts_dir[idx[:n_pontos_tangente]]

    def ajustar_tangente_individual(pontos, x_contato, lado="esq"):
        if len(pontos) < 3:
            return np.nan, None

        try:
            x = pontos[:, 0]
            y = pontos[:, 1]

            distancias = np.abs(x - x_contato)

            max_dist = np.max(distancias)

            if max_dist == 0:
                max_dist = 1.0

            pesos = np.exp(-distancias / max_dist)

            grau = 2 if len(pontos) >= 6 else 1

            coeffs = np.polyfit(x, y, grau, w=pesos)

            p = np.poly1d(coeffs)

            derivada = p.deriv()

            m = float(derivada(x_contato))

            angulo_rad = math.atan(abs(m))
            angulo_deg = math.degrees(angulo_rad)

            if lado == "esq":
                ang_final = 180 - angulo_deg if m > 0 else angulo_deg

            else:
                ang_final = 180 - angulo_deg if m < 0 else angulo_deg

            ponto_visual = (int(x_contato), int(p(x_contato)))

            return ang_final, ponto_visual

        except Exception as e:
            print(f"Erro WLS ({lado}): {e}")

            return np.nan, None

    ang_esq, p_esq = ajustar_tangente_individual(pts_esq, x_min, lado="esq")

    ang_dir, p_dir = ajustar_tangente_individual(pts_dir, x_max, lado="dir")

    return ang_esq, ang_dir, p_esq, p_dir, pts_esq, pts_dir


def angulo_interno_base_polinomial(contorno, y_base, margem_base=10, n_pontos_tangente=15, min_pontos=4):
    """
    Calculate the contact angle using a 4th-order polynomial fit.
    according to the method of Bachmann et al. (Langmuir 2013).
    """
    coords = contorno.reshape(-1, 2)

    faixa = coords[np.abs(coords[:, 1] - y_base) <= margem_base]

    if len(faixa) < min_pontos:
        return np.nan, np.nan, None, None, None, None

    x_min = faixa[:, 0].min()
    x_max = faixa[:, 0].max()

    esq = faixa[faixa[:, 0] <= x_min + n_pontos_tangente]
    dir_ = faixa[faixa[:, 0] >= x_max - n_pontos_tangente]

    def calcular_angulo_polinomio(pontos, x_contato, lado="esq"):
        if len(pontos) < 5:
            return np.nan, None

        coeffs = np.polyfit(pontos[:, 0], pontos[:, 1], 4)
        p = np.poly1d(coeffs)

        derivada = p.deriv()
        m = derivada(x_contato)

        angulo_rad = math.atan(m)
        angulo_deg = math.degrees(angulo_rad)

        if lado == "esq":
            return abs(angulo_deg), (int(x_contato), int(p(x_contato)))
        else:
            return abs(angulo_deg), (int(x_contato), int(p(x_contato)))

    try:
        ang_esq, p_esq = calcular_angulo_polinomio(esq, x_min, lado="esq")
    except Exception:
        ang_esq, p_esq = np.nan, None

    try:
        ang_dir, p_dir = calcular_angulo_polinomio(dir_, x_max, lado="dir")
    except Exception:
        ang_dir, p_dir = np.nan, None

    return ang_esq, ang_dir, p_esq, p_dir, esq, dir_


def calcular_angulo_wls(pontos, lado):
    """
    Calculate the contact angle using Weighted Least Squares (WLS).
    It gives more weight to points near the solid-liquid interface.
    """
    if len(pontos) < 5:
        return 0.0

    x = pontos[:, 0].astype(float)
    y = pontos[:, 1].astype(float)

    x_contato = np.min(x) if lado == "esq" else np.max(x)

    distancias = np.abs(x - x_contato)
    pesos = np.exp(-distancias / (np.max(distancias) + 1e-5))

    try:
        coef = np.polyfit(x, y, 2, w=pesos)
        p = np.poly1d(coef)
        dp = np.polyder(p)

        slope = dp(x_contato)
        ang_rad = math.atan(abs(slope))
        ang_deg = math.degrees(ang_rad)

        if lado == "esq":
            return (180 - ang_deg) if slope > 0 else ang_deg
        else:
            return (180 - ang_deg) if slope < 0 else ang_deg
    except Exception:
        return 0.0


def calcular_angulo_quadratico(pontos, lado):
    """Calculate the contact angle from a quadratic tangent fit."""
    if len(pontos) < 15:
        return 0.0

    try:
        x = pontos[:, 0].astype(float)
        y = pontos[:, 1].astype(float)
        coef = np.polyfit(x, y, 2)
        polynomial = np.poly1d(coef)
        derivative = np.polyder(polynomial)

        x_contact = np.min(x) if lado == "esq" else np.max(x)
        slope = derivative(x_contact)
        angle_deg = math.degrees(math.atan(abs(slope)))

        if lado == "esq":
            return (180 - angle_deg) if slope > 0 else angle_deg
        return (180 - angle_deg) if slope < 0 else angle_deg
    except Exception as error:
        print(f"Angle calculation error: {error}")
        return 0.0
