import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tkinter import messagebox


def _plot_wca_evo(df):
    df_plot = df.copy()
    df_plot[3] = df_plot[3].astype(float)
    df_plot = df_plot.sort_values(by=3, ascending=True)

    y = df_plot[3].values
    x = df_plot[0].values

    plt.plot(x, y, marker="o", linestyle="-", color="b", label="Average Angle")
    for i, txt in enumerate(y):
        plt.annotate(f"{txt:.1f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha="center")

    plt.xticks(rotation=45)
    plt.ylabel("Contact Angle (deg)")
    plt.title("Evolution of the Contact Angle (Sorted by Value)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)


def _plot_wca_evo_two(df):
    df_plot = df.copy()
    df_plot[1] = pd.to_numeric(df_plot[1], errors="coerce")
    df_plot[2] = pd.to_numeric(df_plot[2], errors="coerce")

    df_grouped = df_plot.groupby(0, as_index=False).agg({1: "mean", 2: "mean"})
    df_grouped[3] = df_grouped[[1, 2]].mean(axis=1)
    df_grouped = df_grouped.sort_values(by=3, ascending=True)

    y = df_grouped[3].values
    x = df_grouped[0].values

    plt.plot(x, y, marker="o", linestyle="-", color="b", label="Average Angle")
    for i, txt in enumerate(y):
        if not np.isnan(txt):
            plt.annotate(f"{txt:.1f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha="center")

    plt.xticks(rotation=45)
    plt.ylabel("Contact Angle (deg)")
    plt.title("Evolution of the Contact Angle (Grouped by File)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)


def _plot_wca_hist(df):
    values = pd.to_numeric(df[3], errors="coerce").dropna()
    if values.empty:
        return

    plt.hist(values, bins=15, color="#3b82f6", alpha=0.8)
    plt.xlabel("Contact Angle (deg)")
    plt.ylabel("Frequency")
    plt.title("Contact Angle Distribution")


def _plot_drop_dia(df):
    t = df[0].astype(float)
    d = df[2].astype(float)
    plt.scatter(t, d, color="red", s=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (mm)")
    plt.title("Drop Diameter vs Time")


def _plot_drop_beta(df):
    t = df[0].astype(float)
    beta = df[3].astype(float)
    plt.plot(t, beta, color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("spreading factor beta (D/D0)")
    plt.title("spreading factor beta vs. Time")


def _plot_drop_beta_dia(df):
    t = df[2].astype(float)
    beta = df[3].astype(float)
    plt.plot(t, beta, color="green")
    plt.xlabel("Diameter")
    plt.ylabel("spreading factor beta (D/D0)")
    plt.title("spreading factor beta vs. Diameter")


def _plot_drop_dia_alt(df):
    t = df[2].astype(float)
    beta = df[4].astype(float)
    plt.plot(t, beta, color="green")
    plt.xlabel("Altura")
    plt.ylabel("Diametro")
    plt.title("Altura x diametro")


def _plot_drop_beta_max(df):
    t = df[0].astype(float)
    beta = df[3].astype(float)

    idx_max = beta.idxmax()
    t_expansao = t.loc[:idx_max]
    beta_expansao = beta.loc[:idx_max]

    plt.plot(t_expansao, beta_expansao, color="green", linewidth=2)

    beta_max = beta_expansao.iloc[-1]
    t_max = t_expansao.iloc[-1]
    plt.scatter(t_max, beta_max, color="red", zorder=5)
    plt.annotate(f"Max beta: {beta_max:.2f}", xy=(t_max, beta_max), xytext=(t_max, beta_max * 1.05), horizontalalignment="center")

    plt.xlabel("Time (s)")
    plt.ylabel("spreading factor beta (D/D0)")
    plt.title("Spreading Factor beta vs. Time (Expansion Phase)")
    plt.grid(True, linestyle="--", alpha=0.7)


def _plot_drop_beta_max_fit(df):
    t = df[0].astype(float).values
    beta = df[3].astype(float).values

    idx_max = np.argmax(beta)
    t_exp = t[: idx_max + 1]
    beta_exp = beta[: idx_max + 1]

    coeffs = np.polyfit(t_exp, beta_exp, 2)
    trend_func = np.poly1d(coeffs)
    beta_trend = trend_func(t_exp)

    plt.scatter(t_exp, beta_exp, color="lightgreen", s=10, alpha=0.5, label="Experimental Data")
    plt.plot(t_exp, beta_trend, color="darkgreen", linewidth=2, label="Trend (2nd Order Fit)")

    beta_max = beta_exp[-1]
    t_max = t_exp[-1]
    plt.scatter(t_max, beta_max, color="red", zorder=5)
    plt.annotate(f"Max beta: {beta_max:.2f}", xy=(t_max, beta_max), xytext=(t_max, beta_max * 1.05), horizontalalignment="center")

    plt.xlabel("Time (s)")
    plt.ylabel("spreading factor beta (D/D0)")
    plt.title("Spreading Factor beta: Expansion Trend Analysis")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)


def _plot_drop_beta_max_fit_before_after(df):
    t = df[0].astype(float).values
    beta = df[3].astype(float).values

    idx_max = np.argmax(beta)
    if idx_max == 0 and len(beta) > 1:
        idx_max = np.argmax(beta[1:]) + 1

    t = t - t[0]

    t_max = t[idx_max]
    beta_max = beta[idx_max]
    x_limit_final = t_max * 4

    plt.xlim(0, x_limit_final)
    t_fim_visual = min(t[-1], x_limit_final)
    posicao_centro_retracao = (t_max + t_fim_visual) / 2

    plt.plot(t, beta, color="black", marker="s", markersize=4, markerfacecolor="black", label="Experimental Data")
    plt.axvline(x=t_max, color="gray", linestyle="-", linewidth=1.5)

    plt.ylim(0, max(beta) * 1.2)
    y_max_plot = plt.ylim()[1]
    y_text = y_max_plot * 0.9

    plt.text(t_max / 2, y_text, "Spreading", horizontalalignment="center", fontweight="bold", fontsize=10)
    plt.text(posicao_centro_retracao, y_text, "Retracting", horizontalalignment="center", fontweight="bold", fontsize=10)

    plt.annotate("", xy=(0, y_text * 1.03), xytext=(t_max, y_text * 1.03), arrowprops=dict(arrowstyle="<->", color="black"))
    plt.annotate("", xy=(t_max, y_text * 1.03), xytext=(t_fim_visual, y_text * 1.03), arrowprops=dict(arrowstyle="<->", color="black"))

    y_seta_inferior = y_max_plot * 0.05
    plt.annotate("", xy=(0, y_seta_inferior), xytext=(t_max, y_seta_inferior), arrowprops=dict(arrowstyle="<->", color="red", linestyle="--"))
    plt.text(t_max / 2, y_seta_inferior * 1.3, f"t_max = {t_max:.4f} ms", horizontalalignment="center", color="red", fontweight="bold", fontsize=9)

    plt.xlabel("Time (ms)")
    plt.ylabel("spreading factor beta (D/D0)")
    plt.title("Droplet Impact Dynamics")
    plt.grid(False)
    plt.legend(loc="upper right")


def _plot_his_time_ca(df):
    t = pd.to_numeric(df[1], errors="coerce")
    e = pd.to_numeric(df[6], errors="coerce")
    d = pd.to_numeric(df[7], errors="coerce")

    mask = (~t.isna()) & (~d.isna()) & (~e.isna()) & (e <= 170) & (d <= 170)

    t = t[mask]
    e = np.round(e[mask])
    d = np.round(d[mask])

    plt.scatter(t, e, s=10, label="Left angle (°)", color="blue")
    plt.scatter(t, d, s=10, label="Right angle (°)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Contact Angle (°)")
    plt.title("Contact Angle vs Time")
    plt.legend()


def _plot_his_dia_ca(df):
    df_local = df.copy()
    df_local[2] = pd.to_numeric(df_local[2], errors="coerce")
    df_local[6] = pd.to_numeric(df_local[6], errors="coerce")
    df_local[7] = pd.to_numeric(df_local[7], errors="coerce")

    df_filtered = df_local[(df_local[6] <= 170) & (df_local[7] <= 170)].copy()
    if df_filtered.empty:
        print("Warning: Todos os dados foram filtrados (ângulos > 180°).")
        return

    t = df_filtered[2]
    ang_medio = ((df_filtered[6] + df_filtered[7]) / 2).round().astype(int)
    plt.scatter(t, ang_medio, color="red", s=10)
    plt.xlabel("Diameter (mm)")
    plt.ylabel("Contact Angle (°)")
    plt.title("Drop Diameter vs Contact Angle")


def _plot_his_ang_dia(df):
    ang_esq = pd.to_numeric(df[6], errors="coerce")
    ang_dir = pd.to_numeric(df[7], errors="coerce")
    diametro = pd.to_numeric(df[2], errors="coerce")

    mask = (ang_esq <= 170) & (ang_dir <= 170)
    ang_esq = np.round(ang_esq[mask])
    ang_dir = np.round(ang_dir[mask])
    diametro = diametro[mask]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_esq, color_dir = "tab:blue", "tab:red"
    ax1.set_xlabel("Diameter (px)")
    ax1.set_ylabel("Contact Angle (°)", color="black")
    ax1.scatter(diametro, ang_esq, color=color_esq, label="Left Angle", alpha=0.6)
    ax1.scatter(diametro, ang_dir, color=color_dir, label="Right Angle", alpha=0.6)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.set_title("Correlation: Contact Angles vs. Diameter")



def render_extracted_plot(plot_type, df):
    handlers = {
        "WCA_EVO": _plot_wca_evo,
        "WCA_EVO_TWO": _plot_wca_evo_two,
        "WCA_HIST": _plot_wca_hist,
        "DROP_DIA": _plot_drop_dia,
        "DROP_BETA": _plot_drop_beta,
        "DROP_BETA_DIA": _plot_drop_beta_dia,
        "DROP_DIA_ALT": _plot_drop_dia_alt,
        "DROP_BETA_MAX": _plot_drop_beta_max,
        "DROP_BETA_MAX_FIT": _plot_drop_beta_max_fit,
        "DROP_BETA_MAX_FIT_BEFORE_AFTER": _plot_drop_beta_max_fit_before_after,
        "HIS_TIME_CA": _plot_his_time_ca,
        "HIS_DIA_CA": _plot_his_dia_ca,
        "HIS_ANG_DIA": _plot_his_ang_dia,
    }

    handler = handlers.get(plot_type)
    if handler is None:
        return False

    handler(df)
    return True


def render_legacy_plot(app, plot_type):
    data = []
    for item in app.tree.get_children():
        data.append(app.tree.item(item)["values"])

    if not data:
        messagebox.showwarning("Warning", "The data table is empty!")
        return

    df = pd.DataFrame(data)

    if render_extracted_plot(plot_type, df):
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    plt.figure(figsize=(8, 5))

    if plot_type == "HIS_TIME_CA":
        t = pd.to_numeric(df[1], errors="coerce")
        e = pd.to_numeric(df[6], errors="coerce")
        d = pd.to_numeric(df[7], errors="coerce")

        mask = (~t.isna()) & (~d.isna()) & (~e.isna()) & (e <= 170) & (d <= 170)

        t = t[mask]
        e = np.round(e[mask])
        d = np.round(d[mask])

        plt.scatter(t, e, s=10, label="Left angle (°)", color="blue")
        plt.scatter(t, d, s=10, label="Right angle (°)", color="red")
        plt.xlabel("Time (s)")
        plt.ylabel("Contact Angle (°)")
        plt.title("Contact Angle vs Time")
        plt.legend()

    elif plot_type == "HIS_DIA_CA":
        df[2] = pd.to_numeric(df[2], errors="coerce")
        df[6] = pd.to_numeric(df[6], errors="coerce")
        df[7] = pd.to_numeric(df[7], errors="coerce")

        df_filtered = df[(df[6] <= 170) & (df[7] <= 170)].copy()
        if not df_filtered.empty:
            t = df_filtered[2]
            ang_esq = df_filtered[6]
            ang_dir = df_filtered[7]
            ang_medio = ((ang_esq + ang_dir) / 2).round().astype(int)
            plt.scatter(t, ang_medio, color="red", s=10)
            plt.xlabel("Diameter (mm)")
            plt.ylabel("Contact Angle (°)")
            plt.title("Drop Diameter vs Contact Angle")
        else:
            print("Warning: Todos os dados foram filtrados (ângulos > 180°).")

    elif plot_type == "HIS_DIA_CA_FIT":
        df[2] = np.round(pd.to_numeric(df[2], errors="coerce"), 0)
        df[3] = np.round(pd.to_numeric(df[6], errors="coerce"), 0)
        df[4] = np.round(pd.to_numeric(df[3], errors="coerce"), 0)

        df_filtered = df[(df[3] <= 170) & (df[4] <= 170)].dropna().copy()

        if not df_filtered.empty:
            t = df_filtered[2].values
            ang_esq = df_filtered[3]
            ang_dir = df_filtered[4]
            ang_medio = ((ang_esq + ang_dir) / 2).values

            plt.scatter(t, ang_medio, color="red", s=10, label="Dados Experimentais")

            if len(t) >= app.min_pontos_fit:
                try:
                    z = np.polyfit(t, ang_medio, 1)
                    p = np.poly1d(z)
                    t_linha = np.linspace(t.min(), t.max(), 100)
                    plt.plot(t_linha, p(t_linha), "--", color="blue", linewidth=2, label="Tendência Linear")
                    print(f"Fit realizado: y = {z[0]:.4f}x + {z[1]:.4f} (n={len(t)})")
                except Exception as exc:
                    print(f"Error calculating the adjustment: {exc}")

            plt.xlabel("Diameter (mm)")
            plt.ylabel("Contact Angle (°)")
            plt.title("Drop Diameter vs Contact Angle")
            plt.legend()
        else:
            print("Warning: All data has been filtered. (ângulos > 180°).")

    elif plot_type == "HIS_DIA_TIME":
        tempo = df[1].astype(float)
        diametros = df[2].astype(float)
        plt.plot(tempo, diametros, marker="o", markersize=3, linestyle="-", color="purple")
        plt.xlabel("Time (s)")
        plt.ylabel("Diameter (px)")
        plt.title("Evolution of Droplet Diameter")

    elif plot_type == "HIS_ANG_DIA":
        ang_esq = pd.to_numeric(df[6], errors="coerce")
        ang_dir = pd.to_numeric(df[7], errors="coerce")
        diametro = pd.to_numeric(df[2], errors="coerce")

        mask = (ang_esq <= 170) & (ang_dir <= 170)
        ang_esq = np.round(ang_esq[mask])
        ang_dir = np.round(ang_dir[mask])
        diametro = diametro[mask]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_esq, color_dir = "tab:blue", "tab:red"
        ax1.set_xlabel("Diameter (px)")
        ax1.set_ylabel("Contact Angle (°)", color="black")
        ax1.scatter(diametro, ang_esq, color=color_esq, label="Left Angle", alpha=0.6)
        ax1.scatter(diametro, ang_dir, color=color_dir, label="Right Angle", alpha=0.6)
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.legend(loc="upper left")
        ax1.set_title("Correlation: Contact Angles vs. Diameter")

    elif plot_type == "HIS_MULT":
        t = pd.to_numeric(df[1], errors="coerce")
        diam_base = pd.to_numeric(df[3], errors="coerce")
        av_esq = pd.to_numeric(df[4], errors="coerce")
        av_dir = pd.to_numeric(df[5], errors="coerce")
        ang_esq = pd.to_numeric(df[6], errors="coerce")
        ang_dir = pd.to_numeric(df[7], errors="coerce")

        mask = (~t.isna()) & (~ang_esq.isna()) & (~ang_dir.isna()) & (ang_esq <= 170) & (ang_dir <= 170)

        t = t[mask]
        diam_base = diam_base[mask]
        av_esq = av_esq[mask]
        av_dir = av_dir[mask]
        ang_esq = np.round(ang_esq[mask])
        ang_dir = np.round(ang_dir[mask])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        ax1.scatter(t, ang_esq, s=12, color="blue", label="Left angle. (°)")
        ax1.scatter(t, ang_dir, s=12, color="red", label="Right angle. (°)")
        ax1.set_ylabel(r"$\theta$ [degree]")
        ax1.set_title("Hysteresis Dynamics vs Time")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")

        ax2.scatter(t, diam_base, s=12, color="purple", label="Diâm. Base", facecolors="none", edgecolors="purple")
        ax2.set_ylabel("Base Diameter [px]")
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax3.scatter(t, av_esq, s=12, color="teal", label="Left shift (px)")
        ax3.scatter(t, av_dir, s=12, color="darkorange", label="Right shift (px)")
        ax3.set_xlabel("Time [sec]")
        ax3.set_ylabel("Displacement [px]")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right")

    elif plot_type == "HIS_MULT_FIT":
        from scipy.interpolate import UnivariateSpline

        t = pd.to_numeric(df[1], errors="coerce")
        diam_base = pd.to_numeric(df[3], errors="coerce")
        av_esq = pd.to_numeric(df[4], errors="coerce")
        av_dir = pd.to_numeric(df[5], errors="coerce")
        ang_esq = pd.to_numeric(df[6], errors="coerce")
        ang_dir = pd.to_numeric(df[7], errors="coerce")

        mask = (~t.isna()) & (~ang_esq.isna()) & (~ang_dir.isna()) & (ang_esq <= 170) & (ang_dir <= 170)

        t = t[mask].to_numpy()
        diam_base = diam_base[mask].to_numpy()
        av_esq = av_esq[mask].to_numpy()
        av_dir = av_dir[mask].to_numpy()
        ang_esq = np.round(ang_esq[mask].to_numpy())
        ang_dir = np.round(ang_dir[mask].to_numpy())

        sort_idx = np.argsort(t)
        t, diam_base, av_esq, av_dir, ang_esq, ang_dir = (
            t[sort_idx], diam_base[sort_idx], av_esq[sort_idx], av_dir[sort_idx], ang_esq[sort_idx], ang_dir[sort_idx]
        )

        t_smooth = np.linspace(t.min(), t.max(), 300)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        ax1.scatter(t, ang_esq, s=12, color="blue", alpha=0.4, label="Left angle (°)")
        ax1.scatter(t, ang_dir, s=12, color="red", alpha=0.4, label="Right angle (°)")
        spl_ang_esq = UnivariateSpline(t, ang_esq, s=len(t) * 2)
        spl_ang_dir = UnivariateSpline(t, ang_dir, s=len(t) * 2)
        ax1.plot(t_smooth, spl_ang_esq(t_smooth), color="darkblue", lw=2, label="Trend (Left)")
        ax1.plot(t_smooth, spl_ang_dir(t_smooth), color="darkred", lw=2, label="Trend (Right)")
        ax1.set_ylabel(r"$\theta$ [degree]")
        ax1.set_title("Hysteresis Dynamics vs Time")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")

        ax2.scatter(t, diam_base, s=12, color="purple", alpha=0.4, facecolors="none", edgecolors="purple", label="Diâm. Base")
        spl_diam = UnivariateSpline(t, diam_base, s=len(t) * 1.5)
        ax2.plot(t_smooth, spl_diam(t_smooth), color="indigo", lw=2, label="Trend (Base)")
        ax2.set_ylabel("Base Diameter [px]")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")

        ax3.scatter(t, av_esq, s=12, color="teal", alpha=0.4, label="Left shift (px)")
        ax3.scatter(t, av_dir, s=12, color="darkorange", alpha=0.4, label="Right shift (px)")
        spl_av_esq = UnivariateSpline(t, av_esq, s=len(t) * 2)
        spl_av_dir = UnivariateSpline(t, av_dir, s=len(t) * 2)
        ax3.plot(t_smooth, spl_av_esq(t_smooth), color="darkslategray", lw=2, label="Trend (Left Shift)")
        ax3.plot(t_smooth, spl_av_dir(t_smooth), color="chocolate", lw=2, label="Trend (Right Shift)")
        ax3.set_xlabel("Time [sec]")
        ax3.set_ylabel("Displacement [px]")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right")

    elif plot_type == "HIS_ANG_DIA_FIT":
        frame = pd.to_numeric(df[0], errors="coerce")
        tempo = pd.to_numeric(df[1], errors="coerce")
        diametro_raw = pd.to_numeric(df[2], errors="coerce")
        ang_esq_raw = pd.to_numeric(df[6], errors="coerce")
        ang_dir_raw = pd.to_numeric(df[7], errors="coerce")

        mask_validos = (
            ~diametro_raw.isna() &
            ~ang_esq_raw.isna() &
            ~ang_dir_raw.isna() &
            (ang_esq_raw >= 0) & (ang_esq_raw <= 170) &
            (ang_dir_raw >= 0) & (ang_dir_raw <= 170)
        )

        frame = frame[mask_validos].to_numpy()
        tempo = tempo[mask_validos].to_numpy()
        ang_esq = np.round(ang_esq_raw[mask_validos].to_numpy(), 0)
        ang_dir = np.round(ang_dir_raw[mask_validos].to_numpy(), 0)
        diametro = np.round(diametro_raw[mask_validos].to_numpy(), 0)

        if len(diametro) < 5:
            print("Pontos insuficientes para realizar a segmentação e os ajustes.")
            return

        window = min(15, len(diametro) if len(diametro) % 2 != 0 else len(diametro) - 1)
        if window > 3:
            from scipy.signal import savgol_filter
            diam_suave = savgol_filter(diametro, window_length=window, polyorder=1)
        else:
            diam_suave = diametro.copy()

        derivada_diam = np.gradient(diam_suave)
        limiar_estatico = 0.15
        idx_cresce = np.where(derivada_diam > limiar_estatico)[0]
        idx_constante = np.where(np.abs(derivada_diam) <= limiar_estatico)[0]
        idx_decresce = np.where(derivada_diam < -limiar_estatico)[0]

        fig, ax = plt.subplots(figsize=(9, 6))
        cores_esq = {"cresce": "#1f77b4", "constante": "#17becf", "decresce": "#aec7e8"}
        cores_dir = {"cresce": "#d62728", "constante": "#ff7f0e", "decresce": "#ffbb78"}

        if len(idx_cresce) >= 3:
            d_reg = diametro[idx_cresce]
            ae_reg = ang_esq[idx_cresce]
            ad_reg = ang_dir[idx_cresce]
            ax.scatter(d_reg, ae_reg, color=cores_esq["cresce"], alpha=0.4, edgecolors="none")
            ax.scatter(d_reg, ad_reg, color=cores_dir["cresce"], alpha=0.4, edgecolors="none")
            d_espaco = np.linspace(d_reg.min(), d_reg.max(), 100)
            p_esq = np.polyfit(d_reg, ae_reg, 1)
            p_dir = np.polyfit(d_reg, ad_reg, 1)
            ax.plot(d_espaco, np.polyval(p_esq, d_espaco), color=cores_esq["cresce"], linestyle="-", linewidth=2, label=r"$\theta_{Esq}$ (Crescente)")
            ax.plot(d_espaco, np.polyval(p_dir, d_espaco), color=cores_dir["cresce"], linestyle="--", linewidth=2, label=r"$\theta_{Dir}$ (Crescente)")

        if len(idx_constante) >= 3:
            d_reg = diametro[idx_constante]
            ae_reg = ang_esq[idx_constante]
            ad_reg = ang_dir[idx_constante]
            ax.scatter(d_reg, ae_reg, color=cores_esq["constante"], alpha=0.4, edgecolors="none")
            ax.scatter(d_reg, ad_reg, color=cores_dir["constante"], alpha=0.4, edgecolors="none")
            p_esq_inv = np.polyfit(ae_reg, d_reg, 1)
            ang_espaco_esq = np.linspace(ae_reg.min(), ae_reg.max(), 100)
            diam_fit_esq = np.polyval(p_esq_inv, ang_espaco_esq)
            ax.plot(diam_fit_esq, ang_espaco_esq, color=cores_esq["constante"], linestyle="-", linewidth=2, label=r"$\theta_{Esq}$ (Estático - Y fit)")
            p_dir_inv = np.polyfit(ad_reg, d_reg, 1)
            ang_espaco_dir = np.linspace(ad_reg.min(), ad_reg.max(), 100)
            diam_fit_dir = np.polyval(p_dir_inv, ang_espaco_dir)
            ax.plot(diam_fit_dir, ang_espaco_dir, color=cores_dir["constante"], linestyle="--", linewidth=2, label=r"$\theta_{Dir}$ (Estático - Y fit)")

        if len(idx_decresce) >= 3:
            d_reg = diametro[idx_decresce]
            ae_reg = ang_esq[idx_decresce]
            ad_reg = ang_dir[idx_decresce]
            ax.scatter(d_reg, ae_reg, color=cores_esq["decresce"], alpha=0.4, edgecolors="none")
            ax.scatter(d_reg, ad_reg, color=cores_dir["decresce"], alpha=0.4, edgecolors="none")
            d_espaco = np.linspace(d_reg.min(), d_reg.max(), 100)
            p_esq = np.polyfit(d_reg, ae_reg, 1)
            p_dir = np.polyfit(d_reg, ad_reg, 1)
            ax.plot(d_espaco, np.polyval(p_esq, d_espaco), color=cores_esq["decresce"], linestyle="-", linewidth=2, label=r"$\theta_{Esq}$ (Decrescente)")
            ax.plot(d_espaco, np.polyval(p_dir, d_espaco), color=cores_dir["decresce"], linestyle="--", linewidth=2, label=r"$\theta_{Dir}$ (Decrescente)")

        ax.set_xlabel("Droplet Base Diameter (px)", fontsize=12, fontweight="bold", labelpad=8)
        ax.set_ylabel("Contact Angle (°)", fontsize=12, fontweight="bold", labelpad=8)
        ax.set_title("WLS Hysteresis Correlation: Contact Angles vs. Base Diameter", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.legend(loc="best", fontsize=9, frameon=True, facecolor="#ffffff", edgecolor="#d3d3d3")
        plt.tight_layout()

    elif plot_type == "HIS_ANG_AREA":
        try:
            ang_esq = pd.to_numeric(df[3])
            ang_dir = pd.to_numeric(df[4])
            area = pd.to_numeric(df[6])
        except Exception:
            messagebox.showerror("Erro", "Insufficient data columns for Area.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ang_medio = (ang_esq + ang_dir) / 2
        ax.plot(area, ang_medio, color="black", linestyle="--", label="Média", alpha=0.7)
        ax.set_xlabel("Contact Area (px²)")
        ax.set_ylabel("Contact Angle (°)")
        ax.set_title("Contact Angle vs. Contact Area")
        ax.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
