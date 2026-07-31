import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
