import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed model -> colour mapping (Okabe-Ito, colour-blind safe); identity never depends on rank.
COLORS = {
    "actual": "#222222",
    "lightgbm": "#0072B2",
    "lag28": "#E69F00",
    "trailing28_mean": "#009E73",
    "weekly_seasonal_naive": "#CC79A7",
}
LABELS = {
    "actual": "Actual",
    "lightgbm": "LightGBM",
    "lag28": "Lag-28 naive",
    "trailing28_mean": "Trailing-28 mean",
    "weekly_seasonal_naive": "Weekly seasonal naive",
}
INK, MUTED, GRID = "#222222", "#666666", "#E5E5E5"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def feature_importance(imp, path, top=20):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(d["feature"], d["gain"], color=COLORS["lightgbm"], height=0.6)
    _style(ax, f"LightGBM feature importance (gain), top {len(d)}", "Total gain", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scaled_rmse_by_horizon(df, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ["lightgbm", "lag28", "trailing28_mean", "weekly_seasonal_naive"]:
        ax.plot(df["horizon"], df[name], color=COLORS[name], linewidth=2, marker="o", markersize=3,
                label=LABELS[name])
    _style(ax, "Scaled RMSE by forecast horizon (final holdout)", "Forecast horizon (days after origin)",
           "Scaled RMSE")
    ax.set_xticks(range(1, 29, 3))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def daily_forecast(df, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ["actual", "lightgbm", "trailing28_mean"]:
        ax.plot(df["date"], df[name], color=COLORS[name], linewidth=2 if name != "trailing28_mean" else 1.5,
                linestyle="--" if name == "trailing28_mean" else "-", label=LABELS[name])
    _style(ax, "Total daily units, eligible CA FOODS_3 series (final holdout)", "", "Units sold")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=INK)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
