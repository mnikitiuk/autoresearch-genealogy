"""
Комплексний індустріальний аналіз газового ринку України
Джерела:
  - Фізичні потоки ГТС України (2020-2025)
  - Запаси ПСГ (Україна та Європа, 2011-2025)
  - TTF місячні ціни (EUR/MWh, 2024-2026)
  - UEEХ котирування (грн/1000 м³, 2024-2026)

Метод: сезонна декомпозиція, кореляційний аналіз, прогноз Holt-Winters
Вихід : outputs/plots/industrial_analysis.png
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ── шляхи ──────────────────────────────────────────────────────────────
RAW = Path("data/raw")
EXT = Path("data/external")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── кольорова схема ─────────────────────────────────────────────────────
BG     = "#0f1117"
PANEL  = "#1a1d2e"
GRID   = "#2a2a40"
AXIS_C = "#cccccc"
C_TTF  = "#4fc3f7"   # блакитний — TTF
C_UEEX = "#ffb74d"   # помаранчевий — UEEХ
C_PSG  = "#81c784"   # зелений — ПСГ
C_IMP  = "#ce93d8"   # ліловий — імпорт
C_EXP  = "#ef9a9a"   # рожевий — транзит/експорт
C_DOM  = "#fff176"   # жовтий — власний видобуток
C_FORE = "#ff7043"   # оранжевий — прогноз

UAH_EUR   = 42.0
MWH_THCM  = 10.55   # 1 тис. м³ ≈ 10.55 MWh (середня теплотворність)


# ══════════════════════════════════════════════════════════════════════
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════════════════════════════════

print("[1/5] Завантаження даних...")

# --- Фізичні потоки ГТС ---
flows_raw = pd.read_csv(
    RAW / "2025_10_14_fizychni_potoky_v_ukrainskii_hts.csv",
    parse_dates=["date"])

# --- Запаси ПСГ ---
psg_raw = pd.read_csv(
    RAW / "2025_10_14_zapasy_hazu_yevropeiski_psg.csv",
    parse_dates=["date"])

# --- TTF місячні ---
ttf = pd.read_csv(EXT / "ttf_monthly_eur_mwh.csv", parse_dates=["date"])
ttf = ttf.set_index("date")["ttf_eur_mwh"].sort_index()

# --- UEEХ ---
ueex_raw = pd.read_csv(EXT / "ueex_monthly_uah_thcm.csv", parse_dates=["date"])
ueex = ueex_raw.set_index("date")["ueex_uah_thcm_exvat"].sort_index()
ueex_eur = ueex / (UAH_EUR * MWH_THCM)   # перетворення в EUR/MWh


# ══════════════════════════════════════════════════════════════════════
# 2. ОБРОБКА ПОТОКІВ ГТС
# ══════════════════════════════════════════════════════════════════════

print("[2/5] Обробка потоків ГТС...")

# Місячна агрегація
flows = (flows_raw
         .groupby(["date", "interconnection_point_type", "direction"])["transportation_amount"]
         .sum()
         .reset_index())
flows["month"] = flows["date"].dt.to_period("M").dt.to_timestamp()
flows_m = (flows
           .groupby(["month", "interconnection_point_type", "direction"])["transportation_amount"]
           .sum()
           .reset_index())

def get_flow(itype: str, direction: str) -> pd.Series:
    mask = (flows_m["interconnection_point_type"] == itype) & (flows_m["direction"] == direction)
    return (flows_m[mask]
            .set_index("month")["transportation_amount"]
            .sort_index()
            / 1e6)   # тис. м³ -> млрд м³

cross_import = get_flow("Транскордонні точки", "Вхід")
cross_export = get_flow("Транскордонні точки", "Вихід")
psg_inject   = get_flow("Підземні сховища газу", "Вхід")
psg_withdraw = get_flow("Підземні сховища газу", "Вихід")
domestic_prod = get_flow("Мережі газовидобувних підприємств", "Вихід")
distribution  = get_flow("Мережі ОГРМ", "Вихід")

# ══════════════════════════════════════════════════════════════════════
# 3. ОБРОБКА ПСГ УКРАЇНИ
# ══════════════════════════════════════════════════════════════════════

ua_psg = (psg_raw[psg_raw["country"] == "Україна"]
          .copy()
          .sort_values("date")
          .set_index("date"))
ua_psg_daily = ua_psg[["amount_storage", "amount_storage_perc", "injection", "withdrawal"]].dropna(
    subset=["amount_storage"])
ua_psg_monthly = ua_psg_daily.resample("MS").mean()

# Середній рівень заповнення по місяцях (сезонний профіль)
ua_psg_seasonal = ua_psg_monthly["amount_storage_perc"].groupby(
    ua_psg_monthly.index.month).mean()

# ══════════════════════════════════════════════════════════════════════
# 4. СЕЗОННА ДЕКОМПОЗИЦІЯ ТА ТЕСТ ADF
# ══════════════════════════════════════════════════════════════════════

print("[3/5] Статистичний аналіз...")

# Декомпозиція TTF (потрібно >= 2 повних цикли)
ttf_full = ttf.dropna()
decomp_ttf = None
if len(ttf_full) >= 24:
    decomp_ttf = seasonal_decompose(ttf_full, model="additive", period=12, extrapolate_trend="freq")

# ADF-тест для TTF
adf_stat, adf_pval, *_ = adfuller(ttf_full.dropna())
ttf_stationary = adf_pval < 0.05

# Декомпозиція ПСГ
psg_series = ua_psg_monthly["amount_storage_perc"].dropna()
decomp_psg = None
if len(psg_series) >= 24:
    decomp_psg = seasonal_decompose(psg_series, model="additive", period=12, extrapolate_trend="freq")

# ══════════════════════════════════════════════════════════════════════
# 5. ПРОГНОЗ ЗАПАСІВ ПСГ (Holt-Winters)
# ══════════════════════════════════════════════════════════════════════

print("[4/5] Прогноз ПСГ (Holt-Winters)...")

FORECAST_MONTHS = 18   # до березня 2027
psg_fit_series = psg_series.copy()

hw_model = ExponentialSmoothing(
    psg_fit_series,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated",
)
hw_fit   = hw_model.fit(optimized=True, use_brute=True)
hw_pred  = hw_fit.forecast(FORECAST_MONTHS)

# Довірчий інтервал (±1.5 RMSE)
residuals = hw_fit.resid
rmse = float(np.sqrt(np.mean(residuals ** 2)))
hw_lo = hw_pred - 1.5 * rmse
hw_hi = hw_pred + 1.5 * rmse
hw_lo = hw_lo.clip(lower=0)
hw_hi = hw_hi.clip(upper=100)

# ══════════════════════════════════════════════════════════════════════
# 6. КОРЕЛЯЦІЙНА МАТРИЦЯ (місячні дані)
# ══════════════════════════════════════════════════════════════════════

# Об'єднуємо на місячній частоті
merged = pd.DataFrame({
    "TTF (EUR/MWh)": ttf,
    "UEEХ (EUR/MWh)": ueex_eur,
    "ПСГ % наповн.": ua_psg_monthly["amount_storage_perc"],
    "Імпорт (млрд м³)": cross_import,
    "Транзит (млрд м³)": cross_export,
    "ПСГ закачка": psg_inject,
    "ПСГ відбір": psg_withdraw,
}).dropna()
corr_matrix = merged.corr()


# ══════════════════════════════════════════════════════════════════════
# 7. ПОБУДОВА ГРАФІКІВ
# ══════════════════════════════════════════════════════════════════════

print("[5/5] Побудова графіків...")

fig = plt.figure(figsize=(22, 26), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    5, 2,
    figure=fig,
    hspace=0.55,
    wspace=0.30,
    left=0.06, right=0.97,
    top=0.94, bottom=0.04,
)

TITLE_KW  = dict(color=AXIS_C, fontsize=11, fontweight="bold", pad=8)
LABEL_KW  = dict(color=AXIS_C, fontsize=9)
TICK_KW   = dict(colors=AXIS_C, labelsize=8)


def style_ax(ax: plt.Axes, title: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(axis="both", **TICK_KW)
    ax.xaxis.label.set_color(AXIS_C)
    ax.yaxis.label.set_color(AXIS_C)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title, **TITLE_KW)


# ── 7.1  Ціни TTF та UEEХ ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "Ціни TTF та UEEХ (EUR/MWh): 2024-2026")

ax1.plot(ttf.index, ttf.values, color=C_TTF, lw=2.0, label="TTF EUR/MWh")
ttf_confirmed = ttf[ttf.index <= "2025-10-01"]
ax1.plot(ttf_confirmed.index, ttf_confirmed.values, color=C_TTF, lw=2.5, alpha=1.0)

ax1b = ax1.twinx()
ax1b.set_facecolor(PANEL)
ax1b.tick_params(axis="y", **TICK_KW)

ax1b.scatter(ueex_eur.index, ueex_eur.values, color=C_UEEX, s=60, zorder=5,
             label="UEEХ EUR/MWh")
ax1b.plot(ueex_eur.index, ueex_eur.values, color=C_UEEX, lw=1.5, alpha=0.7, linestyle="--")
ax1b.set_ylabel("UEEХ (EUR/MWh)", color=C_UEEX, fontsize=9)
ax1b.tick_params(axis="y", colors=C_UEEX, labelsize=8)

ax1.set_ylabel("TTF (EUR/MWh)", **LABEL_KW)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           facecolor=PANEL, edgecolor=GRID, labelcolor=AXIS_C, fontsize=8, loc="upper left")

adf_label = "стаціонарний" if ttf_stationary else "нестаціонарний"
ax1.annotate(
    f"ADF p={adf_pval:.3f} ({adf_label})",
    xy=(0.98, 0.90), xycoords="axes fraction",
    ha="right", va="top", fontsize=8, color=AXIS_C,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GRID, alpha=0.8))

# ── 7.2  Фізичні потоки ГТС: імпорт та транзит ───────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "Транскордонні потоки ГТС (млрд м³/міс)")

ax2.fill_between(cross_import.index, cross_import.values,
                 alpha=0.35, color=C_IMP, label="_nolabel_")
ax2.plot(cross_import.index, cross_import.values,
         color=C_IMP, lw=1.8, label="Вхід (імпорт+транзит)")

ax2.fill_between(cross_export.index, cross_export.values,
                 alpha=0.25, color=C_EXP, label="_nolabel_")
ax2.plot(cross_export.index, cross_export.values,
         color=C_EXP, lw=1.8, label="Вихід (транзит+експорт)")

ax2.set_ylabel("млрд м³", **LABEL_KW)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=AXIS_C, fontsize=8)

# ── 7.3  Закачка/відбір ПСГ ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "Активність ПСГ (млрд м³/міс)")

ax3.bar(psg_inject.index, psg_inject.values, width=25,
        color=C_PSG, alpha=0.75, label="Закачка")
ax3.bar(psg_withdraw.index, -psg_withdraw.values, width=25,
        color=C_EXP, alpha=0.75, label="Відбір")
ax3.axhline(0, color=GRID, lw=1.0)

ax3.set_ylabel("млрд м³", **LABEL_KW)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=AXIS_C, fontsize=8)

# ── 7.4  Рівень заповнення ПСГ України + прогноз ────────────────────
ax4 = fig.add_subplot(gs[2, :])
style_ax(ax4, "Рівень заповнення ПСГ України (%) + прогноз Holt-Winters до берез. 2027")

hist_perc = ua_psg_monthly["amount_storage_perc"].dropna()
ax4.plot(hist_perc.index, hist_perc.values,
         color=C_PSG, lw=2.0, label="Факт (щомісяця)")

# Сезонний середній профіль (2014-2025)
seasonal_idx = pd.date_range("2015-01-01", periods=12, freq="MS")
ax4.plot(seasonal_idx, [ua_psg_seasonal[m] for m in seasonal_idx.month],
         color=AXIS_C, lw=1.2, linestyle=":", alpha=0.5, label="Сезонна норма")

# Прогноз
ax4.plot(hw_pred.index, hw_pred.values,
         color=C_FORE, lw=2.2, linestyle="--", label="Прогноз Holt-Winters")
ax4.fill_between(hw_pred.index, hw_lo.values, hw_hi.values,
                 alpha=0.18, color=C_FORE, label="Довірчий інтервал ±1.5 RMSE")

# Позначки мінімуму/максимуму
min_val = hist_perc.min()
max_val = hist_perc.max()
min_date = hist_perc.idxmin()
max_date = hist_perc.idxmax()
ax4.annotate(f"Min {min_val:.1f}%", xy=(min_date, min_val),
             xytext=(15, 10), textcoords="offset points",
             color=C_EXP, fontsize=8, arrowprops=dict(arrowstyle="->", color=C_EXP))
ax4.annotate(f"Max {max_val:.1f}%", xy=(max_date, max_val),
             xytext=(5, -18), textcoords="offset points",
             color=C_PSG, fontsize=8, arrowprops=dict(arrowstyle="->", color=C_PSG))

# Горизонтальні зони
ax4.axhspan(0, 15, alpha=0.07, color=C_EXP, label="Критичний рівень (<15%)")
ax4.axhspan(50, 100, alpha=0.05, color=C_PSG, label="Зона надлишку (>50%)")

ax4.set_ylabel("Заповнення (%)", **LABEL_KW)
ax4.set_ylim(0, 100)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax4.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax4.get_xticklabels(), rotation=30, ha="right")
ax4.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=AXIS_C, fontsize=8,
           ncol=3, loc="upper left")

# ── 7.5  Кореляційна матриця ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
style_ax(ax5, "Кореляційна матриця (місячні дані)")
ax5.set_facecolor(PANEL)

labels = [c.replace(" (EUR/MWh)", "").replace(" (млрд м³)", "").replace(" % наповн.", " %")
          for c in corr_matrix.columns]
n = len(labels)
cmap = plt.cm.RdYlGn
im = ax5.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax5.set_xticks(range(n))
ax5.set_yticks(range(n))
ax5.set_xticklabels(labels, rotation=40, ha="right", fontsize=7, color=AXIS_C)
ax5.set_yticklabels(labels, fontsize=7, color=AXIS_C)
for i in range(n):
    for j in range(n):
        val = corr_matrix.values[i, j]
        color = "black" if abs(val) > 0.5 else AXIS_C
        ax5.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(
    labelsize=7, colors=AXIS_C)

# ── 7.6  Сезонний профіль ПСГ (місяць по місяцю) ────────────────────
ax6 = fig.add_subplot(gs[3, 1])
style_ax(ax6, "Сезонний профіль ПСГ України (середнє по місяцях, всі роки)")

months_ua = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер",
             "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]
bar_vals = [ua_psg_seasonal.get(m, np.nan) for m in range(1, 13)]
bar_colors = [C_EXP if v < 30 else (C_UEEX if v < 50 else C_PSG)
              for v in bar_vals]
ax6.bar(months_ua, bar_vals, color=bar_colors, alpha=0.85, width=0.7)
ax6.axhline(30, color=C_EXP, lw=1.2, linestyle="--", alpha=0.7, label="30% поріг")
ax6.axhline(50, color=C_PSG, lw=1.2, linestyle="--", alpha=0.7, label="50% поріг")
ax6.set_ylabel("Заповнення (%)", **LABEL_KW)
ax6.set_ylim(0, 100)
ax6.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=AXIS_C, fontsize=8)

for i, (m, v) in enumerate(zip(months_ua, bar_vals)):
    if not np.isnan(v):
        ax6.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                 fontsize=7, color=AXIS_C)

# ── 7.7  Сезонна декомпозиція TTF ─────────────────────────────────────
ax7 = fig.add_subplot(gs[4, 0])
style_ax(ax7, "TTF: сезонна компонента (міс.)")

if decomp_ttf is not None:
    seasonal_ttf = decomp_ttf.seasonal
    ax7.bar(seasonal_ttf.index, seasonal_ttf.values, width=20,
            color=[C_EXP if v < 0 else C_PSG for v in seasonal_ttf.values],
            alpha=0.8)
    ax7.axhline(0, color=GRID, lw=1.0)
    ax7.set_ylabel("Сезонна компонента", **LABEL_KW)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax7.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax7.get_xticklabels(), rotation=30, ha="right")
else:
    ax7.text(0.5, 0.5, "Недостатньо даних\nдля декомпозиції",
             ha="center", va="center", transform=ax7.transAxes,
             color=AXIS_C, fontsize=10)

# ── 7.8  Власний видобуток vs. розподіл ──────────────────────────────
ax8 = fig.add_subplot(gs[4, 1])
style_ax(ax8, "Власний видобуток vs. розподіл через ОГРМ (млрд м³/міс)")

ax8.fill_between(domestic_prod.index, domestic_prod.values,
                 alpha=0.35, color=C_DOM)
ax8.plot(domestic_prod.index, domestic_prod.values,
         color=C_DOM, lw=1.8, label="Власний видобуток")
ax8.fill_between(distribution.index, distribution.values,
                 alpha=0.20, color=C_TTF)
ax8.plot(distribution.index, distribution.values,
         color=C_TTF, lw=1.8, label="Розподіл ОГРМ")

ax8.set_ylabel("млрд м³", **LABEL_KW)
ax8.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax8.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax8.get_xticklabels(), rotation=30, ha="right")
ax8.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=AXIS_C, fontsize=8)


# ── Заголовок та підпис ───────────────────────────────────────────────
fig.text(
    0.5, 0.965,
    "Комплексний аналіз газового ринку України: потоки ГТС, ПСГ, ціни та прогноз (2020-2027)",
    ha="center", va="center", fontsize=14, fontweight="bold", color=AXIS_C)
fig.text(
    0.5, 0.955,
    "Джерела: ОГТСУ (фізичні потоки 2020-2025), AGSIEU (ПСГ 2011-2025), TTF/UEEХ ринкові дані",
    ha="center", va="center", fontsize=8, color="#888888")

# ── Збереження ────────────────────────────────────────────────────────
out_path = OUT / "industrial_analysis.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=BG)
plt.close(fig)

print(f"\nГотово. Збережено: {out_path}")
print(f"  TTF діапазон: {ttf.min():.1f} - {ttf.max():.1f} EUR/MWh")
print(f"  ПСГ поточний рівень: {hist_perc.iloc[-1]:.1f}%")
print(f"  Прогноз ПСГ на верес. 2026: {hw_pred['2026-09-01']:.1f}%")
print(f"  Кореляція TTF-ПСГ: {corr_matrix.loc['TTF (EUR/MWh)', 'ПСГ % наповн.']:.3f}")
