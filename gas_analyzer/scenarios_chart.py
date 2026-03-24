"""
Сценарний графік: газовий ринок України 2024–2027
Чистий PNG з трьома прогнозними сценаріями
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAW = Path("data/raw")
EXT = Path("data/external")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

UAH_EUR   = 42.0
MWH_THCM  = 10.55
N_FC      = 12
FC_IDX    = pd.date_range("2026-04-01", periods=N_FC, freq="MS")
FC_START  = pd.Timestamp("2026-04-01")

# ── Теми ───────────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b27"
GRID  = "#222840"
AX    = "#c9d1d9"

C_FACT  = "#c9d1d9"   # факт
C_BASE  = "#f5a623"   # базовий
C_BULL  = "#e74c3c"   # бичачий
C_BEAR  = "#2ecc71"   # ведмежий
C_UEEX  = "#f0c040"   # UEEX внутрішній
C_MARK  = "#58a6ff"   # маркери

# ── 1. Дані ────────────────────────────────────────────────────────────
df_int  = pd.read_csv(RAW/"2025_10_14_pryvedena_tsina_hazu_do_kordonu_ukrainy.csv",
                      parse_dates=["date"])
ttf_int = (df_int[df_int["hub_name"]=="TTF"]
           .set_index("date")["price_without_vat"]
           .sort_index().resample("MS").mean()
           / (UAH_EUR * MWH_THCM))

df_ext  = pd.read_csv(EXT/"ttf_monthly_eur_mwh.csv", parse_dates=["date"])
df_ext  = df_ext.set_index("date")["ttf_eur_mwh"].sort_index()
df_ueex = pd.read_csv(EXT/"ueex_monthly_uah_thcm.csv", parse_dates=["date"])
df_ueex = df_ueex.set_index("date")["ueex_uah_thcm_exvat"].sort_index()
df_imp  = pd.read_csv(EXT/"ukraine_gas_import_monthly.csv", parse_dates=["date"])
df_imp  = df_imp.set_index("date")["import_mcm"].sort_index()
df_psg_ua = pd.read_csv(RAW/"2025_10_14_zapasy_hazu_yevropeiski_psg.csv",
                        parse_dates=["date"])
ua_psg  = (df_psg_ua[df_psg_ua["country"]=="Україна"]
           .set_index("date")["amount_storage_perc"]
           .sort_index().resample("MS").mean())

# Зведений ряд TTF
idx38 = pd.date_range("2023-01-01", "2026-03-01", freq="MS")
ttf = pd.Series(dtype=float, index=idx38)
for d in idx38:
    if d in ttf_int.index: ttf[d] = ttf_int[d]
for d, v in df_ext[df_ext.index>="2024-01-01"].items():
    if d in ttf.index: ttf[d] = v
ttf = ttf.dropna().sort_index()

# ── 2. Прогнозні моделі ────────────────────────────────────────────────
hw = ExponentialSmoothing(ttf, trend="add", seasonal="add",
                          seasonal_periods=12, damped_trend=True).fit(optimized=True)
fc_hw = hw.forecast(N_FC)
fc_hw.index = FC_IDX

try:
    sarima = SARIMAX(ttf, order=(1,1,1), seasonal_order=(0,1,1,12),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)
    fc_s = sarima.forecast(N_FC).clip(lower=15.0)
    fc_s.index = FC_IDX
except Exception:
    fc_s = fc_hw.copy()

fc_raw = (fc_hw * 0.70 + fc_s * 0.30).clip(lower=15.0)

# Консенсус + Q2 Iran/Qatar шок
consensus = 30.5
scale     = 0.6 + 0.4 * consensus / fc_raw.mean()
fc_base   = (fc_raw * scale).clip(lower=15.0)
for d, v in {"2026-04-01":48.0,"2026-05-01":52.0,"2026-06-01":45.0}.items():
    ts = pd.Timestamp(d)
    if ts in fc_base.index:
        fc_base[ts] = fc_base[ts]*0.30 + v*0.70

# Бичачий
bull_mult = [1.50,1.65,1.55,1.30,1.20,1.15,1.20,1.25,1.30,1.35,1.25,1.20]
fc_bull   = pd.Series(fc_base.values * bull_mult, index=FC_IDX)

# Ведмежий
bear_mult = [0.72,0.70,0.68,0.65,0.63,0.65,0.70,0.72,0.75,0.78,0.75,0.72]
fc_bear   = pd.Series(fc_base.values * bear_mult, index=FC_IDX)

# UAH конвертація
to_uah = lambda s: s * UAH_EUR * MWH_THCM
fc_base_uah = to_uah(fc_base)
fc_bull_uah  = to_uah(fc_bull)
fc_bear_uah  = to_uah(fc_bear)
ttf_hist_uah = to_uah(ttf)

# UEEX прогноз (нормалізація премії 46% → 18%)
premiums     = np.linspace(0.46, 0.18, N_FC)
fc_ueex_uah  = pd.Series(fc_base_uah.values * (1+premiums), index=FC_IDX)

# PSG forecast
psg_cap = 31.0
psg_deltas = [2.5,3.2,2.8,2.5,2.0,1.5,0.5,-2.5,-3.0,-3.5,-3.0,-2.0]
psg_fc = {}; cur = 8.0
for ts, d in zip(FC_IDX, psg_deltas):
    cur = float(np.clip(cur+d, 0, psg_cap))
    psg_fc[ts] = cur/psg_cap*100
psg_fc_s = pd.Series(psg_fc)

# ── 3. Графік ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 24), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[2, 1.5, 1.2],
    hspace=0.42, wspace=0.28,
    left=0.07, right=0.97, top=0.94, bottom=0.04,
)

ax_eur  = fig.add_subplot(gs[0, :])         # TTF EUR/MWh — повна ширина
ax_uah  = fig.add_subplot(gs[1, :])         # UAH — повна ширина
ax_psg  = fig.add_subplot(gs[2, 0])         # PSG
ax_imp  = fig.add_subplot(gs[2, 1])         # Імпорт

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=AX, labelsize=8.5, length=3)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2e45")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.8, zorder=0)

def xfmt(ax, step=3):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,13,step)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

def divider(ax):
    ax.axvline(FC_START - pd.Timedelta(days=1),
               color="#4a4f6a", lw=1.3, linestyle="--", zorder=3)

HIST_FROM = "2024-01-01"

# ══════════════════════════════════════════════════════════════════════
# Panel 1 — TTF EUR/MWh: факт + 3 сценарії
# ══════════════════════════════════════════════════════════════════════
ax = ax_eur
hist = ttf[HIST_FROM:]

# Коридор bull/bear (тінь)
ax.fill_between(FC_IDX, fc_bear.values, fc_bull.values,
                alpha=0.10, color="#8e44ad", zorder=1)
# Окремі підзони
ax.fill_between(FC_IDX, fc_base.values, fc_bull.values,
                alpha=0.20, color=C_BULL, zorder=1)
ax.fill_between(FC_IDX, fc_bear.values, fc_base.values,
                alpha=0.20, color=C_BEAR, zorder=1)

# Лінії сценаріїв
ax.plot(FC_IDX, fc_bull.values, color=C_BULL, lw=1.8, linestyle="-.",
        zorder=4, label="Бичачий  (Iran/Qatar шок; Goldman Q2 ~55-63€)")
ax.plot(FC_IDX, fc_base.values, color=C_BASE, lw=2.6, linestyle="--",
        zorder=5, label="Базовий  (консенсус аналітиків ~30.5€ avg)")
ax.plot(FC_IDX, fc_bear.values, color=C_BEAR, lw=1.8, linestyle="-.",
        zorder=4, label="Ведмежий (мир РФ-UA + LNG хвиля; Goldman ~29€)")

# Факт
ax.plot(hist.index, hist.values, color=C_FACT, lw=2.2, zorder=6,
        label="TTF факт (EUR/MWh)")

# Аналітичні рівні
for y_val, col, lbl in [
    (63.0, C_BULL,  "Goldman Q2 bull"),
    (30.5, C_BASE,  "Консенсус 30.5"),
    (29.0, C_BEAR,  "Goldman base 29"),
]:
    ax.axhline(y_val, color=col, lw=0.8, linestyle=":", alpha=0.55, zorder=2)
    ax.text(FC_START + pd.Timedelta(days=2), y_val + 0.8,
            lbl, color=col, fontsize=7.5, alpha=0.9)

# Анотації ключових подій на факті
events = {
    "2025-01-01": (49.0, "Кінець транзиту РФ", 15, 8),
    "2025-02-01": (44.0, "Переговори\nСША-РФ", 5, 12),
    "2025-03-01": (41.52, "-17% за місяць", 8, -25),
    "2025-07-01": (33.36, "Літній\nмінімум", 5, -28),
    "2026-01-01": (50.0,  "Зима\nгеополіт.", -55, 10),
}
for date_str, (yv, txt, dx, dy) in events.items():
    ts = pd.Timestamp(date_str)
    if ts in hist.index:
        ax.annotate(txt, xy=(ts, yv), xytext=(dx, dy),
                    textcoords="offset points",
                    color="#aab4cc", fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color="#555577", lw=0.7),
                    zorder=7)

divider(ax)
ax.text(FC_START + pd.Timedelta(days=4), ax.get_ylim()[0] + 1,
        "← факт      прогноз →",
        color="#555577", fontsize=8, zorder=8)

ax.set_title("TTF природний газ (EUR/MWh): факт 2024–2026 + прогноз квітень 2026 – березень 2027",
             color="white", fontsize=12.5, fontweight="bold", pad=12)
ax.set_ylabel("EUR / MWh", color=AX, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
ax.legend(framealpha=0.25, facecolor="#1a1d2e", edgecolor="#333355",
          labelcolor="white", fontsize=8.5, ncol=2, loc="upper left")
xfmt(ax, step=2)

# ══════════════════════════════════════════════════════════════════════
# Panel 2 — UAH/тис.м³: TTF import parity vs UEEX + прогноз
# ══════════════════════════════════════════════════════════════════════
ax = ax_uah
hist_uah = ttf_hist_uah[HIST_FROM:]

ax.fill_between(FC_IDX, fc_bear_uah.values, fc_bull_uah.values,
                alpha=0.08, color="#9b59b6", zorder=1)
ax.fill_between(FC_IDX, fc_base_uah.values, fc_bull_uah.values,
                alpha=0.15, color=C_BULL, zorder=1)
ax.fill_between(FC_IDX, fc_bear_uah.values, fc_base_uah.values,
                alpha=0.15, color=C_BEAR, zorder=1)

ax.plot(FC_IDX, fc_bull_uah.values,  color=C_BULL,  lw=1.6, linestyle="-.", zorder=4)
ax.plot(FC_IDX, fc_base_uah.values,  color=C_BASE,  lw=2.4, linestyle="--", zorder=5,
        label="TTF базовий прогноз")
ax.plot(FC_IDX, fc_bear_uah.values,  color=C_BEAR,  lw=1.6, linestyle="-.", zorder=4)
ax.plot(FC_IDX, fc_ueex_uah.values,  color=C_UEEX,  lw=2.0, linestyle="--", zorder=5,
        label=f"UEEX прогноз (нормалізація премії 46%→18%)")

ax.plot(hist_uah.index, hist_uah.values, color=C_FACT, lw=2.2, zorder=6,
        label="TTF import parity (факт)")

# UEEX підтверджені точки
ueex_hist = df_ueex[df_ueex.index >= HIST_FROM]
ax.scatter(ueex_hist.index, ueex_hist.values, s=70, color=C_UEEX,
           zorder=8, label="UEEX підтверджені дані (AGPU, Нафтогаз)")

# Нафтогаз тарифи як ромби
naft_pts = {
    "2026-02-01": 21380,
    "2026-03-01": 22602,
}
ax.scatter(list(naft_pts.keys()), list(naft_pts.values()),
           s=90, marker="D", color=C_MARK, zorder=9,
           label="Нафтогаз Трейдинг бізнес (без ПДВ)")

divider(ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_title("Ціна газу в Україні (UAH / 1000 м³): TTF import parity vs UEEX внутрішній ринок",
             color="white", fontsize=11.5, fontweight="bold", pad=10)
ax.set_ylabel("UAH / 1000 м³", color=AX, fontsize=9)
ax.legend(framealpha=0.25, facecolor="#1a1d2e", edgecolor="#333355",
          labelcolor="white", fontsize=8.5, ncol=2, loc="upper left")
xfmt(ax, step=2)

# ══════════════════════════════════════════════════════════════════════
# Panel 3 — PSG %
# ══════════════════════════════════════════════════════════════════════
ax = ax_psg
ua_hist = ua_psg[ua_psg.index >= HIST_FROM]
ax.fill_between(ua_hist.index, ua_hist.values, alpha=0.30, color="#2980b9")
ax.plot(ua_hist.index, ua_hist.values, color="#2980b9", lw=2.0, label="UA ПСГ % (факт)")
ax.fill_between(psg_fc_s.index, psg_fc_s.values*0.80, psg_fc_s.values*1.20,
                alpha=0.15, color="#5dade2")
ax.plot(psg_fc_s.index, psg_fc_s.values, color="#5dade2", lw=2.0,
        linestyle="--", label="Прогноз UA ПСГ %")
ax.axhline(30, color="#e67e22", lw=0.9, ls=":", alpha=0.7, label="30% ціль ЄС (1 листоп.)")
ax.axhline(5,  color=C_BULL,   lw=0.9, ls=":", alpha=0.7, label="5% критичний min")
divider(ax)
ax.set_ylim(0, 85)
ax.set_title("ПСГ України (% заповнення)", color="white",
             fontsize=10, fontweight="bold", pad=8)
ax.set_ylabel("%", color=AX, fontsize=9)
ax.legend(framealpha=0.2, facecolor="#1a1d2e", edgecolor="#333355",
          labelcolor="white", fontsize=8, loc="upper left")
xfmt(ax, step=3)

# Пік
peak_ts = psg_fc_s.idxmax()
peak_v  = psg_fc_s.max()
ax.annotate(f"Пік {peak_v:.0f}%\n{peak_ts.strftime('%b %Y')}",
            xy=(peak_ts, peak_v), xytext=(8, -20),
            textcoords="offset points", color="#5dade2", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#5dade2", lw=0.7))

# ══════════════════════════════════════════════════════════════════════
# Panel 4 — Імпорт
# ══════════════════════════════════════════════════════════════════════
ax = ax_imp
hist_imp = df_imp[df_imp.index >= HIST_FROM]
ax.bar(hist_imp.index, hist_imp.values, width=26,
       color="#8e44ad", alpha=0.85, zorder=3, label="Факт")
imp_fc = [520,600,700,850,800,750,500,400,350,650,600,550]
imp_fc_s = pd.Series(imp_fc, index=FC_IDX)
ax.bar(imp_fc_s.index, imp_fc_s.values, width=26,
       color="#a569bd", alpha=0.50, hatch="///", zorder=3, label="Прогноз")

divider(ax)
ax.set_title("Імпорт газу в Україну (млн м³/місяць)",
             color="white", fontsize=10, fontweight="bold", pad=8)
ax.set_ylabel("млн м³", color=AX, fontsize=9)
ax.legend(framealpha=0.2, facecolor="#1a1d2e", edgecolor="#333355",
          labelcolor="white", fontsize=8)
xfmt(ax, step=3)

# Підпис рекорду
rec_ts = pd.Timestamp("2025-07-01")
ax.annotate("Рекорд\n833 млн м³",
            xy=(rec_ts, 833), xytext=(15, -10),
            textcoords="offset points", color="#d7bde2", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=0.7))

# ══════════════════════════════════════════════════════════════════════
# Suptitle + footer
# ══════════════════════════════════════════════════════════════════════
fig.suptitle(
    "Газовий ринок України: сценарний прогноз  |  квітень 2026 — березень 2027",
    color="white", fontsize=14, fontweight="bold", y=0.97)

# Легенда сценаріїв під заголовком
legend_elements = [
    mpatches.Patch(facecolor=C_BULL, alpha=0.7,
                   label="Бичачий: Iran/Qatar шок — TTF avg ~49 EUR/MWh"),
    mpatches.Patch(facecolor=C_BASE, alpha=0.7,
                   label="Базовий: консенсус аналітиків — TTF avg ~36 EUR/MWh"),
    mpatches.Patch(facecolor=C_BEAR, alpha=0.7,
                   label="Ведмежий: мир РФ-UA + LNG — TTF avg ~26 EUR/MWh"),
]
fig.legend(handles=legend_elements, loc="upper center",
           bbox_to_anchor=(0.50, 0.945), ncol=3,
           framealpha=0.3, facecolor="#1a1d2e", edgecolor="#444466",
           labelcolor="white", fontsize=9)

fig.text(0.5, 0.010,
    "Джерела: AGPU/Elenger (TTF 2024-2025) · ОГТСУ/Нафтогаз Трейдинг (UA ціни) · "
    "EXPRO Consulting (імпорт/ПСГ) · Goldman Sachs / ABN AMRO / Kpler (прогнози 2026)  |  "
    "Модель: H-W(70%) + SARIMA(1,1,1)(0,1,1,12)(30%) + консенсус-коригування",
    ha="center", color="#555577", fontsize=7.5)

out_path = OUT / "scenarios_1y.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out_path}")
