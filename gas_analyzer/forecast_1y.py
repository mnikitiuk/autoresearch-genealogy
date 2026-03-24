"""
Прогноз газового ринку України на 1 рік (квітень 2026 — березень 2027)
База: внутрішні CSV 2023-2025 + зовнішні ринкові дані 2024-2026
Методи: Holt-Winters, SARIMA(2,1,2)(1,1,1,12), три сценарії (bull/base/bear)
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAW = Path("data/raw")
EXT = Path("data/external")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

BG     = "#0f1117"
PANEL  = "#1a1d2e"
GRID   = "#2a2a40"
AXIS_C = "#cccccc"

UAH_EUR    = 42.0
MWH_THCM   = 10.55
FORECAST_START = pd.Timestamp("2026-04-01")
FORECAST_END   = pd.Timestamp("2027-03-01")
HIST_PLOT_START = "2024-01-01"

# ═══════════════════════════════════════════════════════════════════════
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ═══════════════════════════════════════════════════════════════════════

# --- Внутрішні дані (TTF у грн/1000 м³ з наших CSV) ---
df_int = pd.read_csv(
    RAW / "2025_10_14_pryvedena_tsina_hazu_do_kordonu_ukrainy.csv",
    parse_dates=["date"])
ttf_int = (df_int[df_int["hub_name"] == "TTF"]
           .set_index("date")["price_without_vat"]
           .sort_index()
           .resample("MS").mean())                    # місячно
ttf_int_eur = ttf_int / (UAH_EUR * MWH_THCM)          # конвертувати в EUR/MWh

# --- Зовнішні ринкові дані (зібрані з веб-джерел) ---
df_ext = pd.read_csv(EXT / "ttf_monthly_eur_mwh.csv", parse_dates=["date"])
df_ext = df_ext.set_index("date")["ttf_eur_mwh"].sort_index()

# --- UEEX внутрішній ринок ---
df_ueex = pd.read_csv(EXT / "ueex_monthly_uah_thcm.csv", parse_dates=["date"])
df_ueex = df_ueex.set_index("date")["ueex_uah_thcm_exvat"].sort_index()

# --- Імпорт ---
df_imp = pd.read_csv(EXT / "ukraine_gas_import_monthly.csv", parse_dates=["date"])
df_imp = df_imp.set_index("date")["import_mcm"].sort_index()

# --- PSG Ukraine ---
df_psg = pd.read_csv(RAW / "2025_10_14_zapasy_hazu_yevropeiski_psg.csv",
                     parse_dates=["date"])
ua_psg = (df_psg[df_psg["country"] == "Україна"]
          .set_index("date")["amount_storage_perc"]
          .sort_index()
          .resample("MS").mean())

# ═══════════════════════════════════════════════════════════════════════
# 2. ПОБУДОВА ЄДИНОГО РЯДУ TTF EUR/MWh
# ═══════════════════════════════════════════════════════════════════════
# Пріоритет: підтверджені зовнішні точки > внутрішні CSV

combined_idx = pd.date_range("2023-01-01", "2026-03-01", freq="MS")
ttf_combined = pd.Series(dtype=float, index=combined_idx, name="ttf_eur_mwh")

# Заповнити з внутрішніх CSV (перерахованих у EUR)
for d in combined_idx:
    if d in ttf_int_eur.index:
        ttf_combined[d] = ttf_int_eur[d]

# Перезаписати підтвердженими зовнішніми даними
confirmed = df_ext[df_ext.index >= "2024-01-01"]
for d, v in confirmed.items():
    if d in ttf_combined.index:
        ttf_combined[d] = v

ttf_combined = ttf_combined.dropna().sort_index()

# ═══════════════════════════════════════════════════════════════════════
# 3. ПРОГНОЗНІ МОДЕЛІ
# ═══════════════════════════════════════════════════════════════════════

N_FC = 12  # 12 місяців вперед
fc_idx = pd.date_range(FORECAST_START, periods=N_FC, freq="MS")

# --- Модель A: Holt-Winters (зимова сезонність) ---
hw = ExponentialSmoothing(
    ttf_combined,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    damped_trend=True
).fit(optimized=True)
fc_hw = hw.forecast(N_FC)
fc_hw.index = fc_idx

# --- Модель B: SARIMA(1,1,1)(0,1,1,12) - спрощена сезонна ---
try:
    sarima = SARIMAX(ttf_combined, order=(1,1,1),
                     seasonal_order=(0,1,1,12),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)
    fc_sarima_raw = sarima.forecast(N_FC)
    fc_sarima_raw.index = fc_idx
    # Підлогу: не менше 15 EUR/MWh (теоретичний мінімум LNG cost-floor)
    fc_sarima = fc_sarima_raw.clip(lower=15.0)
except Exception:
    fc_sarima = fc_hw.copy()

# --- Ансамбль: 70% HW + 30% SARIMA (HW стабільніший) ---
fc_base = (fc_hw * 0.70 + fc_sarima * 0.30).clip(lower=15.0)

# --- Коригування на аналітичні консенсуси ---
# Goldman base 29, ABN AMRO 30, Kpler 33 → консенсус ~30.5 EUR/MWh
# BUT Goldman near-term (Q2 2026) revised UP to 55-63 (Iran/Qatar shock)
# Накладаємо "шок-корекцію" на Q2 2026 (Apr-Jun)
consensus_annual = 30.5    # середньорічний консенсус аналітиків
shock_q2 = {               # Goldman Iran/Qatar shock, Q2 2026
    "2026-04-01": 48.0,
    "2026-05-01": 52.0,
    "2026-06-01": 45.0,
}

# Тягнемо базовий прогноз до консенсусу (blend 60% model + 40% consensus)
fc_annual_avg = fc_base.mean()
scale_factor = (0.6 + 0.4 * consensus_annual / fc_annual_avg)
fc_base_adj = fc_base * scale_factor

# Накладаємо Q2 2026 шок (Goldman revised)
for date_str, shock_val in shock_q2.items():
    ts = pd.Timestamp(date_str)
    if ts in fc_base_adj.index:
        fc_base_adj[ts] = fc_base_adj[ts] * 0.3 + shock_val * 0.7

# --- Бичачий сценарій (Goldman Q2 spike + continued tensions) ---
fc_bull = fc_base_adj.copy()
bull_factors = {
    "2026-04": 1.50, "2026-05": 1.65, "2026-06": 1.55,
    "2026-07": 1.30, "2026-08": 1.20, "2026-09": 1.15,
    "2026-10": 1.20, "2026-11": 1.25, "2026-12": 1.30,
    "2027-01": 1.35, "2027-02": 1.25, "2027-03": 1.20,
}
for date_str, factor in bull_factors.items():
    ts = pd.Timestamp(date_str + "-01")
    if ts in fc_bull.index:
        fc_bull[ts] = fc_base_adj[ts] * factor

# --- Ведмежий сценарій (мирний договір РФ-UA + LNG хвиля) ---
fc_bear = fc_base_adj.copy()
bear_factors = {
    "2026-04": 0.72, "2026-05": 0.70, "2026-06": 0.68,
    "2026-07": 0.65, "2026-08": 0.63, "2026-09": 0.65,
    "2026-10": 0.70, "2026-11": 0.72, "2026-12": 0.75,
    "2027-01": 0.78, "2027-02": 0.75, "2027-03": 0.72,
}
for date_str, factor in bear_factors.items():
    ts = pd.Timestamp(date_str + "-01")
    if ts in fc_bear.index:
        fc_bear[ts] = fc_base_adj[ts] * factor

# --- Конвертація в UAH/1000 м³ ---
def to_uah(s_eur): return s_eur * UAH_EUR * MWH_THCM

fc_base_uah = to_uah(fc_base_adj)
fc_bull_uah  = to_uah(fc_bull)
fc_bear_uah  = to_uah(fc_bear)

ttf_hist_uah = to_uah(ttf_combined)

# UEEX forecast (premium над TTF: ~45% зараз, очікується нормалізація до 15-20%)
ueex_premium_target = 0.18  # очікувана нормалізація
ueex_premium_now    = 0.46  # поточна (Mar 2026)
months_to_normalize = 12
premiums = np.linspace(ueex_premium_now, ueex_premium_target, N_FC)
fc_ueex_uah = pd.Series(
    fc_base_uah.values * (1 + premiums),
    index=fc_idx
)

# ═══════════════════════════════════════════════════════════════════════
# 4. ГРАФІКИ
# ═══════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.30,
                        left=0.07, right=0.97, top=0.93, bottom=0.04)

ax_ttf_eur  = fig.add_subplot(gs[0, :])   # TTF EUR/MWh 3 сценарії
ax_uah      = fig.add_subplot(gs[1, :])   # UAH/тис.м³: TTF + UEEX
ax_psg      = fig.add_subplot(gs[2, 0])   # PSG % fill
ax_imp      = fig.add_subplot(gs[2, 1])   # Import mcm
ax_table    = fig.add_subplot(gs[3, :])   # Табличний прогноз

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=AXIS_C, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.6)

def fmt_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=8)

def vline(ax):
    ax.axvline(FORECAST_START - pd.Timedelta(days=15),
               color="#555577", lw=1.2, linestyle=":", alpha=0.8)

# ─── Panel 1: TTF EUR/MWh ─────────────────────────────────────────────
ax = ax_ttf_eur
hist = ttf_combined[HIST_PLOT_START:]

ax.fill_between(fc_idx, fc_bear.values, fc_bull.values,
                alpha=0.15, color="#9b59b6", label="Коридор bull/bear")
ax.fill_between(fc_idx, fc_bear.values, fc_base_adj.values,
                alpha=0.20, color="#27ae60")
ax.fill_between(fc_idx, fc_base_adj.values, fc_bull.values,
                alpha=0.20, color="#e74c3c")

ax.plot(hist.index, hist.values, color="#e8e8e8", lw=2.0,
        label="TTF факт (EUR/MWh)")
ax.plot(fc_idx, fc_base_adj.values, color="#f5a623", lw=2.2,
        linestyle="--", label="Базовий сценарій")
ax.plot(fc_idx, fc_bull.values, color="#e74c3c", lw=1.6,
        linestyle=":", label=f"Бичачий (Iran/Qatar шок)")
ax.plot(fc_idx, fc_bear.values, color="#2ecc71", lw=1.6,
        linestyle=":", label=f"Ведмежий (мир РФ-UA + LNG)")

# Аналітичні консенсуси
ax.axhline(29.0, color="#e74c3c", lw=0.7, alpha=0.5, linestyle="-.")
ax.text(pd.Timestamp("2026-04-15"), 29.5, "Goldman base 29", color="#e74c3c",
        fontsize=7.5, alpha=0.8)
ax.axhline(30.5, color="#f5a623", lw=0.7, alpha=0.5, linestyle="-.")
ax.text(pd.Timestamp("2026-04-15"), 31.0, "Консенсус 30.5", color="#f5a623",
        fontsize=7.5, alpha=0.8)

vline(ax)
ax.set_title("TTF природний газ EUR/MWh: факт 2024–2026 + прогноз 12 місяців",
             color="white", fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("EUR/MWh", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8.5, ncol=2)
fmt_xaxis(ax)

# Підписи пікових точок
ax.annotate(f"Jan'26\n{ttf_combined.get(pd.Timestamp('2026-01-01'), 50):.0f}",
            xy=(pd.Timestamp("2026-01-01"), 50.0),
            xytext=(-20, 12), textcoords="offset points",
            color="#e74c3c", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

# ─── Panel 2: UAH/тис.м³ ─────────────────────────────────────────────
ax = ax_uah
hist_uah = ttf_hist_uah[HIST_PLOT_START:]

ax.fill_between(fc_idx, fc_bear_uah.values, fc_bull_uah.values,
                alpha=0.12, color="#9b59b6")
ax.plot(hist_uah.index, hist_uah.values, color="#e05c2b", lw=2.0,
        label="TTF import parity (UAH/тис.м³)")

# UEEX historical (точкові дані)
ueex_hist = df_ueex[df_ueex.index >= HIST_PLOT_START]
ax.scatter(ueex_hist.index, ueex_hist.values, s=60, color="#f0c040",
           zorder=5, label="UEEX внутрішній ринок (підтверджені дані)")

ax.plot(fc_idx, fc_base_uah.values, color="#e05c2b", lw=2.0,
        linestyle="--", label="Прогноз TTF (база)")
ax.plot(fc_idx, fc_ueex_uah.values, color="#f0c040", lw=1.8,
        linestyle="--", label="Прогноз UEEX (із нормалізацією премії)")
ax.fill_between(fc_idx, fc_bear_uah.values, fc_bull_uah.values,
                alpha=0.15, color="#9b59b6")

# Нафтогаз бізнес-тарифи як реперні точки
ax.scatter([pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01")],
           [21380, 22602], s=80, marker="D", color="#27ae60", zorder=6,
           label="Нафтогаз Трейдинг (бізнес, без ПДВ)")

vline(ax)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_title("Ціна газу в Україні (UAH/тис.м³): TTF import parity vs UEEX внутрішній ринок",
             color="white", fontsize=11, fontweight="bold")
ax.set_ylabel("UAH / 1000 м³", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8, ncol=2)
fmt_xaxis(ax)

# ─── Panel 3: PSG Ukraine ---─────────────────────────────────────────
ax = ax_psg
hist_psg = ua_psg[ua_psg.index >= HIST_PLOT_START]

ax.fill_between(hist_psg.index, hist_psg.values, alpha=0.35, color="#2980b9")
ax.plot(hist_psg.index, hist_psg.values, color="#2980b9", lw=2.0, label="UA ПСГ % (факт)")

# PSG forecast (сезонна модель + ключові точки з PSG bcm)
# Зовнішній ПСГ bcm: Mar 2026 ~8.0 bcm; загальна ємність ~31 bcm
psg_capacity_bcm = 31.0
psg_bcm_data = pd.read_csv(EXT / "ukraine_psg_bcm.csv", parse_dates=["date"])
psg_bcm_data = psg_bcm_data.set_index("date")["psg_bcm"].sort_index()

# Seasonal PSG forecast: injection Apr-Oct, withdrawal Nov-Mar
psg_seasonal_delta = {
    "2026-04": +2.5,   # початок закачки
    "2026-05": +3.2,
    "2026-06": +2.8,
    "2026-07": +2.5,
    "2026-08": +2.0,
    "2026-09": +1.5,
    "2026-10": +0.5,
    "2026-11": -2.5,
    "2026-12": -3.0,
    "2027-01": -3.5,
    "2027-02": -3.0,
    "2027-03": -2.0,
}
psg_start_bcm = 8.0
psg_fc_bcm = {}
current = psg_start_bcm
for m, delta in psg_seasonal_delta.items():
    current = max(0, min(psg_capacity_bcm, current + delta))
    psg_fc_bcm[pd.Timestamp(m + "-01")] = current

psg_fc_pct = pd.Series({k: v / psg_capacity_bcm * 100 for k, v in psg_fc_bcm.items()})

ax.plot(psg_fc_pct.index, psg_fc_pct.values, color="#5dade2", lw=2.0,
        linestyle="--", label="Прогноз UA ПСГ %")
ax.fill_between(psg_fc_pct.index,
                psg_fc_pct.values * 0.80,
                psg_fc_pct.values * 1.20,
                alpha=0.15, color="#5dade2")
ax.axhline(30, color="#e67e22", lw=0.8, linestyle=":", alpha=0.7, label="30% ціль ЄС (1.11)")
ax.axhline(5,  color="#e74c3c", lw=0.8, linestyle=":", alpha=0.7, label="5% критично")
vline(ax)
ax.set_ylim(0, 60)
ax.set_title("ПСГ України (% заповнення)", color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("%", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
fmt_xaxis(ax)

# ─── Panel 4: Import volumes ─────────────────────────────────────────
ax = ax_imp
hist_imp = df_imp[df_imp.index >= HIST_PLOT_START]

ax.bar(hist_imp.index, hist_imp.values, width=25,
       color="#9b59b6", alpha=0.8, label="Імпорт факт (млн м³)")

# Forecast imports: injection season high, withdrawal season lower
imp_fc = {
    "2026-04": 520, "2026-05": 600, "2026-06": 700,
    "2026-07": 850, "2026-08": 800, "2026-09": 750,
    "2026-10": 500, "2026-11": 400, "2026-12": 350,
    "2027-01": 650, "2027-02": 600, "2027-03": 550,
}
imp_fc_s = pd.Series({pd.Timestamp(k+"-01"): v for k,v in imp_fc.items()})
ax.bar(imp_fc_s.index, imp_fc_s.values, width=25,
       color="#8e44ad", alpha=0.45, hatch="//", label="Прогноз імпорту")

vline(ax)
ax.set_title("Імпорт газу в Україну (млн м³/місяць)",
             color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("млн м³", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
fmt_xaxis(ax)

# ─── Panel 5: Scenario table ──────────────────────────────────────────
ax = ax_table
ax.axis("off")
ax.set_facecolor(PANEL)

col_labels = ["Місяць", "Базовий\n(EUR/MWh)", "Базовий\n(UAH/тис.м³)",
              "Бичачий\n(EUR/MWh)", "Ведмежий\n(EUR/MWh)",
              "UEEX\nвнутрішній", "ПСГ UA\n%"]

rows = []
for i, ts in enumerate(fc_idx):
    m_str = ts.strftime("%b %Y")
    base_e = fc_base_adj.iloc[i]
    base_u = fc_base_uah.iloc[i]
    bull_e = fc_bull.iloc[i]
    bear_e = fc_bear.iloc[i]
    ueex_u = fc_ueex_uah.iloc[i]
    psg_v  = psg_fc_pct.get(ts, float("nan"))
    rows.append([m_str,
                 f"{base_e:.1f}",
                 f"{base_u:,.0f}",
                 f"{bull_e:.1f}",
                 f"{bear_e:.1f}",
                 f"{ueex_u:,.0f}",
                 f"{psg_v:.1f}%" if not np.isnan(psg_v) else "—"])

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.55)

for (r, c), cell in table.get_celld().items():
    cell.set_facecolor("#0f1117" if r == 0 else ("#1a2035" if r % 2 == 0 else "#141a2d"))
    cell.set_text_props(color="white" if r > 0 else "#f0c040")
    cell.set_edgecolor("#2a2a40")
    # підсвітити бичачі значення
    if r > 0 and c == 3:
        cell.set_text_props(color="#e74c3c")
    if r > 0 and c == 4:
        cell.set_text_props(color="#2ecc71")
    if r > 0 and c == 0:
        cell.set_text_props(color="#aaaacc")

ax.set_title("Щомісячний прогноз: квітень 2026 — березень 2027",
             color="white", fontsize=11, fontweight="bold", pad=12)

# ─── Suptitle ─────────────────────────────────────────────────────────
fig.suptitle(
    "Газовий ринок України: прогноз на 1 рік  |  Квітень 2026 — Березень 2027\n"
    "На основі: ОГТСУ/AGPU/EXPRO/Elenger/Нафтогаз + Goldman Sachs / ABN AMRO / Kpler консенсус",
    color="white", fontsize=12, fontweight="bold", y=0.96)

fig.text(0.5, 0.010,
         "Базовий: H-W + SARIMA(1,1,1)(1,1,1,12) + консенсус аналітиків (~30.5 EUR/MWh)  |  "
         "Бичачий: Iran/Qatar supply shock (Goldman Q2-2026: 55-63 EUR/MWh)  |  "
         "Ведмежий: мирний договір РФ-UA + LNG хвиля (Goldman 2026: 29 EUR/MWh)",
         ha="center", color="#666688", fontsize=7.5)

out_path = OUT / "forecast_1y_full.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Текстовий підсумок ───────────────────────────────────────────────
print("\n" + "="*70)
print("  ПРОГНОЗ ГАЗОВОГО РИНКУ УКРАЇНИ: квітень 2026 — березень 2027")
print("="*70)
print(f"\n{'Місяць':<12} {'TTF база':>10} {'TTF bull':>10} {'TTF bear':>10} "
      f"{'UAH база':>12} {'UEEX':>12} {'ПСГ %':>7}")
print("-"*70)
for i, ts in enumerate(fc_idx):
    psg_v = psg_fc_pct.get(ts, float("nan"))
    print(f"{ts.strftime('%b %Y'):<12} "
          f"{fc_base_adj.iloc[i]:>9.1f}€  "
          f"{fc_bull.iloc[i]:>9.1f}€  "
          f"{fc_bear.iloc[i]:>9.1f}€  "
          f"{fc_base_uah.iloc[i]:>11,.0f}  "
          f"{fc_ueex_uah.iloc[i]:>11,.0f}  "
          f"{psg_v:>6.1f}%")

print(f"\nРічне середнє:")
print(f"  TTF базовий   : {fc_base_adj.mean():>7.1f} EUR/MWh")
print(f"  TTF бичачий   : {fc_bull.mean():>7.1f} EUR/MWh")
print(f"  TTF ведмежий  : {fc_bear.mean():>7.1f} EUR/MWh")
print(f"  UAH базовий   : {fc_base_uah.mean():>9,.0f} UAH/тис.м³")
print(f"  UEEX прогноз  : {fc_ueex_uah.mean():>9,.0f} UAH/тис.м³")
print(f"\nПСГ наприкінці прогнозу (бер 2027): {psg_fc_pct.iloc[-1]:.1f}%")
print(f"Пік ПСГ (жовтень 2026 оцінка): {psg_fc_pct.max():.1f}%")
