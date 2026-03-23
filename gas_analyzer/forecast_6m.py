"""
Прогноз газового ринку України: 6 місяців
на основі повного датасету (до жовтня 2025)
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

RAW = Path("data/raw")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

BG    = "#0f1117"
PANEL = "#1a1d2e"
GRID  = "#2a2a40"
AXIS_C = "#cccccc"

# ── Load all data ──────────────────────────────────────────────────────
df_price = pd.read_csv(RAW / "2025_10_14_pryvedena_tsina_hazu_do_kordonu_ukrainy.csv",
                       parse_dates=["date"])
df_psg   = pd.read_csv(RAW / "2025_10_14_zapasy_hazu_yevropeiski_psg.csv",
                       parse_dates=["date"])
df_flow  = pd.read_csv(RAW / "2025_10_14_fizychni_potoky_v_ukrainskii_hts.csv",
                       parse_dates=["date"])

# ── TTF price series (daily, fill gaps) ───────────────────────────────
ttf_all = (df_price[df_price["hub_name"]=="TTF"]
           .set_index("date")["price_without_vat"]
           .sort_index()
           .resample("D").mean()
           .interpolate("linear"))

# train / history split
last_date = ttf_all.index[-1]
forecast_start = last_date + pd.Timedelta(days=1)
forecast_end   = last_date + pd.Timedelta(days=183)  # 6 months
future_idx = pd.date_range(forecast_start, forecast_end, freq="D")

# ── TTF: Holt-Winters (additive trend, multiplicative seasonality) ────
hw_model = ExponentialSmoothing(
    ttf_all,
    trend="add",
    seasonal="add",
    seasonal_periods=365,
    damped_trend=True
).fit(optimized=True)

ttf_fc = hw_model.forecast(len(future_idx))
ttf_ci_lo = ttf_fc * 0.88
ttf_ci_hi = ttf_fc * 1.12

# ── TTF: ARIMA backup (monthly resampled) ────────────────────────────
ttf_m = ttf_all.resample("ME").mean()
arima = ARIMA(ttf_m, order=(2,1,2)).fit()
arima_fc = arima.forecast(6)
arima_idx = pd.date_range(
    ttf_m.index[-1] + pd.DateOffset(months=1), periods=6, freq="ME"
)

# ── UA PSG series ─────────────────────────────────────────────────────
ua_psg = (df_psg[df_psg["country"]=="Україна"]
          .set_index("date")["amount_storage_perc"]
          .sort_index()
          .resample("D").mean()
          .interpolate("linear"))

# Seasonal model: use previous year's seasonal pattern as forecast skeleton
# The PSG cycle is highly seasonal (injection Mar-Oct, withdrawal Oct-Mar)
psg_2024 = ua_psg["2024-11-01":"2025-04-30"]  # similar period
# Build future PSG forecast by fitting seasonal shape with current level offset
psg_seasonal = ua_psg.groupby(ua_psg.index.dayofyear).mean()

def psg_forecast_from(start_date: pd.Timestamp, n_days: int,
                      start_val: float) -> pd.Series:
    idx = pd.date_range(start_date, periods=n_days, freq="D")
    seasonal_vals = np.array([psg_seasonal.get(d.dayofyear,
                              psg_seasonal.mean()) for d in idx])
    # scale seasonal pattern to start from current level
    scale = start_val / (psg_seasonal.loc[start_date.dayofyear]
                         if start_date.dayofyear in psg_seasonal.index
                         else psg_seasonal.mean())
    # linear blend: 60% seasonal, 40% scaled
    forecast = seasonal_vals * 0.6 + seasonal_vals * scale * 0.4
    return pd.Series(forecast, index=idx)

psg_fc = psg_forecast_from(forecast_start, len(future_idx),
                            ua_psg.iloc[-1])
psg_ci_lo = psg_fc * 0.82
psg_ci_hi = psg_fc * 1.18

# ── EU PSG ────────────────────────────────────────────────────────────
eu_all_raw = df_psg.dropna(subset=["amount_storage_perc","technical_capacity"])
eu_total = (eu_all_raw.groupby("date")
            .apply(lambda g: np.average(g["amount_storage_perc"],
                                        weights=g["technical_capacity"]))
            .rename("eu_avg")
            .resample("D").mean()
            .interpolate("linear"))

# EU PSG forecast (same seasonal approach)
eu_seasonal = eu_total.groupby(eu_total.index.dayofyear).mean()
eu_fc_vals  = np.array([eu_seasonal.get(d.dayofyear,
              eu_seasonal.mean()) for d in future_idx])
eu_psg_fc = pd.Series(eu_fc_vals, index=future_idx)

# ── Net cross-border flows ─────────────────────────────────────────────
cb = df_flow[df_flow["interconnection_point_type"]=="Транскордонні точки"]
cb_out = cb[cb["direction"]=="Вихід"].groupby("date")["transportation_amount"].sum() / 1e6
cb_in  = cb[cb["direction"]=="Вхід"].groupby("date")["transportation_amount"].sum() / 1e6
net = (cb_out - cb_in).resample("W").mean()

# Flow forecast: linear trend + seasonal
X = np.arange(len(net)).reshape(-1,1)
lr = LinearRegression().fit(X, net.values)
n_future_weeks = 26
X_fc = np.arange(len(net), len(net)+n_future_weeks).reshape(-1,1)
flow_fc_trend = lr.predict(X_fc)
# add seasonal oscillation (observed average annual pattern)
week_of_year = [w % 52 for w in range(len(net), len(net)+n_future_weeks)]
# simple sine: winter high import, summer low
flow_seasonal = -8 * np.sin(np.array(week_of_year) / 52 * 2 * np.pi + np.pi)
flow_fc = flow_fc_trend + flow_seasonal * 0.5
flow_fc_idx = pd.date_range(
    net.index[-1] + pd.Timedelta(weeks=1), periods=n_future_weeks, freq="W"
)
flow_fc_series = pd.Series(flow_fc, index=flow_fc_idx)

# ── Layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 20), facecolor=BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28,
                        left=0.07, right=0.97, top=0.92, bottom=0.05)

ax_price = fig.add_subplot(gs[0, :])
ax_ua    = fig.add_subplot(gs[1, 0])
ax_eu    = fig.add_subplot(gs[1, 1])
ax_flow  = fig.add_subplot(gs[2, :])

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=AXIS_C, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

def styled_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

HIST_START = "2024-01-01"

# ────────────────────────────────────────────────────────────────────────
# Panel 1: TTF Price + 6M forecast
# ────────────────────────────────────────────────────────────────────────
ax = ax_price
hist = ttf_all[HIST_START:]

ax.plot(hist.index, hist.values, color="#e05c2b", lw=1.8, label="TTF факт")

# Holt-Winters forecast
ax.plot(future_idx, ttf_fc.values, color="#ff8c55", lw=2.0,
        linestyle="--", label="Прогноз H-W (6м)")
ax.fill_between(future_idx, ttf_ci_lo.values, ttf_ci_hi.values,
                alpha=0.2, color="#ff8c55", label="Довірчий інтервал ±12%")

# ARIMA monthly overlay
ax.plot(arima_idx, arima_fc.values, color="#f0e050", lw=1.4,
        linestyle=":", marker="o", markersize=4, label="ARIMA (місячний, резерв)")

# vertical divider
ax.axvline(last_date, color="#7f8c8d", lw=1.2, linestyle=":", alpha=0.9)
ax.text(last_date + pd.Timedelta(days=3), ax.get_ylim()[0]*1.01,
        "← факт  /  прогноз →", color="#aaaaaa", fontsize=8)

ax.set_title("Ціна газу TTF (грн / 1000 м³): факт 2024–2025 + прогноз на 6 місяців",
             color="white", fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("грн / 1000 м³", color=AXIS_C, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8, ncol=2)
styled_xaxis(ax)

# Annotate forecast stats
fc_mean = ttf_fc.mean()
fc_min  = ttf_fc.min()
fc_max  = ttf_fc.max()
ax.text(0.72, 0.92,
        f"Прогноз:\n  сер. {fc_mean:,.0f}\n  мін. {fc_min:,.0f}\n  макс. {fc_max:,.0f}",
        transform=ax.transAxes, color="#ff8c55", fontsize=9,
        va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d2e",
                            edgecolor="#444466", alpha=0.8))

# ────────────────────────────────────────────────────────────────────────
# Panel 2: UA PSG
# ────────────────────────────────────────────────────────────────────────
ax = ax_ua
ua_hist = ua_psg[HIST_START:]

ax.fill_between(ua_hist.index, ua_hist.values, alpha=0.35, color="#2980b9")
ax.plot(ua_hist.index, ua_hist.values, color="#2980b9", lw=1.8, label="UA ПСГ факт")

ax.fill_between(future_idx, psg_ci_lo.values, psg_ci_hi.values,
                alpha=0.18, color="#5dade2")
ax.plot(future_idx, psg_fc.values, color="#5dade2", lw=2.0,
        linestyle="--", label="Прогноз UA ПСГ")

ax.axvline(last_date, color="#7f8c8d", lw=1.0, linestyle=":", alpha=0.8)
ax.axhline(30, color="#e67e22", lw=0.8, linestyle=":", alpha=0.7, label="30% ціль ЄС (1 листопада)")
ax.axhline(5, color="#e74c3c", lw=0.8, linestyle=":", alpha=0.7, label="5% критичний мінімум")

ax.set_ylim(0, 50)
ax.set_title("Заповненість ПСГ України (%) + прогноз",
             color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("%", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
styled_xaxis(ax)

# Forecast end value
psg_end = psg_fc.iloc[-1]
ax.text(0.65, 0.15,
        f"Очікуваний рівень\nчерез 6м: {psg_end:.1f}%",
        transform=ax.transAxes, color="#5dade2", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d2e",
                  edgecolor="#444466", alpha=0.8))

# ────────────────────────────────────────────────────────────────────────
# Panel 3: EU PSG
# ────────────────────────────────────────────────────────────────────────
ax = ax_eu
eu_hist = eu_total[HIST_START:]

ax.fill_between(eu_hist.index, eu_hist.values, alpha=0.25, color="#27ae60")
ax.plot(eu_hist.index, eu_hist.values, color="#27ae60", lw=1.8, label="ЄС ПСГ факт (зважене)")
ax.plot(future_idx, eu_psg_fc.values, color="#58d68d", lw=2.0,
        linestyle="--", label="Прогноз ЄС ПСГ")
ax.fill_between(future_idx, eu_psg_fc.values * 0.93, eu_psg_fc.values * 1.07,
                alpha=0.15, color="#58d68d")

ax.axvline(last_date, color="#7f8c8d", lw=1.0, linestyle=":", alpha=0.8)
ax.axhline(90, color="#e74c3c", lw=0.8, linestyle=":", alpha=0.7, label="90% ціль ЄС (1 листопада)")

ax.set_ylim(0, 100)
ax.set_title("Заповненість ПСГ ЄС (зважене, %) + прогноз",
             color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("%", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
styled_xaxis(ax)

# ────────────────────────────────────────────────────────────────────────
# Panel 4: Cross-border flows forecast
# ────────────────────────────────────────────────────────────────────────
ax = ax_flow
hist_flow = net["2024-01-01":]

pos_h = hist_flow.clip(lower=0)
neg_h = hist_flow.clip(upper=0)
ax.fill_between(hist_flow.index, pos_h, alpha=0.5, color="#27ae60")
ax.fill_between(hist_flow.index, neg_h, alpha=0.5, color="#e74c3c")
ax.plot(hist_flow.index, hist_flow.values, color="#ecf0f1", lw=0.8, alpha=0.6, label="Факт")

pos_f = flow_fc_series.clip(lower=0)
neg_f = flow_fc_series.clip(upper=0)
ax.fill_between(flow_fc_series.index, pos_f, alpha=0.35, color="#58d68d",
                label="Прогноз (вихід)")
ax.fill_between(flow_fc_series.index, neg_f, alpha=0.35, color="#f1948a",
                label="Прогноз (вхід)")
ax.plot(flow_fc_series.index, flow_fc_series.values,
        color="#ecf0f1", lw=1.5, linestyle="--", alpha=0.7)

ax.axhline(0, color="#555577", lw=0.8)
ax.axvline(last_date, color="#7f8c8d", lw=1.2, linestyle=":", alpha=0.8)

ax.set_title("Нетто транскордонні потоки ГТС (млн м³/добу, тижнева середня) + прогноз",
             color="white", fontsize=12, fontweight="bold")
ax.set_ylabel("млн м³/добу", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8, ncol=3)
styled_xaxis(ax)

# ── Suptitle & footnote ───────────────────────────────────────────────
fig.suptitle(
    "Газовий ринок України: прогноз на 6 місяців  |  "
    f"Дані до {last_date.strftime('%d.%m.%Y')}  |  Горизонт: {forecast_end.strftime('%B %Y')}",
    color="white", fontsize=13, fontweight="bold", y=0.96)

fig.text(0.5, 0.015,
         "Методи: Holt-Winters (ціни), ARIMA(2,1,2) (місячний cross-check), "
         "сезонна інтерполяція (ПСГ), лінійний тренд + сезонна хвиля (потоки).  "
         "Прогноз має індикативний характер. Довірчі інтервали ±12% (ціни), ±18% (ПСГ).",
         ha="center", color="#666688", fontsize=7.5)

out_path = OUT / "forecast_6m.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out_path}")

# ── Text summary ──────────────────────────────────────────────────────
print("\n" + "="*58)
print("  ПРОГНОЗ: НАСТУПНІ 6 МІСЯЦІВ")
print(f"  {forecast_start.strftime('%d.%m.%Y')} — {forecast_end.strftime('%d.%m.%Y')}")
print("="*58)

# Monthly breakdown
ttf_fc.index = future_idx
monthly = ttf_fc.resample("ME").mean()
print(f"\n{'Місяць':<14} {'TTF прогноз':>16}  {'UA ПСГ':>10}  {'ЄС ПСГ':>10}")
print("-"*54)

psg_fc_m  = psg_fc.resample("ME").mean()
eu_psg_m  = eu_psg_fc.resample("ME").mean()

for month in monthly.index:
    price_m = monthly.get(month, float("nan"))
    ua_m    = psg_fc_m.get(month, float("nan"))
    eu_m    = eu_psg_m.get(month, float("nan"))
    label   = month.strftime("%b %Y")
    print(f"{label:<14} {price_m:>13,.0f} грн  {ua_m:>8.1f}%  {eu_m:>8.1f}%")

print(f"\nРезюме прогнозу:")
print(f"  Середня ціна TTF   : {fc_mean:>10,.0f} грн/1000 м³")
print(f"  Мін. ціна          : {fc_min:>10,.0f} грн/1000 м³")
print(f"  Макс. ціна         : {fc_max:>10,.0f} грн/1000 м³")
print(f"  UA ПСГ через 6м    : {psg_fc.iloc[-1]:>9.1f}%")
print(f"  ЄС ПСГ через 6м    : {eu_psg_fc.iloc[-1]:>9.1f}%")
print(f"  Потоки: тренд      : {'негативний (імпорт)' if flow_fc_series.mean() < 0 else 'позитивний'}")
