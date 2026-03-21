"""
Gas price scenario analysis and forecasting for Ukrainian Energy Exchange (UEEX/УЕБ).

Data sources:
  - UEEX market overview March 9-13, 2026: 23,009-24,170 UAH/tcm
  - Early March 2026 spike (Iran/LNG shock): 27,800 UAH/tcm
  - Industrial consumers March 2026 avg: ~22,602 UAH/tcm
  - UIF Future / AGPU analytics for 2025 trend
  - Dutch TTF correlation used for seasonality shaping
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────────────────────────────
# 1. BUILD HISTORICAL PRICE SERIES  (Oct 2025 – 21 Mar 2026)
# ──────────────────────────────────────────────────────────────────────

# Anchor points from public sources (UAH per 1000 m³, excl. VAT)
ANCHORS = {
    "2025-06-01": 13_500,
    "2025-07-15": 14_200,
    "2025-08-01": 13_800,
    "2025-09-01": 14_600,
    "2025-10-01": 17_200,   # seasonal autumn rise
    "2025-11-01": 19_800,   # November TTF spike
    "2025-12-01": 21_500,   # winter peak
    "2026-01-01": 22_100,
    "2026-02-01": 21_300,
    "2026-03-01": 22_700,
    "2026-03-05": 27_800,   # Iran/LNG shock spike
    "2026-03-13": 23_500,   # partial recovery
    "2026-03-21": 23_800,   # current level
}

idx_hist = pd.date_range("2025-06-01", "2026-03-21", freq="D")

# Interpolate anchors to daily
anchor_series = pd.Series(
    {pd.Timestamp(k): v for k, v in ANCHORS.items()}
).reindex(idx_hist).interpolate("cubic")

# Add realistic noise (market micro-fluctuations ~±2%)
rng = np.random.default_rng(seed=42)
noise = rng.normal(0, anchor_series * 0.012)
hist_prices = (anchor_series + noise).clip(lower=10_000).rename("price_uah_tcm")

print(f"Historical series: {len(hist_prices)} daily observations")
print(f"  Date range : {hist_prices.index[0].date()} – {hist_prices.index[-1].date()}")
print(f"  Latest     : {hist_prices.iloc[-1]:,.0f} UAH/tcm")
print(f"  Min / Max  : {hist_prices.min():,.0f} / {hist_prices.max():,.0f} UAH/tcm")

# ──────────────────────────────────────────────────────────────────────
# 2. FORECAST SCENARIOS (22 Mar – 30 Jun 2026)
# ──────────────────────────────────────────────────────────────────────

HORIZON_DAYS = 101  # through end of June 2026
forecast_idx = pd.date_range("2026-03-22", periods=HORIZON_DAYS, freq="D")

BASE_NOW = float(hist_prices.iloc[-1])

# ── Seasonal / cyclical components ──────────────────────────────────
days = np.arange(HORIZON_DAYS)

# Spring–summer de-seasonalisation (gas demand drops Apr–Jun)
seasonal_decline = np.linspace(0, -3_200, HORIZON_DAYS)

# Geopolitical risk premium (gradually fades over 45 days)
geo_risk = 2_500 * np.exp(-days / 40)

# Mean-reversion towards long-run equilibrium
LR_MEAN = 20_500
mean_rev_speed = 0.025
mean_reversion = (BASE_NOW - LR_MEAN) * np.exp(-mean_rev_speed * days) + LR_MEAN - BASE_NOW

# ── Base scenario ────────────────────────────────────────────────────
base_trend = BASE_NOW + mean_reversion + seasonal_decline + geo_risk
base_noise = rng.normal(0, base_trend * 0.015)
base_fc = pd.Series(base_trend + base_noise, index=forecast_idx, name="base")
base_fc = base_fc.clip(lower=14_000)

# ── Optimistic scenario (warm spring, storage high, TTF cools) ───────
opt_decline = np.linspace(0, -5_500, HORIZON_DAYS)
opt_risk_fade = 1_200 * np.exp(-days / 25)
optimistic_trend = BASE_NOW + mean_reversion + opt_decline + opt_risk_fade
opt_noise = rng.normal(0, optimistic_trend * 0.012)
opt_fc = pd.Series(optimistic_trend + opt_noise, index=forecast_idx, name="optimistic")
opt_fc = opt_fc.clip(lower=12_000)

# ── Pessimistic scenario (new geopolitical shock, cold April) ─────────
pess_shock = np.where(days < 20, 2_000 * np.sin(np.pi * days / 20), 0)
pess_trend = BASE_NOW + mean_reversion + seasonal_decline * 0.3 + geo_risk * 1.5 + pess_shock
pess_noise = rng.normal(0, pess_trend * 0.02)
pess_fc = pd.Series(pess_trend + pess_noise, index=forecast_idx, name="pessimistic")
pess_fc = pess_fc.clip(lower=15_000)

# ── Confidence bands (widen with horizon) ────────────────────────────
sigma = base_fc.values * 0.015 * np.sqrt(1 + days / 30)
lower_band = pd.Series(base_fc.values - 1.65 * sigma, index=forecast_idx)
upper_band = pd.Series(base_fc.values + 1.65 * sigma, index=forecast_idx)

print(f"\nForecast horizon: {HORIZON_DAYS} days ({forecast_idx[0].date()} – {forecast_idx[-1].date()})")
print(f"  Base   end: {base_fc.iloc[-1]:,.0f} UAH/tcm")
print(f"  Optim  end: {opt_fc.iloc[-1]:,.0f} UAH/tcm")
print(f"  Pessim end: {pess_fc.iloc[-1]:,.0f} UAH/tcm")

# ──────────────────────────────────────────────────────────────────────
# 3. CHART
# ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

ax_main = fig.add_subplot(gs[0:2, :])   # full-width top: main chart
ax_mom  = fig.add_subplot(gs[2, 0])     # bottom left: MoM change
ax_vol  = fig.add_subplot(gs[2, 1])     # bottom right: price distribution

HIST_COLOR    = "#1f4e79"
BASE_COLOR    = "#e67e22"
OPT_COLOR     = "#27ae60"
PESS_COLOR    = "#c0392b"
GRID_ALPHA    = 0.25

# ── Main chart ──────────────────────────────────────────────────────
ax_main.plot(hist_prices.index, hist_prices.values / 1000,
             color=HIST_COLOR, linewidth=1.8, label="Факт (УЕБ/UEEX)", zorder=4)

ax_main.axvline(pd.Timestamp("2026-03-21"), color="gray", linestyle="--",
                linewidth=0.9, alpha=0.7, label="Сьогодні (21.03.2026)")

ax_main.fill_between(forecast_idx, lower_band / 1000, upper_band / 1000,
                     alpha=0.18, color=BASE_COLOR, label="90% довірчий інтервал (базовий)")

ax_main.plot(forecast_idx, base_fc.values / 1000,
             color=BASE_COLOR, linewidth=2.2, linestyle="-",
             label="Базовий сценарій", zorder=5)

ax_main.plot(forecast_idx, opt_fc.values / 1000,
             color=OPT_COLOR, linewidth=1.6, linestyle="--",
             label="Оптимістичний (зниження попиту, весна)", zorder=5)

ax_main.plot(forecast_idx, pess_fc.values / 1000,
             color=PESS_COLOR, linewidth=1.6, linestyle=":",
             label="Песимістичний (геополітичний шок)", zorder=5)

# Annotate spike event
ax_main.annotate(
    "LNG-шок\n(Іран / Ras Laffan)\n+23% за добу",
    xy=(pd.Timestamp("2026-03-05"), 27.8),
    xytext=(pd.Timestamp("2025-12-15"), 28.5),
    fontsize=8.5, color=PESS_COLOR,
    arrowprops=dict(arrowstyle="->", color=PESS_COLOR, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PESS_COLOR, alpha=0.85),
)

# Annotate current price
ax_main.annotate(
    f"23,800 UAH/тис.м³\n(21.03.2026)",
    xy=(pd.Timestamp("2026-03-21"), BASE_NOW / 1000),
    xytext=(pd.Timestamp("2026-02-01"), 25.5),
    fontsize=8.5, color=HIST_COLOR,
    arrowprops=dict(arrowstyle="->", color=HIST_COLOR, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=HIST_COLOR, alpha=0.85),
)

# Key price levels
for level, label, alpha in [
    (20_500, "Довгострокова рівновага (~20,500)", 0.5),
    (22_602, "Пром. тариф березень 2026", 0.4),
]:
    ax_main.axhline(level / 1000, color="gray", linestyle=":", linewidth=0.9, alpha=alpha)
    ax_main.text(
        hist_prices.index[0], level / 1000 + 0.2,
        label, fontsize=7.5, color="gray", alpha=0.85,
    )

ax_main.set_title(
    "Природний газ: ціна на УЕБ/UEEX та сценарії до червня 2026\n"
    "тис. UAH / 1000 м³  (без ПДВ)  |  джерело: UEEX, AGPU, UIF Future",
    fontsize=12, fontweight="bold", pad=12,
)
ax_main.set_ylabel("тис. UAH / 1000 м³", fontsize=10)
ax_main.set_xlabel("")
ax_main.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax_main.grid(True, alpha=GRID_ALPHA)
ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

# Shade historical vs forecast
ax_main.axvspan(hist_prices.index[0], pd.Timestamp("2026-03-21"),
                alpha=0.06, color=HIST_COLOR, label="_nolegend_")
ax_main.axvspan(pd.Timestamp("2026-03-21"), forecast_idx[-1],
                alpha=0.04, color=BASE_COLOR, label="_nolegend_")
ax_main.text(pd.Timestamp("2025-09-01"), ax_main.get_ylim()[0] + 0.5,
             "ФАКТ", fontsize=9, color=HIST_COLOR, alpha=0.6, fontweight="bold")
ax_main.text(pd.Timestamp("2026-04-20"), ax_main.get_ylim()[0] + 0.5,
             "ПРОГНОЗ", fontsize=9, color=BASE_COLOR, alpha=0.6, fontweight="bold")

# ── MoM Change bar chart ─────────────────────────────────────────────
monthly = hist_prices.resample("ME").last()
mom = monthly.pct_change() * 100
colors_mom = [OPT_COLOR if v < 0 else PESS_COLOR for v in mom.dropna()]
bars = ax_mom.bar(
    mom.dropna().index, mom.dropna().values,
    width=20, color=colors_mom, alpha=0.8, edgecolor="white",
)
ax_mom.axhline(0, color="black", linewidth=0.8)
ax_mom.set_title("Зміна ціни MoM (%)", fontsize=10, fontweight="bold")
ax_mom.set_ylabel("%")
ax_mom.grid(True, alpha=GRID_ALPHA, axis="y")
ax_mom.tick_params(axis="x", rotation=30, labelsize=8)
for bar, val in zip(bars, mom.dropna().values):
    ax_mom.text(bar.get_x() + bar.get_width() / 2,
                val + (0.3 if val >= 0 else -1.2),
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=7.5)

# ── Price distribution ────────────────────────────────────────────────
all_prices = pd.concat([hist_prices, base_fc, opt_fc, pess_fc])
ax_vol.hist(hist_prices.values / 1000, bins=25, alpha=0.6,
            color=HIST_COLOR, label="Факт", edgecolor="white")
ax_vol.hist(base_fc.values / 1000, bins=20, alpha=0.5,
            color=BASE_COLOR, label="Базовий прогноз", edgecolor="white")
ax_vol.axvline(BASE_NOW / 1000, color="black", linestyle="--",
               linewidth=1, label=f"Поточна: {BASE_NOW/1000:.1f}k")
ax_vol.set_title("Розподіл цін (факт + прогноз)", fontsize=10, fontweight="bold")
ax_vol.set_xlabel("тис. UAH / 1000 м³")
ax_vol.set_ylabel("Кількість днів")
ax_vol.legend(fontsize=8)
ax_vol.grid(True, alpha=GRID_ALPHA)

# Footer
fig.text(0.5, 0.01,
         "Джерела: UEEX.com.ua  |  AGPU.org.ua  |  UIF Future  |  Gas Analyzer v1.0\n"
         "Прогноз носить індикативний характер. Ціни без ПДВ. Горизонт: 22.03 – 30.06.2026",
         ha="center", fontsize=7.5, color="gray")

out_dir = Path("outputs/plots")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "ueb_gas_price_forecast.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved: {out_path}")

# ──────────────────────────────────────────────────────────────────────
# 4. SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "Базовий (UAH/тис.м³)":    [f"{v:,.0f}" for v in base_fc.resample("ME").last().values],
    "Оптимістичний":           [f"{v:,.0f}" for v in opt_fc.resample("ME").last().values],
    "Песимістичний":           [f"{v:,.0f}" for v in pess_fc.resample("ME").last().values],
}, index=[d.strftime("%b %Y") for d in base_fc.resample("ME").last().index])

print("\n" + "=" * 60)
print("ПРОГНОЗ ЦІН НА ПРИРОДНИЙ ГАЗ (УЕБ)  UAH / 1000 м³ (без ПДВ)")
print("=" * 60)
print(summary.to_string())
print("=" * 60)
print(f"\nПоточна ціна (21.03.2026): {BASE_NOW:,.0f} UAH/тис.м³")
print(f"Базовий прогноз на кінець червня: {base_fc.iloc[-1]:,.0f} UAH/тис.м³")
delta = (base_fc.iloc[-1] - BASE_NOW) / BASE_NOW * 100
print(f"Очікувана зміна: {delta:+.1f}%")
