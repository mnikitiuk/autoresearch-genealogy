"""
Exploratory visualization of uploaded market data sources.
Generates a combined 4-panel overview: prices, Ukraine PSG, cross-border flows, EU storage.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

RAW = Path("data/raw")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────
df_price = pd.read_csv(RAW / "2025_10_14_pryvedena_tsina_hazu_do_kordonu_ukrainy.csv",
                       parse_dates=["date"])
df_flow  = pd.read_csv(RAW / "2025_10_14_fizychni_potoky_v_ukrainskii_hts.csv",
                       parse_dates=["date"])
df_psg   = pd.read_csv(RAW / "2025_10_14_zapasy_hazu_yevropeiski_psg.csv",
                       parse_dates=["date"])

# ── Derived series ────────────────────────────────────────────────────

# 1. TTF spot price (no VAT)
ttf = (df_price[df_price["hub_name"] == "TTF"]
       .set_index("date")["price_without_vat"]
       .resample("W").mean()
       .dropna())

# 2. CEGH (closest hub to Ukraine)
cegh = (df_price[df_price["hub_name"] == "CEGH"]
        .set_index("date")["price_without_vat"]
        .resample("W").mean()
        .dropna())

# 3. Ukraine PSG storage %
ua_psg = (df_psg[df_psg["country"] == "Україна"]
          .set_index("date")["amount_storage_perc"]
          .resample("D").mean()
          .dropna())

# 4. EU total storage % (weighted mean by technical capacity)
eu_psg = (df_psg.dropna(subset=["amount_storage_perc", "technical_capacity"])
          .groupby("date")
          .apply(lambda g: np.average(g["amount_storage_perc"],
                                      weights=g["technical_capacity"]))
          .rename("eu_avg")
          .resample("D").mean()
          .dropna())

# 5. Ukraine cross-border: net export (Вихід - Вхід) for transit points
cb = df_flow[df_flow["interconnection_point_type"] == "Транскордонні точки"].copy()
cb_out = (cb[cb["direction"] == "Вихід"]
          .groupby("date")["transportation_amount"].sum())
cb_in  = (cb[cb["direction"] == "Вхід"]
          .groupby("date")["transportation_amount"].sum())
net_flow = ((cb_out - cb_in) / 1e6).resample("W").mean().dropna()   # млн м³/день

# 6. Ukraine domestic: gas production + UGS withdrawal
ugs_wd = (df_flow[(df_flow["interconnection_point_type"] == "Підземні сховища газу") &
                  (df_flow["direction"] == "Вихід")]
          .groupby("date")["transportation_amount"].sum()
          .div(1e6).resample("W").mean().dropna())

prod = (df_flow[(df_flow["interconnection_point_type"] == "Мережі газовидобувних підприємств") &
                (df_flow["direction"] == "Вихід")]
        .groupby("date")["transportation_amount"].sum()
        .div(1e6).resample("W").mean().dropna())

# ── Plot ──────────────────────────────────────────────────────────────
COLORS = {
    "ttf":    "#e05c2b",
    "cegh":   "#f5a623",
    "ua_psg": "#2980b9",
    "eu_psg": "#95a5a6",
    "net":    "#27ae60",
    "prod":   "#8e44ad",
    "ugs":    "#16a085",
}

fig, axes = plt.subplots(4, 1, figsize=(14, 20), constrained_layout=True)
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#cccccc", labelsize=9)
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

def set_year_grid(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(axis="x", which="major", color="#333355", linewidth=0.8)
    ax.grid(axis="x", which="minor", color="#2a2a40", linewidth=0.4, linestyle=":")
    ax.grid(axis="y", color="#2a2a40", linewidth=0.4)

# ── Panel 1: Gas prices ───────────────────────────────────────────────
ax = axes[0]
ax.plot(ttf.index, ttf.values, color=COLORS["ttf"], lw=1.5, label="TTF (без ПДВ)")
ax.plot(cegh.index, cegh.values, color=COLORS["cegh"], lw=1.5, label="CEGH (без ПДВ)")

# Shade crisis period
ax.axvspan(pd.Timestamp("2021-10-01"), pd.Timestamp("2022-08-01"),
           alpha=0.12, color="#e74c3c", label="Енергетична криза")
ax.axvspan(pd.Timestamp("2022-02-24"), pd.Timestamp("2022-03-10"),
           alpha=0.25, color="#e74c3c")

ax.set_title("Ціна газу на хабах до кордону України (грн / 1000 м³)",
             color="white", fontsize=11, pad=8)
ax.set_ylabel("грн / 1000 м³", color="#cccccc")
ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
set_year_grid(ax)

# ── Panel 2: Ukraine + EU PSG ─────────────────────────────────────────
ax = axes[1]
ax.fill_between(ua_psg.index, ua_psg.values, alpha=0.35,
                color=COLORS["ua_psg"])
ax.plot(ua_psg.index, ua_psg.values, color=COLORS["ua_psg"], lw=1.8,
        label="Україна ПСГ (% заповнення)")
ax.plot(eu_psg.index, eu_psg.values, color=COLORS["eu_psg"], lw=1.2,
        linestyle="--", label="ЄС середнє (зважене)")

ax.axhline(30, color="#e74c3c", lw=0.8, linestyle=":", alpha=0.7,
           label="Критичний рівень 30%")
ax.set_ylim(0, 105)
ax.set_title("Заповненість підземних сховищ газу",
             color="white", fontsize=11, pad=8)
ax.set_ylabel("%", color="#cccccc")
ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
set_year_grid(ax)

# ── Panel 3: Net cross-border flow ────────────────────────────────────
ax = axes[2]
pos = net_flow.clip(lower=0)
neg = net_flow.clip(upper=0)
ax.fill_between(net_flow.index, pos.values, alpha=0.5,
                color=COLORS["net"], label="Нетто-вихід (транзит)")
ax.fill_between(net_flow.index, neg.values, alpha=0.5,
                color="#e74c3c", label="Нетто-вхід (імпорт)")
ax.plot(net_flow.index, net_flow.values, color=COLORS["net"], lw=0.8, alpha=0.6)
ax.axhline(0, color="#555577", lw=0.8)
ax.set_title("Транскордонні потоки ГТС (нетто, тижнева середня)",
             color="white", fontsize=11, pad=8)
ax.set_ylabel("млн м³/добу", color="#cccccc")
ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
set_year_grid(ax)

# ── Panel 4: Production + UGS withdrawal ─────────────────────────────
ax = axes[3]
ax.stackplot(prod.index, prod.values,
             colors=[COLORS["prod"]], alpha=0.6,
             labels=["Видобуток (вихід з надр)"])
ax.plot(ugs_wd.index, ugs_wd.values, color=COLORS["ugs"], lw=1.5,
        label="Відбір з ПСГ")
ax.set_title("Внутрішня пропозиція: видобуток та відбір з ПСГ",
             color="white", fontsize=11, pad=8)
ax.set_ylabel("млн м³/добу", color="#cccccc")
ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
set_year_grid(ax)

# ── Footer ────────────────────────────────────────────────────────────
fig.suptitle("Газовий ринок України: огляд ключових показників  |  2017–2025",
             color="white", fontsize=13, fontweight="bold", y=1.01)

fig.text(0.5, -0.005,
         "Джерела: ОГТСУ (фізичні потоки), UIF/UEEX/AGPU (ціни), GIE AGSI (ПСГ ЄС+UA)",
         ha="center", color="#888899", fontsize=8)

out_path = OUT / "gas_market_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
