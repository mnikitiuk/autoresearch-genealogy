"""
Аналітика газового ринку: весна 2025 (березень–травень)
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
from matplotlib.patches import FancyArrowPatch

RAW = Path("data/raw")
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

SPRING = ("2025-03-01", "2025-05-31")
BG     = "#0f1117"
PANEL  = "#1a1d2e"
GRID   = "#2a2a40"
AXIS_C = "#cccccc"

# ── Load ──────────────────────────────────────────────────────────────
df_price = pd.read_csv(RAW / "2025_10_14_pryvedena_tsina_hazu_do_kordonu_ukrainy.csv",
                       parse_dates=["date"])
df_flow  = pd.read_csv(RAW / "2025_10_14_fizychni_potoky_v_ukrainskii_hts.csv",
                       parse_dates=["date"])
df_psg   = pd.read_csv(RAW / "2025_10_14_zapasy_hazu_yevropeiski_psg.csv",
                       parse_dates=["date"])

def spring(df, col="date"):
    return df[(df[col] >= SPRING[0]) & (df[col] <= SPRING[1])].copy()

# ── Derived series ────────────────────────────────────────────────────
# Prices
sp = spring(df_price)
ttf  = sp[sp["hub_name"]=="TTF"].set_index("date")["price_without_vat"].sort_index()
cegh = sp[sp["hub_name"]=="CEGH"].set_index("date")["price_without_vat"].sort_index()
the  = sp[sp["hub_name"]=="THE"].set_index("date")["price_without_vat"].sort_index()

# YoY comparison: spring 2024
sp24 = df_price[(df_price["date"] >= "2024-03-01") & (df_price["date"] <= "2024-05-31")]
ttf24 = sp24[sp24["hub_name"]=="TTF"].set_index("date")["price_without_vat"].sort_index()

# PSG Ukraine
ua_psg = spring(df_psg[df_psg["country"]=="Україна"])
ua_psg = ua_psg.set_index("date").sort_index()

# PSG EU selected countries for comparison
key_countries = ["Германія", "Австрія", "Польща", "Угорщина", "Франція"]
eu_sel = spring(df_psg[df_psg["country"].isin(key_countries)])
eu_psg_fill = eu_sel.groupby(["date","country"])["amount_storage_perc"].mean().unstack("country")

# EU total (weighted)
eu_all = spring(df_psg).dropna(subset=["amount_storage_perc","technical_capacity"])
eu_total = (eu_all.groupby("date")
            .apply(lambda g: np.average(g["amount_storage_perc"],
                                        weights=g["technical_capacity"]))
            .rename("eu_avg"))

# Cross-border flows
cb = spring(df_flow[df_flow["interconnection_point_type"]=="Транскордонні точки"])
cb_out = cb[cb["direction"]=="Вихід"].groupby("date")["transportation_amount"].sum() / 1e6
cb_in  = cb[cb["direction"]=="Вхід"].groupby("date")["transportation_amount"].sum() / 1e6
net_flow = (cb_out - cb_in).resample("W").mean()

# Top border points by volume
top_points = (cb.groupby(["interconnection_point","direction"])["transportation_amount"]
              .mean().div(1e6).unstack("direction").fillna(0)
              .assign(net=lambda x: x.get("Вихід",0) - x.get("Вхід",0))
              .sort_values("net", ascending=False).head(8))

# ── KPI calculations ──────────────────────────────────────────────────
ttf_mean   = ttf.mean()
ttf_min    = ttf.min();  ttf_min_d = ttf.idxmin().strftime("%d.%m")
ttf_max    = ttf.max();  ttf_max_d = ttf.idxmax().strftime("%d.%m")
ttf24_mean = ttf24.mean()
yoy_pct    = (ttf_mean - ttf24_mean) / ttf24_mean * 100

ua_start = ua_psg["amount_storage_perc"].iloc[0]
ua_end   = ua_psg["amount_storage_perc"].iloc[-1]
ua_min   = ua_psg["amount_storage_perc"].min()
ua_min_d = ua_psg["amount_storage_perc"].idxmin().strftime("%d.%m")

net_total = net_flow.sum()  # млн м³ net за весну (тижнева середня * 7 * тижнів)

# ── Layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30,
                        left=0.07, right=0.97, top=0.93, bottom=0.04)

ax_price  = fig.add_subplot(gs[0, :])     # full width — prices
ax_psg_ua = fig.add_subplot(gs[1, 0])     # left  — UA PSG
ax_psg_eu = fig.add_subplot(gs[1, 1])     # right — EU PSG
ax_flow   = fig.add_subplot(gs[2, :])     # full width — flows
ax_pts    = fig.add_subplot(gs[3, 0])     # left  — top border points
ax_kpi    = fig.add_subplot(gs[3, 1])     # right — KPI summary

for ax in fig.axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=AXIS_C, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

def month_fmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    ax.grid(axis="x", which="minor", color=GRID, linewidth=0.3, linestyle=":")

# ── Panel 1: Prices ───────────────────────────────────────────────────
ax = ax_price
ax.plot(ttf.index,  ttf.values,  color="#e05c2b", lw=2.0, label="TTF 2025")
ax.plot(cegh.index, cegh.values, color="#f5a623", lw=1.5, label="CEGH 2025", linestyle="--")
ax.plot(the.index,  the.values,  color="#f0e050", lw=1.2, label="THE 2025",  linestyle=":")

# YoY comparison TTF 2024 (shifted +365 days for visual overlay)
ttf24_shifted = ttf24.copy()
ttf24_shifted.index = ttf24_shifted.index + pd.DateOffset(years=1)
ax.plot(ttf24_shifted.index, ttf24_shifted.values,
        color="#e05c2b", lw=1.2, alpha=0.35, linestyle="-.",
        label=f"TTF 2024 (+365д, порівняння)")

ax.fill_between(ttf.index, ttf.values, ttf24_shifted.reindex(ttf.index), alpha=0.08,
                where=(ttf.values > ttf24_shifted.reindex(ttf.index).values),
                color="#e74c3c", label="2025 > 2024")
ax.fill_between(ttf.index, ttf.values, ttf24_shifted.reindex(ttf.index), alpha=0.08,
                where=(ttf.values < ttf24_shifted.reindex(ttf.index).values),
                color="#2ecc71", label="2025 < 2024")

ax.set_title("Ціна газу на хабах (грн / 1000 м³) — весна 2025",
             color="white", fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("грн / 1000 м³", color=AXIS_C, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8, ncol=3)
month_fmt(ax)

# Annotate min/max
ax.annotate(f"мін {ttf_min:,.0f}\n{ttf_min_d}",
            xy=(ttf.idxmin(), ttf_min), xytext=(10, -30),
            textcoords="offset points", color="#2ecc71", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=0.8))
ax.annotate(f"макс {ttf_max:,.0f}\n{ttf_max_d}",
            xy=(ttf.idxmax(), ttf_max), xytext=(10, 10),
            textcoords="offset points", color="#e74c3c", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

# ── Panel 2: UA PSG ───────────────────────────────────────────────────
ax = ax_psg_ua
fill_col = "#2980b9"
ax.fill_between(ua_psg.index, ua_psg["amount_storage_perc"], alpha=0.4, color=fill_col)
ax.plot(ua_psg.index, ua_psg["amount_storage_perc"],
        color=fill_col, lw=2.0, label="Україна ПСГ %")
ax.plot(eu_total.index, eu_total.values,
        color="#95a5a6", lw=1.2, linestyle="--", label="ЄС середнє")
ax.axhline(5, color="#e74c3c", lw=0.8, linestyle=":", alpha=0.8, label="5% — критично")

ax.set_ylim(0, 25)
ax.set_title("Заповненість ПСГ України vs ЄС", color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("%", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
month_fmt(ax)

# Annotate min
ax.annotate(f"мін {ua_min:.1f}%\n{ua_min_d}",
            xy=(ua_psg["amount_storage_perc"].idxmin(), ua_min),
            xytext=(5, 15), textcoords="offset points",
            color="#e74c3c", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8))

# ── Panel 3: EU PSG countries ─────────────────────────────────────────
ax = ax_psg_eu
colors_eu = ["#3498db","#e67e22","#2ecc71","#9b59b6","#1abc9c"]
for (country, series), col in zip(eu_psg_fill.items(), colors_eu):
    s = series.dropna()
    ax.plot(s.index, s.values, color=col, lw=1.5, label=country)

ax.set_title("Заповненість ПСГ: ключові країни ЄС", color="white", fontsize=10, fontweight="bold")
ax.set_ylabel("%", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)
month_fmt(ax)

# ── Panel 4: Cross-border flows ───────────────────────────────────────
ax = ax_flow
pos = net_flow.clip(lower=0)
neg = net_flow.clip(upper=0)
ax.fill_between(net_flow.index, pos, alpha=0.55, color="#27ae60", label="Нетто-вихід (транзит/експорт)")
ax.fill_between(net_flow.index, neg, alpha=0.55, color="#e74c3c", label="Нетто-вхід (імпорт)")
ax.plot(net_flow.index, net_flow.values, color="#ecf0f1", lw=0.8, alpha=0.7)
ax.axhline(0, color="#555577", lw=0.8)

ax.set_title("Транскордонні потоки ГТС: нетто тижнева середня (млн м³/добу) — весна 2025",
             color="white", fontsize=12, fontweight="bold")
ax.set_ylabel("млн м³/добу", color=AXIS_C, fontsize=9)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=9)
month_fmt(ax)

# ── Panel 5: Top border points ────────────────────────────────────────
ax = ax_pts
points = top_points.index.str[:22]
x = np.arange(len(points))
w = 0.35
out_vals = top_points.get("Вихід", pd.Series(0, index=top_points.index)).values
in_vals  = top_points.get("Вхід",  pd.Series(0, index=top_points.index)).values

bars1 = ax.barh(x + w/2, out_vals, w, color="#27ae60", alpha=0.8, label="Вихід (середнє млн м³/д)")
bars2 = ax.barh(x - w/2, in_vals,  w, color="#e74c3c", alpha=0.8, label="Вхід")
ax.set_yticks(x)
ax.set_yticklabels(points, fontsize=8, color=AXIS_C)
ax.set_title("Топ точки підключення: середній добовий обсяг", color="white", fontsize=10, fontweight="bold")
ax.set_xlabel("млн м³/добу", color=AXIS_C, fontsize=8)
ax.legend(framealpha=0.2, labelcolor="white", fontsize=8)

# ── Panel 6: KPI summary table ────────────────────────────────────────
ax = ax_kpi
ax.axis("off")

yoy_arrow = "▲" if yoy_pct > 0 else "▼"
yoy_col   = "#e74c3c" if yoy_pct > 0 else "#2ecc71"

kpis = [
    ("ЦІНИ ГАЗУ (TTF, весна 2025)", None, "header"),
    ("Середня ціна", f"{ttf_mean:,.0f} грн/1000м³", "normal"),
    ("Мінімум", f"{ttf_min:,.0f} ({ttf_min_d})", "green"),
    ("Максимум", f"{ttf_max:,.0f} ({ttf_max_d})", "red"),
    (f"Зміна до весни 2024", f"{yoy_arrow} {abs(yoy_pct):.1f}%", "yoy"),
    ("", "", "spacer"),
    ("ПСГ УКРАЇНИ (весна 2025)", None, "header"),
    ("Початок березня", f"{ua_start:.1f}%", "normal"),
    ("Мінімум", f"{ua_min:.1f}% ({ua_min_d})", "red"),
    ("Кінець травня", f"{ua_end:.1f}%", "green"),
    ("Зміна за весну", f"+{ua_end-ua_start:.1f} п.п.", "green" if ua_end > ua_start else "red"),
    ("", "", "spacer"),
    ("ПОТОКИ ГТС (весна 2025)", None, "header"),
    ("Характер", "Нетто-вхід (імпорт)", "red"),
    ("Тижнева середня нетто", f"{net_flow.mean():.2f} млн м³/д", "normal"),
]

y = 0.97
for label, value, style in kpis:
    if style == "spacer":
        y -= 0.025
        continue
    if style == "header":
        ax.text(0.03, y, label, transform=ax.transAxes,
                fontsize=9.5, color="#f0c040", fontweight="bold",
                va="top")
        ax.plot([0.02, 0.98], [y-0.005, y-0.005],
                color="#333355", lw=0.8, transform=ax.transAxes)
        y -= 0.06
        continue

    color_map = {"normal": AXIS_C, "green": "#2ecc71", "red": "#e74c3c",
                 "yoy": yoy_col}
    val_col = color_map.get(style, AXIS_C)

    ax.text(0.03, y, label, transform=ax.transAxes,
            fontsize=9, color="#9999bb", va="top")
    ax.text(0.97, y, value, transform=ax.transAxes,
            fontsize=9, color=val_col, va="top", ha="right", fontweight="bold")
    y -= 0.055

ax.set_title("Ключові показники", color="white", fontsize=10, fontweight="bold")

# ── Suptitle ──────────────────────────────────────────────────────────
fig.suptitle("Газовий ринок України: аналітика весни 2025  |  березень–травень",
             color="white", fontsize=14, fontweight="bold", y=0.96)
fig.text(0.5, 0.005,
         "Джерела: ОГТСУ (фізичні потоки), energy-map.info (ціни), GIE AGSI (ПСГ ЄС+UA)",
         ha="center", color="#666688", fontsize=8)

out_path = OUT / "spring_2025_analytics.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved: {out_path}")

# ── Print text summary ────────────────────────────────────────────────
print("\n" + "="*55)
print("  АНАЛІТИКА ГАЗОВОГО РИНКУ: ВЕСНА 2025")
print("="*55)
print(f"\nЦІНИ (TTF, без ПДВ, грн/1000 м³):")
print(f"  Середня : {ttf_mean:>10,.0f}")
print(f"  Мін     : {ttf_min:>10,.0f}  ({ttf_min_d})")
print(f"  Макс    : {ttf_max:>10,.0f}  ({ttf_max_d})")
print(f"  До 2024 : {yoy_arrow} {abs(yoy_pct):.1f}%  (середня 2024: {ttf24_mean:,.0f})")

print(f"\nПСГ УКРАЇНИ (% заповнення):")
print(f"  01.03   : {ua_start:.1f}%")
print(f"  Мінімум : {ua_min:.1f}%  ({ua_min_d})")
print(f"  31.05   : {ua_end:.1f}%")
print(f"  Зміна   : {ua_end-ua_start:+.1f} п.п.")

print(f"\nПСГ ЄС (зважене середнє):")
print(f"  01.03   : {eu_total.iloc[0]:.1f}%")
print(f"  31.05   : {eu_total.iloc[-1]:.1f}%")

print(f"\nТРАНСКОРДОННІ ПОТОКИ:")
net_by_month = net_flow.resample("ME").mean()
for month, val in net_by_month.items():
    direction = "транзит/вихід" if val > 0 else "імпорт/вхід"
    print(f"  {month.strftime('%b')}: {val:+.2f} млн м³/д  ({direction})")
