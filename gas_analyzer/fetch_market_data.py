"""
Фетчер ринкових даних з OilPriceAPI + NBU API.
Оновлює CSV-файли в data/external/ актуальними даними.

Запуск:
    python fetch_market_data.py
    python fetch_market_data.py --from 2025-01-01 --to 2026-03-31
    python fetch_market_data.py --symbol TTF --dry-run

Залежності: pip install requests pandas
API ключ OilPriceAPI: вкажіть у змінній середовища або параметрі --api-token
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

# ── Налаштування ──────────────────────────────────────────────────────
API_TOKEN = os.environ.get(
    "OILPRICE_API_TOKEN",
    "5801d4dabe85341ae30085cc5c33a5d5381482555e0d6fbd46fca5b7d4cefe7b",
)
BASE_URL   = "https://oilpriceapi.com/api/v1"
EXT_DIR    = Path(__file__).parent / "data" / "external"

# OilPriceAPI коди для пошуку TTF (в порядку пріоритету)
TTF_CODES  = ["TTF", "EU_TTF", "NG_EU_TTF", "NATGAS_TTF", "NG_EU"]

# NBU API (відкритий, без ключа)
NBU_FX_URL = "https://bank.gov.ua/NBU_Exchange/exchange_site"


# ════════════════════════════════════════════════════════════════════════
# 1. OilPriceAPI — пошук доступних символів
# ════════════════════════════════════════════════════════════════════════

def api_get(endpoint: str, params: dict) -> dict | None:
    """GET-запит до OilPriceAPI з обробкою помилок."""
    params["api_token"] = API_TOKEN
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        print(f"  [warn] {url}: HTTP {resp.status_code} — {resp.text[:200]}")
    except requests.RequestException as exc:
        print(f"  [error] {url}: {exc}")
    return None


def find_ttf_code() -> str | None:
    """Знайти правильний API-код для TTF серед TTF_CODES."""
    print("Пошук TTF-символу в OilPriceAPI...")
    for code in TTF_CODES:
        data = api_get("prices/latest", {"by_code": code})
        if data and data.get("data"):
            price = data["data"].get("price") or data["data"].get("price_close")
            if price:
                print(f"  Знайдено: {code} = {price} EUR/MWh")
                return code
        time.sleep(0.5)
    print("  Не вдалось знайти TTF-символ автоматично.")
    return None


# ════════════════════════════════════════════════════════════════════════
# 2. Завантаження денних TTF → агрегація місячно
# ════════════════════════════════════════════════════════════════════════

def fetch_ttf_daily(code: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Завантажує денні ціни TTF і повертає DataFrame."""
    print(f"  Завантаження TTF ({code}): {date_from} — {date_to}")
    records = []

    # OilPriceAPI може повертати дані сторінками або єдиним блоком
    data = api_get("prices", {
        "by_code": code,
        "from":    date_from,
        "to":      date_to,
    })

    if data and isinstance(data.get("data"), list):
        for row in data["data"]:
            try:
                dt   = row.get("created_at", row.get("date", ""))[:10]
                price = float(row.get("price") or row.get("price_close") or 0)
                if price > 0:
                    records.append({"date": pd.to_datetime(dt), "price": price})
            except (ValueError, TypeError):
                pass
    elif data and isinstance(data.get("data"), dict):
        # Деякі ендпоінти повертають єдиний запис
        row = data["data"]
        dt    = str(row.get("created_at", row.get("date", date_to)))[:10]
        price = float(row.get("price") or row.get("price_close") or 0)
        if price > 0:
            records.append({"date": pd.to_datetime(dt), "price": price})

    if not records:
        print("    Дані відсутні або порожні.")
        return pd.DataFrame(columns=["date", "price"])

    df = pd.DataFrame(records).drop_duplicates("date").sort_values("date")
    print(f"    Отримано {len(df)} денних записів.")
    return df


def aggregate_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Агрегує денні дані до місячних середніх."""
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "ttf_eur_mwh"])
    daily_df = daily_df.copy()
    daily_df["month"] = daily_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (daily_df.groupby("month")["price"]
               .mean()
               .reset_index()
               .rename(columns={"month": "date", "price": "ttf_eur_mwh"}))
    monthly["ttf_eur_mwh"] = monthly["ttf_eur_mwh"].round(2)
    return monthly


# ════════════════════════════════════════════════════════════════════════
# 3. NBU API — курс EUR/UAH
# ════════════════════════════════════════════════════════════════════════

def fetch_nbu_eur_uah(date_from: str, date_to: str) -> pd.DataFrame:
    """Завантажує місячний середній курс EUR/UAH з NBU API."""
    print(f"  Завантаження EUR/UAH NBU: {date_from} — {date_to}")
    records = []
    current = pd.Timestamp(date_from).to_period("M")
    end     = pd.Timestamp(date_to).to_period("M")

    while current <= end:
        month_str = current.to_timestamp().strftime("%Y%m")
        try:
            resp = requests.get(NBU_FX_URL, params={
                "start": month_str + "01",
                "end":   current.to_timestamp(how="end").strftime("%Y%m%d"),
                "valcode": "EUR",
                "sort":    "exchangedate",
                "order":   "desc",
                "json":    "",
            }, timeout=10)
            if resp.status_code == 200:
                rows = resp.json()
                if rows:
                    rates = [float(r["rate"]) for r in rows if r.get("rate")]
                    if rates:
                        avg_rate = round(sum(rates) / len(rates), 4)
                        records.append({
                            "date": current.to_timestamp(),
                            "eur_uah": avg_rate,
                        })
        except Exception as exc:
            print(f"    [warn] NBU {current}: {exc}")
        time.sleep(0.3)
        current += 1

    if not records:
        return pd.DataFrame(columns=["date", "eur_uah"])
    df = pd.DataFrame(records).sort_values("date")
    print(f"    Отримано {len(df)} місячних ставок.")
    return df


# ════════════════════════════════════════════════════════════════════════
# 4. Оновлення CSV-файлів
# ════════════════════════════════════════════════════════════════════════

def merge_and_save(
    new_df:     pd.DataFrame,
    csv_path:   Path,
    date_col:   str,
    value_col:  str,
    source_tag: str,
    confidence: str = "confirmed",
    dry_run:    bool = False,
) -> int:
    """
    Об'єднує нові дані з існуючим CSV.
    Нові рядки додаються; існуючі з confidence='approximate'/'estimated'
    перезаписуються якщо нові дані є підтвердженими.
    Повертає кількість оновлених/доданих рядків.
    """
    if new_df.empty:
        return 0

    new_df = new_df.copy()
    new_df[date_col] = pd.to_datetime(new_df[date_col])

    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=[date_col])
    else:
        existing = pd.DataFrame(columns=[date_col, value_col, "source", "confidence"])

    existing[date_col] = pd.to_datetime(existing[date_col])
    existing = existing.set_index(date_col)

    updated = 0
    for _, row in new_df.iterrows():
        dt  = row[date_col]
        val = row[value_col]
        if dt in existing.index:
            old_conf = existing.loc[dt, "confidence"]
            if old_conf in ("approximate", "estimated"):
                existing.loc[dt, value_col]    = val
                existing.loc[dt, "source"]     = source_tag
                existing.loc[dt, "confidence"] = confidence
                updated += 1
        else:
            existing.loc[dt] = {
                value_col:    val,
                "source":     source_tag,
                "confidence": confidence,
            }
            updated += 1

    result = existing.sort_index().reset_index()
    result[date_col] = result[date_col].dt.strftime("%Y-%m-%d")

    if dry_run:
        print(f"  [dry-run] {csv_path.name}: {updated} рядків було б оновлено.")
    else:
        result.to_csv(csv_path, index=False)
        print(f"  Збережено {csv_path.name}: {updated} рядків оновлено/додано.")

    return updated


# ════════════════════════════════════════════════════════════════════════
# 5. Швидкий тест з'єднання
# ════════════════════════════════════════════════════════════════════════

def test_connection() -> bool:
    """Перевіряє доступність OilPriceAPI."""
    data = api_get("prices/latest", {})
    if data:
        print("  OilPriceAPI: з'єднання OK.")
        return True
    print("  OilPriceAPI: з'єднання НЕВДАЛЕ. Перевірте API-ключ і мережу.")
    return False


# ════════════════════════════════════════════════════════════════════════
# 6. Головна функція
# ════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Фетчер ринкових даних (OilPriceAPI + NBU)")
    p.add_argument("--from", dest="date_from", default="2024-01-01",
                   help="Початкова дата (YYYY-MM-DD)")
    p.add_argument("--to", dest="date_to",
                   default=date.today().strftime("%Y-%m-%d"),
                   help="Кінцева дата (YYYY-MM-DD)")
    p.add_argument("--symbol", default=None,
                   help="API-код для TTF (авто-визначення якщо не вказано)")
    p.add_argument("--api-token", default=None,
                   help="OilPriceAPI токен (або змінна OILPRICE_API_TOKEN)")
    p.add_argument("--no-nbu", action="store_true",
                   help="Не завантажувати курс EUR/UAH з NBU")
    p.add_argument("--dry-run", action="store_true",
                   help="Не зберігати файли, лише показати результат")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global API_TOKEN
    if args.api_token:
        API_TOKEN = args.api_token

    print("=" * 60)
    print("  Gas Market Data Fetcher")
    print(f"  Період: {args.date_from} — {args.date_to}")
    print("=" * 60)

    # ── Перевірка з'єднання ────────────────────────────────────────
    print("\n[1/3] Перевірка з'єднання...")
    if not test_connection():
        sys.exit(1)

    # ── TTF ────────────────────────────────────────────────────────
    print("\n[2/3] Завантаження TTF...")
    code = args.symbol or find_ttf_code()
    if code:
        daily_ttf  = fetch_ttf_daily(code, args.date_from, args.date_to)
        monthly_ttf = aggregate_monthly(daily_ttf)
        if not monthly_ttf.empty:
            print(f"\n  Місячні середні TTF (EUR/MWh):")
            print(monthly_ttf.to_string(index=False))
            merge_and_save(
                new_df    = monthly_ttf,
                csv_path  = EXT_DIR / "ttf_monthly_eur_mwh.csv",
                date_col  = "date",
                value_col = "ttf_eur_mwh",
                source_tag= f"OilPriceAPI/{code} (auto-fetched {date.today()})",
                confidence= "confirmed",
                dry_run   = args.dry_run,
            )
    else:
        print("  TTF-дані пропущено: символ не знайдено.")

    # ── EUR/UAH ────────────────────────────────────────────────────
    if not args.no_nbu:
        print("\n[3/3] Завантаження EUR/UAH (NBU)...")
        fx_df = fetch_nbu_eur_uah(args.date_from, args.date_to)
        if not fx_df.empty:
            print(f"\n  Місячний курс EUR/UAH:")
            print(fx_df.tail(6).to_string(index=False))
            merge_and_save(
                new_df    = fx_df,
                csv_path  = EXT_DIR / "eur_uah_monthly.csv",
                date_col  = "date",
                value_col = "eur_uah",
                source_tag= f"NBU API (auto-fetched {date.today()})",
                confidence= "confirmed",
                dry_run   = args.dry_run,
            )
    else:
        print("\n[3/3] NBU EUR/UAH пропущено (--no-nbu).")

    print("\n" + "=" * 60)
    print("  Готово.")
    print(f"  Файли збережено: {EXT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
