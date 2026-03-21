"""
Generate a sample CSV dataset for testing the Gas Analyzer pipeline.
Run: python data/samples/generate_sample.py
Output: data/samples/sample_measurements.csv
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.ingestion.loader import DataLoader

out = Path(__file__).parent / "sample_measurements.csv"
df = DataLoader.load_sample()
df.to_csv(out)
print(f"Sample dataset written to {out}  ({len(df):,} rows)")
