from setuptools import setup, find_packages

setup(
    name="gas-analyzer",
    version="1.0.0",
    description="Gas composition analysis and forecasting system",
    packages=find_packages(where="."),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "loguru>=0.7",
        "tqdm>=4.65",
    ],
    extras_require={
        "ml": ["xgboost>=1.7", "lightgbm>=4.0"],
        "prophet": ["prophet>=1.1"],
        "deep": ["torch>=2.0"],
        "db": ["sqlalchemy>=2.0", "psycopg2-binary>=2.9"],
        "mqtt": ["paho-mqtt>=1.6"],
        "serial": ["pyserial>=3.5"],
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "jupyter>=1.0"],
    },
)
