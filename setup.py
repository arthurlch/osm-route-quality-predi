from setuptools import setup, find_packages

setup(
    name="street_quality_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "networkx",
        "joblib",
    ],
)
