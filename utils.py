from .train import STANDARD_TARGET, STANDARD_FEATURES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_features(df: pd.DataFrame,
                     target_column: str = STANDARD_TARGET,
                     output_dir: Optional[str] = None) -> dict:

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    feature_stats = {}

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    for col in numeric_cols:
        if df[col].isna().sum() > 0.5 * len(df):
            logger.info(f"Skipping {col}: too many missing values")
            continue

        stats = df.groupby(target_column)[col].agg(
            ['mean', 'median', 'std', 'count']).to_dict()
        feature_stats[col] = stats

        if output_dir:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, hue=target_column,
                         kde=True, element="step")
            plt.title(f"Distribution of {col} by {target_column}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
            plt.close()

    for col in categorical_cols:
        if df[col].isna().sum() > 0.5 * len(df):
            logger.info(f"Skipping {col}: too many missing values")
            continue

        freq = pd.crosstab(df[col], df[target_column], normalize='index')
        feature_stats[col] = freq.to_dict()

        if output_dir and len(df[col].unique()) <= 15:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=col, hue=target_column, data=df)
            plt.title(f"Distribution of {col} by {target_column}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
            plt.close()

    if len(numeric_cols) > 1 and output_dir:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlation.png"))
        plt.close()

    return feature_stats


def find_available_models(models_dir: str = 'models') -> List[Dict[str, Any]]:

    import joblib

    if not os.path.exists(models_dir):
        logger.warning(f"No models directory found at: {models_dir}")
        return []

    models_info = []

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            data = joblib.load(model_path)
            metadata = data.get('metadata', {})

            model_info = {
                'filename': model_file,
                'path': model_path,
                'region': metadata.get('region', 'Unknown'),
                'features': metadata.get('features', []),
                'creation_date': metadata.get('creation_date', 'Unknown')
            }

            models_info.append(model_info)
        except Exception as e:
            logger.warning(f"Error loading model {model_file}: {str(e)}")

    return models_info


def check_data_quality(edges: pd.DataFrame) -> Dict[str, Any]:
    total_rows = len(edges)

    feature_counts = {}
    for feature in STANDARD_FEATURES:
        if feature in edges.columns:
            non_null_count = edges[feature].notna().sum()
            feature_counts[feature] = {
                'present': True,
                'non_null_count': non_null_count,
                'coverage_percent': (non_null_count / total_rows) * 100
            }
        else:
            feature_counts[feature] = {
                'present': False,
                'non_null_count': 0,
                'coverage_percent': 0
            }

    present_features = [f for f in STANDARD_FEATURES if f in edges.columns]
    if present_features:
        coverage_scores = [edges[f].notna().mean() for f in present_features]
        overall_quality = sum(coverage_scores) / len(coverage_scores)
    else:
        overall_quality = 0

    outliers = {}
    for col in edges.select_dtypes(include=['number']).columns:
        if edges[col].notna().sum() > 10:  # Only check columns with sufficient data
            q1 = edges[col].quantile(0.25)
            q3 = edges[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = ((edges[col] < lower_bound)
                             | (edges[col] > upper_bound)).sum()
            outliers[col] = {
                'outlier_count': outlier_count,
                'outlier_percent': (outlier_count / edges[col].notna().sum()) * 100
            }

    return {
        'total_rows': total_rows,
        'feature_counts': feature_counts,
        'overall_quality_score': overall_quality,
        'outliers': outliers,
        'recommendation': get_recommendation(overall_quality, feature_counts)
    }


def get_recommendation(quality_score: float, feature_counts: Dict[str, Dict[str, Any]]) -> str:

    if quality_score < 0.3:
        return "Data quality is poor. Consider collecting more data before model training."

    if quality_score < 0.7:
        missing_features = []
        for feature, info in feature_counts.items():
            if not info['present'] or info['coverage_percent'] < 30:
                missing_features.append(feature)

        if missing_features:
            return f"Data is incomplete. Focus on collecting: {', '.join(missing_features)}"
        else:
            return "Data quality is moderate. Consider data augmentation techniques."

    return "Data quality is good. Proceed with model training."

import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import Optional, Dict, Any, Union

def load_network(source: str, network_type: str = 'drive',
                 adapter_type: Optional[str] = None,
                 adapter_config: Optional[Dict[str, Any]] = None) -> Union[Any, nx.MultiDiGraph]:

    print(f"Loading network from {source} with type {network_type}")
    
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, geometry=None, highway='residential', length=100)
    G.add_edge(1, 2, geometry=None, highway='residential', length=200)
    
    return G


def extract_edges(data: Any, adapter_type: Optional[str] = None,
                  adapter_config: Optional[Dict[str, Any]] = None) -> gpd.GeoDataFrame:
  
    print(f"Extracting edges with adapter type {adapter_type}")
    
    import shapely.geometry
    
    data = {
        'highway': ['residential', 'residential', 'primary'],
        'length': [100, 200, 300],
        'width': [5, 6, 10],
        'lanes': [1, 1, 2],
        'maxspeed': [30, 30, 50],
        'geometry': [
            shapely.geometry.LineString([(0, 0), (1, 1)]),
            shapely.geometry.LineString([(1, 1), (2, 2)]),
            shapely.geometry.LineString([(2, 2), (3, 3)])
        ]
    }
    
    gdf = gpd.GeoDataFrame(data, geometry='geometry')
    return gdf