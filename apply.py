import os
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.pipeline import Pipeline
import logging

from .utils import load_network, extract_edges, STANDARD_FEATURES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def predict_quality_streets(model: Pipeline,
                            edges: gpd.GeoDataFrame,
                            threshold: float = 0.5,
                            output_path: Optional[str] = None,
                            return_all: bool = False) -> gpd.GeoDataFrame:

    logger.info(f"Predicting quality streets for {len(edges)} edges")

    df = edges.copy()

    features = []

    if hasattr(model, '__getitem__') and isinstance(model, dict) and 'metadata' in model:
        features = [f for f in model['metadata'].get(
            'features', []) if f in df.columns]
        model = model['model']  # Get actual model

    if not features:
        features = [f for f in STANDARD_FEATURES if f in df.columns]

    logger.info(f"Using features: {features}")

    for col in ['width', 'lanes', 'maxspeed', 'length']:
        if col in df.columns and col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[features]

    if hasattr(model, 'predict_proba'):
        try:
            probas = model.predict_proba(X)
            df['quality_probability'] = probas[:, 1]
            df['predicted_quality'] = (
                df['quality_probability'] >= threshold).astype(int)
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            df['predicted_quality'] = model.predict(X)
            df['quality_probability'] = df['predicted_quality'].astype(float)
    else:
        df['predicted_quality'] = model.predict(X)
        df['quality_probability'] = df['predicted_quality'].astype(float)

    df['quality_score'] = df['quality_probability']

    if not return_all:
        result = df[df['predicted_quality'] == 1].copy()
    else:
        result = df.copy()

    logger.info(
        f"Predicted {len(df[df['predicted_quality'] == 1])} quality streets out of {len(df)} total")

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved prediction results to: {output_path}")

    return result


def transfer_model(model_path: str,
                   new_region: str,
                   network_type: str = 'drive',
                   output_dir: str = '.') -> Tuple[gpd.GeoDataFrame, str]:

    from .train import load_model

    model, metadata = load_model(model_path)
    logger.info(
        f"Using model trained on {metadata.get('region', 'Unknown')} to predict quality streets in {new_region}")

    G = load_network(new_region, network_type)
    edges = extract_edges(G)
    logger.info(f"Extracted {len(edges)} edges from {new_region}")

    safe_region_name = new_region.replace(', ', '_').replace(' ', '_')
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, f"predicted_quality_{safe_region_name}_{timestamp}.csv")

    quality_streets = predict_quality_streets(
        model, edges, output_path=output_path)

    return quality_streets, output_path


def apply_ensemble(model_paths: List[str],
                   new_region: str,
                   network_type: str = 'drive',
                   output_dir: str = '.',
                   weights: Optional[List[float]] = None) -> Tuple[gpd.GeoDataFrame, str]:

    from .train import load_model

    G = load_network(new_region, network_type)
    edges = extract_edges(G)
    logger.info(f"Extracted {len(edges)} edges from {new_region}")

    df = edges.copy()
    df['quality_probability'] = 0.0

    if weights is not None:
        if len(weights) != len(model_paths):
            raise ValueError(
                f"Number of weights ({len(weights)}) does not match number of models ({len(model_paths)})")
        weights = np.array(weights) / sum(weights)
    else:
        weights = np.ones(len(model_paths)) / len(model_paths)

    for i, model_path in enumerate(model_paths):
        model, metadata = load_model(model_path)
        logger.info(
            f"Using model {i+1}/{len(model_paths)} trained on {metadata.get('region', 'Unknown')}")

        prediction = predict_quality_streets(model, edges, return_all=True)
        df['quality_probability'] += weights[i] * \
            prediction['quality_probability']

    df['predicted_quality'] = (df['quality_probability'] >= 0.5).astype(int)
    df['quality_score'] = df['quality_probability']

    quality_streets = df[df['predicted_quality'] == 1].copy()

    safe_region_name = new_region.replace(', ', '_').replace(' ', '_')
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir, f"ensemble_quality_{safe_region_name}_{timestamp}.csv")
    quality_streets.to_csv(output_path, index=False)

    logger.info(
        f"Ensemble predicted {len(quality_streets)} quality streets out of {len(df)} total")
    logger.info(f"Saved ensemble prediction results to: {output_path}")

    return quality_streets, output_path
