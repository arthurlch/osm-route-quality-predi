import os
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, Union, Dict, Any
from sklearn.pipeline import Pipeline


def predict_quality_streets(model: Pipeline,
                            edges: gpd.GeoDataFrame,
                            threshold: float = 0.5,
                            output_path: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Apply a trained model to predict quality streets in a new region    """
    df = edges.copy()

    try:
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            expected_features = []
            for name, _ in model.named_steps['preprocessor'].transformers:
                if name != 'remainder':
                    transformer_features = model.named_steps['preprocessor'].named_transformers_[
                        name].feature_names_in_
                    expected_features.extend(transformer_features)

            features = [f for f in expected_features if f in df.columns]
        else:
            features = [c for c in ['highway', 'lanes', 'maxspeed', 'service', 'length', 'width']
                        if c in df.columns]
    except (AttributeError, KeyError):
        features = [c for c in ['highway', 'lanes', 'maxspeed', 'service', 'length', 'width']
                    if c in df.columns]

    print(f"Using features: {features}")

    for col in ['width', 'lanes', 'maxspeed', 'length']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[features]

    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
        df['quality_probability'] = probas[:, 1]
        df['predicted_quality'] = (
            df['quality_probability'] >= threshold).astype(int)
    else:
        df['predicted_quality'] = model.predict(X)
        df['quality_probability'] = df['predicted_quality'].astype(float)

    quality_streets = df[df['predicted_quality'] == 1].copy()

    quality_streets['quality_score'] = quality_streets['quality_probability']

    print(
        f"Predicted {len(quality_streets)} quality streets out of {len(df)} total")

    if output_path:
        quality_streets.to_csv(output_path, index=False)
        print(f"Saved predicted quality streets to: {output_path}")

    return quality_streets


def transfer_model(model_path: str,
                   new_region: str,
                   network_type: str = 'drive',
                   output_dir: str = '.',
                   adapter_type: Optional[str] = None,
                   adapter_config: Optional[Dict[str, Any]] = None) -> Tuple[gpd.GeoDataFrame, str]:
    """
    Apply the trained model to a new region """
    from ..extract import load_network, extract_edges
    from .train import load_model

    model, metadata = load_model(model_path)
    print(
        f"Using model trained on {metadata['region']} to predict quality streets in {new_region}")

    G = load_network(new_region, network_type, adapter_type, adapter_config)
    edges = extract_edges(G, adapter_type, adapter_config)
    print(f"Extracted {len(edges)} edges from {new_region}")

    safe_region_name = new_region.replace(', ', '_').replace(' ', '_')
    output_path = os.path.join(
        output_dir, f"predicted_quality_{safe_region_name}.csv")

    quality_streets = predict_quality_streets(
        model, edges, output_path=output_path)

    return quality_streets, output_path
