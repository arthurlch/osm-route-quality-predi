import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def prepare_model_data(edges: pd.DataFrame, quality_streets: pd.DataFrame,
                       target_col: str = 'is_quality') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare data for model training by combining edges and quality streets dataframes!
    """

    df = edges.copy()
    df[target_col] = 0
    df.loc[quality_streets.index, target_col] = 1

    # Extract relevant features (decided based on what feature and data are important)
    feats = [c for c in ['highway', 'lanes', 'maxspeed', 'service', 'length', 'width']
             if c in df.columns]

    return df[feats], df[target_col], feats


def build_model(X: pd.DataFrame, y: pd.Series,
                test_size: float = 0.3,
                random_state: int = 42) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:

    num_features = X.select_dtypes(include=['number']).columns.tolist()
    cat_features = X.select_dtypes(
        include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, random_state=random_state))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model: Pipeline,
               features: List[str],
               region: str,
               model_dir: str = 'models') -> str:

    os.makedirs(model_dir, exist_ok=True)

    safe_region_name = region.replace(', ', '_').replace(' ', '_')
    model_path = os.path.join(
        model_dir, f"street_quality_{safe_region_name}.joblib")

    metadata = {
        'features': features,
        'region': region,
        'creation_date': pd.Timestamp.now().isoformat()
    }

    joblib.dump({'model': model, 'metadata': metadata}, model_path)

    print(f"Model saved: {model_path}")
    return model_path


def load_model(model_path: str) -> Tuple[Pipeline, Dict[str, Any]]:

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data = joblib.load(model_path)

    model = data['model']
    metadata = data['metadata']

    print(f"Loaded model trained on: {metadata['region']}")
    print(f"Model features: {metadata['features']}")

    return model, metadata
