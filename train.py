import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Dict, Any, Optional, Set
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


STANDARD_FEATURES = ['highway', 'lanes', 'maxspeed', 'service', 'length', 'width',
                     'surface', 'cycleway', 'sidewalk', 'lit', 'access', 'oneway']

STANDARD_TARGET = 'is_quality'


def prepare_model_data(edges: pd.DataFrame,
                       quality_streets: pd.DataFrame,
                       target_col: str = STANDARD_TARGET,
                       additional_features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:

    df = edges.copy()

    df[target_col] = 0
    df.loc[quality_streets.index, target_col] = 1

    available_features = set(df.columns)
    base_features = set(STANDARD_FEATURES)

    if additional_features:
        base_features.update(additional_features)

    features = list(base_features.intersection(available_features))

    if not features:
        raise ValueError("No usable features found in the dataset")

    for col in ['width', 'lanes', 'maxspeed', 'length']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Using features: {features}")
    return df[features], df[target_col], features


def build_model(X: pd.DataFrame,
                y: pd.Series,
                test_size: float = 0.3,
                random_state: int = 42,
                optimize: bool = False) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough'  # Include columns that don't match either transformer
    )

    if optimize:
        classifier = GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid={
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            cv=3,
            n_jobs=-1,
            verbose=1
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Training model...")
    model.fit(X_train, y_train)

    if optimize:
        print(
            f"Best parameters: {model.named_steps['classifier'].best_params_}")

    return model, X_test, y_test


def save_model(model: Pipeline,
               features: List[str],
               region: str,
               model_dir: str = 'models',
               additional_metadata: Optional[Dict[str, Any]] = None) -> str:

    os.makedirs(model_dir, exist_ok=True)

    safe_region_name = region.replace(', ', '_').replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        model_dir, f"street_quality_{safe_region_name}_{timestamp}.joblib")

    metadata = {
        'features': features,
        'region': region,
        'creation_date': pd.Timestamp.now().isoformat(),
        'standard_features': STANDARD_FEATURES,
        'target_column': STANDARD_TARGET
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    joblib.dump({'model': model, 'metadata': metadata}, model_path)

    print(f"Model saved: {model_path}")
    return model_path


def load_model(model_path: str) -> Tuple[Pipeline, Dict[str, Any]]:

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    data = joblib.load(model_path)

    model = data['model']
    metadata = data.get('metadata', {})

    print(f"Loaded model trained on: {metadata.get('region', 'Unknown')}")
    print(f"Model features: {metadata.get('features', [])}")

    return model, metadata
