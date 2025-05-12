import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_features(df: pd.DataFrame,
                     target_column: str = 'is_narrow',
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
            print(f"Skipping {col}: too many missing values")
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
            print(f"Skipping {col}: too many missing values")
            continue

        freq = pd.crosstab(df[col], df[target_column], normalize='index')
        feature_stats[col] = freq.to_dict()

        if output_dir and len(df[col].unique()) <= 20:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x=col, hue=target_column)
            plt.title(f"Distribution of {col} by {target_column}")
            plt.xticks(rotation=45)
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
        print(f"No models directory found at: {models_dir}")
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
            print(f"Error loading model {model_file}: {str(e)}")

    return models_info
