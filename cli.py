#!/usr/bin/env python
"""
Command-line interface for street quality prediction with sparse data handling.
"""
from evaluate import evaluate_model
from apply import predict_quality_streets, apply_ensemble
from train import prepare_model_data, build_model, save_model, load_model
from utils import (
    load_network, extract_edges, enrich_sparse_data,
    STANDARD_TARGET, STANDARD_FEATURES, analyze_features, check_data_quality, find_similar_regions
)
import argparse
import logging
import os
import sys
import pandas as pd
import geopandas as gpd
from typing import List

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(args):
    G = load_network(args.input)
    edges = extract_edges(G)

    quality_check = check_data_quality(edges)
    logger.info(
        f"Data quality score: {quality_check['overall_quality_score']:.2f}")
    logger.info(f"Recommendation: {quality_check['recommendation']}")

    if args.sparse or quality_check['overall_quality_score'] < 0.7:
        logger.info(
            "Enhancing sparse data with derived features and imputation")
        edges = enrich_sparse_data(edges)

    if args.quality.endswith('.csv'):
        quality_streets = pd.read_csv(args.quality)
    elif args.quality.endswith(('.geojson', '.shp')):
        quality_streets = gpd.read_file(args.quality)
    else:
        raise ValueError(f"Unsupported quality streets format: {args.quality}")

    if args.analyze:
        logger.info("Analyzing feature relationships")
        analysis_dir = os.path.join(
            'analysis', args.region.replace(', ', '_').replace(' ', '_'))
        os.makedirs(analysis_dir, exist_ok=True)
        analyze_features(edges, STANDARD_TARGET, output_dir=analysis_dir)
        logger.info(f"Analysis saved to {analysis_dir}")

    X, y, features = prepare_model_data(edges, quality_streets,
                                        additional_features=args.additional_features)
    model, X_test, y_test = build_model(X, y)

    os.makedirs('models', exist_ok=True)
    model_path = save_model(model, features, args.region, model_dir='models')

    logger.info(f"Model saved to {model_path}")

    if args.evaluate:
        metrics = evaluate_model(
            model, X_test, y_test, output_dir='models/evaluation')
        logger.info(
            f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

    return 0


def predict(args):
    model, metadata = load_model(args.model)
    logger.info(f"Loaded model trained on {metadata.get('region', 'Unknown')}")

    G = load_network(args.input)
    edges = extract_edges(G)

    quality_check = check_data_quality(edges)
    logger.info(
        f"Data quality score: {quality_check['overall_quality_score']:.2f}")

    if args.sparse or quality_check['overall_quality_score'] < 0.7:
        logger.info(
            "Enhancing sparse data with derived features and imputation")
        edges = enrich_sparse_data(edges)

    quality_streets = predict_quality_streets(
        model, edges, output_path=args.output)

    if 'data_completeness' in edges.columns:
        quality_streets['prediction_confidence'] = quality_streets['quality_probability'] * \
            quality_streets['data_completeness']
        logger.info(
            "Added prediction confidence scores based on data completeness")

    quality_streets.to_csv(args.output, index=False)
    logger.info(
        f"Predicted {len(quality_streets)} quality streets. Results saved to {args.output}")

    return 0


def auto_ensemble(args):
    similar_regions = find_similar_regions(args.input)

    if not similar_regions:
        logger.error("No similar regions found with trained models")
        return 1

    logger.info(
        f"Found {len(similar_regions)} similar regions with trained models")

    model_paths = []
    weights = []

    for region, similarity in similar_regions:
        from glob import glob
        region_safe = region.replace(', ', '_').replace(' ', '_')
        model_pattern = os.path.join(
            'models', f"street_quality_{region_safe}_*.joblib")
        region_models = glob(model_pattern)

        if region_models:
            model_paths.append(sorted(region_models)[-1])
            weights.append(similarity)

    if not model_paths:
        logger.error("No models found for similar regions")
        return 1

    #  weights
    weights = [w / sum(weights) for w in weights]

    logger.info(
        f"Using ensemble of {len(model_paths)} models with weights: {weights}")

    G = load_network(args.input)
    edges = extract_edges(G)

    edges = enrich_sparse_data(edges)

    quality_streets, output_path = apply_ensemble(
        model_paths, args.input, output_dir=args.output, weights=weights
    )

    logger.info(f"Ensemble predicted {len(quality_streets)} quality streets")
    logger.info(f"Results saved to: {output_path}")

    return 0


def ensemble(args):
    model_paths = args.models.split(',')

    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        if len(weights) != len(model_paths):
            raise ValueError(
                f"Number of weights ({len(weights)}) doesn't match models ({len(model_paths)})")

    G = load_network(args.input)
    edges = extract_edges(G)

    quality_check = check_data_quality(edges)
    logger.info(
        f"Data quality score: {quality_check['overall_quality_score']:.2f}")

    if args.sparse or quality_check['overall_quality_score'] < 0.7:
        logger.info(
            "Enhancing sparse data with derived features and imputation")
        edges = enrich_sparse_data(edges)

    quality_streets, output_path = apply_ensemble(
        model_paths, args.input, output_dir=args.output, weights=weights
    )

    logger.info(f"Ensemble predicted {len(quality_streets)} quality streets")
    logger.info(f"Results saved to: {output_path}")

    return 0


def evaluate(args):
    from train import load_model
    from evaluate import evaluate_model

    model, metadata = load_model(args.model)

    if args.input.endswith('.csv'):
        data = pd.read_csv(args.input)
    elif args.input.endswith(('.geojson', '.shp')):
        data = gpd.read_file(args.input)
    else:
        raise ValueError(f"Unsupported test data format: {args.input}")

    if 'features' in metadata:
        features = metadata['features']
        logger.info(f"Using {len(features)} features from model metadata")
    else:
        features = [f for f in STANDARD_FEATURES if f in data.columns]
        logger.info(f"Using {len(features)} standard features")

    target_col = metadata.get('target_column', STANDARD_TARGET)

    if target_col not in data.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in test data")

    X_test = data[features]
    y_test = data[target_col]

    os.makedirs(args.output, exist_ok=True)

    metrics = evaluate_model(model, X_test, y_test, output_dir=args.output)

    logger.info(f"Evaluation complete. Results saved to {args.output}")
    logger.info(
        f"Accuracy: {metrics['accuracy']:.4f}, F1 score: {metrics['f1_score']:.4f}")
    logger.info(
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Street quality prediction tool")

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        'input', help='Path to network data or region name')
    train_parser.add_argument('quality', help='Path to quality streets data')
    train_parser.add_argument('region', help='Name of the region')
    train_parser.add_argument(
        '--sparse', action='store_true', help='Enable sparse data handling')
    train_parser.add_argument(
        '--evaluate', action='store_true', help='Evaluate after training')
    train_parser.add_argument(
        '--analyze', action='store_true', help='Analyze feature relationships')
    train_parser.add_argument('--additional-features',
                              nargs='+', help='Additional features to use')

    predict_parser = subparsers.add_parser(
        'predict', help='Predict quality streets')
    predict_parser.add_argument('model', help='Path to trained model')
    predict_parser.add_argument(
        'input', help='Path to network data or region name')
    predict_parser.add_argument('--output', default='quality_streets_output.csv',
                                help='Output path for predictions')
    predict_parser.add_argument(
        '--sparse', action='store_true', help='Enable sparse data handling')

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('model', help='Path to trained model')
    eval_parser.add_argument('input', help='Path to test data')
    eval_parser.add_argument(
        '--output', default='evaluation', help='Output directory for results')

    ensemble_parser = subparsers.add_parser(
        'ensemble', help='Apply an ensemble of models')
    ensemble_parser.add_argument(
        'models', help='Comma-separated paths to models')
    ensemble_parser.add_argument(
        'input', help='Path to network data or region name')
    ensemble_parser.add_argument(
        '--weights', help='Comma-separated model weights')
    ensemble_parser.add_argument(
        '--output', default='.', help='Output directory for results')
    ensemble_parser.add_argument(
        '--sparse', action='store_true', help='Enable sparse data handling')

    auto_parser = subparsers.add_parser('auto-ensemble',
                                        help='Automatically create an ensemble based on region similarity')
    auto_parser.add_argument('input', help='Target region name')
    auto_parser.add_argument('--output', default='.',
                             help='Output directory for results')

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'train':
            return train(args)
        elif args.command == 'predict':
            return predict(args)
        elif args.command == 'evaluate':
            return evaluate(args)
        elif args.command == 'ensemble':
            return ensemble(args)
        elif args.command == 'auto-ensemble':
            return auto_ensemble(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
