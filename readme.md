# Street Quality Prediction

## Core Purpose

This system predicts high-quality streets from OpenStreetMap network data with specific focus on regions where OSM attribute coverage is incomplete. It addresses a common geospatial analysis limitation where traditional quality assessment methods fail due to missing street attributes.

## Technical Approach

The implementation employs multiple strategies to overcome data sparsity challenges:

1. **Network-Derived Features**: Extracts graph-theoretical properties (centrality metrics, connectivity) that remain available even when tag data is sparse
2. **Spatial Imputation**: Applies KNN imputation with distance weighting to approximate missing attributes based on spatial relationships
3. **Cross-Region Transfer**: Implements ensemble methods to transfer knowledge from data-rich regions to data-poor regions
4. **Confidence Quantification**: Provides prediction confidence metrics correlated with input data completeness

## Requirements

- Python 3.8+
- NetworkX, GeoPandas, scikit-learn
- uv package manager (for environment setup)

## Installation

```bash
# Clone repository
git clone https://github.com/arthurlch/osm-route-quality-prediction
cd osm-route-quality-prediction

# Initialize environment
chmod +x start.sh
./start.sh

# Activate environment
source .venv/bin/activate
```

## Usage Reference

### Training

Train a model on regions with known quality streets:

```bash
python cli.py train [network_path] [quality_streets_path] [region_name] [options]

# Required arguments:
#   network_path         Path to OSM network data (.graphml, .osm)
#   quality_streets_path Path to known quality streets (.csv, .geojson, .shp)
#   region_name          Name identifier for the region

# Options:
#   --sparse             Enable sparse data handling pipeline
#   --evaluate           Run evaluation after training
#   --analyze            Generate feature analysis visualizations
```

### Prediction

Apply trained models to new regions:

```bash
python cli.py predict [model_path] [network_path] [options]

# Required arguments:
#   model_path           Path to trained model (.joblib)
#   network_path         Path to OSM network data

# Options: 
#   --output PATH        Output file path (default: quality_streets_output.csv)
#   --sparse             Enable sparse data handling
```

### Ensemble Methods

Apply multiple models as an ensemble:

```bash
python cli.py ensemble [model_paths] [network_path] [options]

# Required arguments:
#   model_paths          Comma-separated paths to models
#   network_path         Path to OSM network data

# Options:
#   --weights VALUES     Comma-separated model weights
#   --output DIR         Output directory 
#   --sparse             Enable sparse data handling
```

For regions with minimal data, use the auto-ensemble function:

```bash
python cli.py auto-ensemble [region_name]

# Required:
#   region_name          Target region name

# This automatically:
# 1. Identifies similar regions with trained models
# 2. Weights models by region similarity
# 3. Creates an optimized ensemble for the target region
```

### Evaluation

Evaluate model performance against test data:

```bash
python cli.py evaluate [model_path] [test_data_path] [options]

# Required:
#   model_path           Path to trained model (.joblib)
#   test_data_path       Path to test data (.csv, .geojson, .shp)

# Options:
#   --output DIR         Output directory for metrics (default: evaluation/)
```

## Input Data Specifications

### Network Data
- GraphML files from OSMnx exports
- Raw OSM XML/PBF files
- NetworkX serialized graphs

### Quality Streets Data
Required columns:
- Unique identifiers matching network edges
- Target column (`is_quality` by default) with binary values

## Sparse Data Processing Pipeline

When the `--sparse` flag is enabled, the system applies these enhancements:

1. **Topology Analysis**: Calculates sinuosity, bearing, segment length
2. **Network Analysis**: Computes centrality, clustering, connectivity metrics
3. **KNN Imputation**: Approximates missing values based on spatial proximity
4. **Confidence Scoring**: Adds data completeness metrics to predictions

## Project Structure

```
street_quality_prediction/
├── utils.py           # Core utilities and data processing
├── train.py           # Model training functionality
├── apply.py           # Prediction and ensemble methods
├── evaluate.py        # Evaluation metrics and visualization
├── cli.py             # Command-line interface
├── start.sh           # Environment setup script
├── requirements.txt   # Dependencies
└── models/            # Saved model storage
```

## Practical Applications

- **Urban Planning**: Identify high-quality corridors for infrastructure investment
- **Navigation**: Generate cycling or pedestrian-friendly routing preferences
- **Research**: Analyze street network quality patterns across regions

## Technical Limitations

- Prediction accuracy correlates with training data quality
- Derived features cannot fully compensate for critical missing attributes
- Spatial imputation assumes spatial autocorrelation of attributes
- Transfer learning effectiveness depends on regional similarity

## Future Development

The system architecture supports these planned enhancements:

- Integration with satellite imagery analysis for additional feature extraction (non-prio)
- API endpoints for service-oriented deployment 
- Hyperparameter optimization for model tuning (prio)
- Web interface for non-technical users
