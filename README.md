# Mechanistic Evidence Pipeline using Translator Component Toolkit (TCT)

## Overview
This project provides a modular framework to analyze biological evidence-supported paths extracted from the Translator knowledge base. The goal is to validate mechanistic hypotheses by tracing the relationship between specific ligands, targets, and biological outcomes, while ensuring tissue specificity and evidence provenance.

The pipeline is designed to be executed in a Jupyter Notebook, leveraging a set of backend Python modules to prevent notebook bloat and ensure reproducibility.

## Architecture & Code Logic

The project is structured into a modular library to separate data ingestion, biological analysis, and visualization.

### 1. Modular Components
- **`data_loader.py`**: Handles the ingestion of raw exports from the Translator UI/CI instance. It includes preprocessing steps to standardize data formats and filter results by specific tissue types to eliminate non-relevant biological noise.
- **`path_analyzer.py`**: The core analytical engine. It processes paths to verify directionality (e.g., activation vs. inhibition) and maps each edge of the path to its original publication source.
- **`hypothesis_tester.py`**: A specialized module for comparative analysis. It allows the user to contrast different ligands or identify contradictory evidence (e.g., comparing pro-regenerative vs. pro-fibrotic signals).
- **`visualizer.py`**: Transforms processed data into publication-ready formats, including mechanistic flowcharts and evidence summary tables.

### 2. Workflow Logic
The execution follows a linear pipeline:
`Raw UI Export` $\rightarrow$ `Tissue Filtering` $\rightarrow$ `Path Validation (Directionality)` $\rightarrow$ `Comparative Analysis` $\rightarrow$ `Visualization`.

## Input and Output

### Inputs
- **Data Source**: JSON or CSV files exported from the Translator CI instance.
- **Parameters**: Tissue filters and target node identifiers defined within the orchestration notebook.

### Outputs
- **Evidence Tables**: Detailed lists of paths including the direction of effect and the associated publication provenance.
- **Mechanistic Graphs**: Visual representations of the biological flow from ligand to outcome.
- **Comparative Reports**: Dataframes highlighting the differences between specific ligands or the presence of contradictory evidence.

## How to Reproduce

### Prerequisites
- Python 3.x
- Translator Component Toolkit (TCT) installed in the environment.
- Access to the Translator CI instance for initial query exports.

### Execution Steps
1. **Data Extraction**: Perform the required Pathfinder queries in the Translator UI and export the results as files.
2. **Environment Setup**: Ensure all `.py` modules are in the same directory as the notebook.
3. **Notebook Orchestration**:
    - Run the `data_loader` to import your exports.
    - Apply the `path_analyzer` to validate the biological paths.
    - Use `hypothesis_tester` to challenge the primary hypothesis.
    - Generate final figures using `visualizer`.

## Maintenance
To add new biological filters or analysis metrics, modify the corresponding module in the `/modules` directory. The notebook should only be used for parameter definition and result visualization.
