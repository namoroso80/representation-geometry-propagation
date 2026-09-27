# Propagation dynamics of representation geometry through learning, decisions and explanations

Code accompanying the manuscript:

**Propagation dynamics of representation geometry through learning, decisions and explanations**

Nicola Amoroso

## Overview

This repository contains the computational framework used to study how controlled changes in representation geometry propagate through successive layers of a learning system.

Empirical observations are represented as positive-semidefinite, unit-trace operators. Geometry is continuously deformed through the Schatten family

$d_p(rho_i,rho_j) = ||rho_i-rho_j||_p$

and its propagation is followed through

$rho -> D_p -> K_p -> f_p -> XAI_p$,

from operator geometry to pairwise distances, kernel representations, fitted decisions and perturbation-based explanations.

The repository contains the code used for the controlled simulations, the six empirical systems, nested classification analyses, decision-space analyses, geometry-conditioned perturbation explanations, spectral analyses and post-hoc propagation analyses reported in the manuscript.

## Empirical systems

The study considers six empirical systems spanning different scientific domains:

- structural morphometric MRI;
- resting-state functional MRI;
- EEG connectivity;
- industrial acoustic monitoring;
- traffic sensing;
- single-cell transcriptomics.

All empirical datasets analysed in this study are available from their original research resources or repositories. The original empirical data are not redistributed with this repository.

Dataset provenance, access information and dataset-specific preprocessing are documented in `DATA.md` and in the Supplementary Information accompanying the manuscript.

## Repository structure

    .
    ├── geometry_pipeline_v2_1/  # Core Schatten geometry and learning pipeline
    ├── adapters/                # Dataset-specific adapters
    ├── simulations/             # Controlled geometry-deformation simulations
    ├── representation_audits/   # Representation and numerical audits
    ├── posthoc_analysis/        # Propagation and support-vector analyses
    ├── examples/
    │   └── toy_example/         # Synthetic end-to-end smoke test
    ├── DATA.md
    ├── environment.yml
    ├── LICENSE
    └── README.md

## Core pipeline

The analyses reported in the manuscript use the Schatten grid

p = {1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 16}.

For each geometry, the analysis comprises:

1. construction or loading of the empirical operator representation;
2. computation of pairwise Schatten distances;
3. spectral characterization of operator differences;
4. nested cross-validated RBF-SVM classification;
5. analysis of held-out decision-function reorganization;
6. geometry-conditioned perturbation explanations;
7. comparison of geometry, kernel, support-vector, dual-weight and contribution-field reorganization.

Dataset-specific model selection and preprocessing follow the procedures described in the manuscript and Supplementary Information.

## Reproducibility

The empirical analyses can be reproduced by:

1. obtaining the corresponding datasets from their original sources;
2. applying the dataset-specific preprocessing and representation procedures;
3. running the supplied adapters and configuration files;
4. executing the geometry pipeline and corresponding post-hoc analyses.

The repository does not contain the original empirical datasets or duplicate copies of source data.

Some analyses, particularly the complete industrial-acoustic and traffic-sensing experiments, are computationally intensive and may require substantial memory, storage and computation time.

## Quick start with the synthetic example

A small synthetic dataset is included as a generator to verify installation and exercise the complete computational workflow.

From the repository root:

    python examples/toy_example/generate_toy_data.py

    python geometry_pipeline_v2_1/run_pipeline.py \
        --config examples/toy_example/toy_config.yaml

The example runs the classifier, spectral, decision, XAI and summary stages and writes the resulting files to:

    examples/toy_example/output/

The toy dataset is provided solely for software validation and demonstration. It is not part of the empirical study and is not intended to reproduce the numerical results reported in the manuscript. The toy example is intentionally small and uses a reduced Schatten grid and cross-validation schedule to provide a fast end-to-end smoke test of the full pipeline.

## Main analysis components

### Controlled simulations

`simulations/` contains the rank-controlled simulations and associated audits used to characterize the direct effect of Schatten deformation independently of the empirical systems.

### Empirical geometry pipeline

`geometry_pipeline_v2_1/` contains the common computational framework used for classification, spectral, decision-space and perturbation-XAI analyses.

Dataset-specific configurations are provided in `geometry_pipeline_v2_1/configs/`.

### Dataset adapters

`adapters/` contains the dataset-specific adapters required for the empirical representations that use dedicated preparation procedures.

### Representation audits

`representation_audits/` contains dataset-specific procedures used to validate representation construction, dimensionality choices and the exact low-rank implementation used for the traffic-sensing system.

### Propagation analyses

`posthoc_analysis/` contains the analyses used to follow geometric deformation through kernel organization, support-vector membership, dual coefficients and support-vector contribution fields, together with subject-resampling uncertainty analyses.

## Numerical implementation

For the traffic-sensing system, the 958-dimensional operators possess an exact low-rank representation. The supplied low-rank backend exploits this structure without reducing the sensor representation or approximating its Schatten geometry.

Dense-versus-low-rank numerical validation is documented in the Supplementary Information and corresponding scripts in `representation_audits/`.

## Software environment

The analyses were implemented in Python.

The archived environment reflects the software versions used for the final analyses. Some dataset-specific preprocessing steps require only a subset of these dependencies.

A Conda environment can be created with:

    conda env create -f environment.yml
    conda activate representation-geometry-propagation

## Code availability

The version of the code corresponding to the manuscript will be permanently archived on Zenodo.

- GitHub: [to be added]
- Zenodo DOI: [to be added]

## Citation

If you use this code, please cite:

Amoroso, N. *Propagation dynamics of representation geometry through learning, decisions and explanations*. [citation to be updated]

## License

The source code in this repository is released under the MIT License. See `LICENSE` for details.

The empirical datasets are not distributed with this repository and remain subject to the terms and conditions of their respective source repositories.
