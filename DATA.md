# Data provenance and preparation

The empirical datasets analysed in this study are available from their original research resources or repositories. The original empirical data are not redistributed with this code repository.

This document summarizes the provenance, sample definition and representation construction used for the six empirical systems. Full methodological details are provided in the manuscript and Supplementary Information.

For every system, preprocessing produced a symmetric relation matrix A_s for each observation s. Matrices were required to be positive semidefinite within numerical tolerance and were normalized to unit trace,

rho_s = A_s / Tr(A_s).

The resulting operators were held fixed throughout the geometric analysis. Dataset-specific preprocessing and operator construction did not depend on the Schatten order p.

## 1. Structural morphometric MRI — Alzheimer's disease

### Source

Structural MRI data were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI).

### Analysis sample

The system comprised 215 individuals:

- 96 individuals with Alzheimer's disease;
- 119 controls.

Each individual constituted one observational unit.

### Representation construction

Structural MRI preprocessing and construction of the morphometric representation followed the network-based framework described in the references cited in the manuscript and Supplementary Information.

Each individual was represented by 549 morphometric patches. Pairwise relationships between patch-level morphometric profiles were quantified to obtain a subject-specific 549 x 549 morphometric relation matrix.

### Final representation

- Observations: 215
- Relational units: 549 morphometric patches
- Matrix dimension: 549 x 549
- Binary task: Alzheimer's disease versus controls

## 2. Resting-state functional MRI — autism spectrum disorder

### Source

Preprocessed resting-state fMRI data were obtained from the Autism Brain Imaging Data Exchange (ABIDE) through the ABIDE Preprocessed Connectomes Project (ABIDE-PCP).

### Analysis sample

The system comprised 871 individuals:

- 403 individuals with autism spectrum disorder;
- 468 controls.

Each individual constituted one observational unit.

### Preprocessing

The CPAC preprocessing pipeline was used with quality checking and band-pass filtering enabled and without global signal regression.

Regional time series were obtained from the Automated Anatomical Labeling (AAL) parcellation using the `rois_aal` derivative.

### Representation construction

The 116 AAL regions constituted the relational units.

For each individual, functional connectivity was quantified as the Pearson correlation between every pair of regional time series. The resulting 116 x 116 matrix was symmetrized and its diagonal was set to unity.

### Final representation

- Observations: 871
- Relational units: 116 AAL regions
- Matrix dimension: 116 x 116
- Binary task: autism spectrum disorder versus controls

## 3. EEG — Alzheimer's disease

### Source

Resting-state EEG recordings were obtained from the publicly available OpenNeuro dataset `ds004504`, distributed in BIDS format.

### Source population and analysis sample

The source dataset comprises 88 participants:

- 36 with Alzheimer's disease;
- 23 with frontotemporal dementia;
- 29 cognitively normal controls.

For the binary task used in this study, the frontotemporal-dementia group was excluded.

The final system therefore comprised 65 observations:

- 36 Alzheimer's disease;
- 29 controls.

### Preprocessing

Recordings were:

- average-referenced;
- band-pass filtered between 1 and 45 Hz;
- resampled to 250 Hz.

No temporal cropping was applied.

A common set of 19 EEG channels was retained across all observations.

### Representation construction

The 19 channels constituted the relational units.

For each observation, pairwise channel relations were quantified as Pearson correlations between the corresponding preprocessed time series.

The resulting 19 x 19 matrix was symmetrized and its diagonal was set to unity.

### Final representation

- Observations: 65
- Relational units: 19 EEG channels
- Matrix dimension: 19 x 19
- Binary task: Alzheimer's disease versus cognitively normal controls

## 4. Industrial acoustics — machine anomaly detection

### Source

Acoustic recordings were obtained from the publicly available MIMII (Malfunctioning Industrial Machine Investigation and Inspection) dataset.

### Analysis sample

The system used recordings of industrial fans acquired under the -6 dB signal-to-noise condition, spanning machine IDs:

- `id_00`
- `id_02`
- `id_04`
- `id_06`

The final system comprised 5,550 recordings:

- 1,475 anomalous;
- 4,075 normal.

Each 10-s audio clip constituted one observational unit.

### Preprocessing

Audio recordings were sampled at 16 kHz and converted to log-mel power spectrograms using:

- 128 mel-frequency bands;
- FFT size: 1,024 samples;
- hop length: 256 samples.

Each recording yielded 626 temporal frames.

### Representation construction

The 128 mel-frequency bands constituted the relational units.

For each recording, pairwise relations between frequency bands were quantified as Pearson correlations across temporal frames.

The resulting 128 x 128 matrix was symmetrized and its diagonal was set to unity.

### Final representation

- Observations: 5,550 recordings
- Relational units: 128 mel-frequency bands
- Matrix dimension: 128 x 128
- Binary task: anomalous versus normal machine operation

## 5. Traffic sensing — weekday versus weekend

### Source

Traffic data were obtained from the publicly available PEMS-SF dataset, derived from the California Department of Transportation Performance Measurement System (PeMS).

The source contains freeway-lane occupancy measurements from the San Francisco Bay Area collected every 10 min between 1 January 2008 and 30 March 2009.

Each retained day contains 144 within-day measurements from 963 sensors.

Public holidays and two days affected by anomalous sensor outages had already been excluded in construction of the source dataset, yielding 440 daily observations.

### Analysis sample

The original PEMS-SF training and test partitions were combined because the present study used its own fixed cross-validation framework.

The final binary system comprised:

- 313 weekdays;
- 127 weekend days.

Each day constituted one observational unit.

### Unsupervised sensor validity filtering

Sensors were retained only if their within-day time series had finite, non-zero variance in every one of the 440 observations.

This criterion retained 958 of the original 963 sensors.

Neither sensor selection nor representation construction used the day labels.

### Representation construction

For each day, the time series of every retained sensor was standardized across its 144 within-day measurements.

The resulting standardized matrix Z_s has dimension 958 x 144.

The daily relational representation was constructed as

A_s = Z_s Z_s^T / ((T - 1)n),

where T = 144 and n = 958.

This yields a 958 x 958 positive-semidefinite sensor-relation matrix with unit trace.

Because centring the 144-point sensor profiles constrains the rank, the resulting operators have rank at most 143. All 440 operators in the final system attained rank 143.

### Exact low-rank computation

The rank-143 structure permits an exact low-rank factorization. This was used to evaluate the Schatten geometry and perturbation analyses without reducing the 958-sensor representation.

The low-rank implementation is exact rather than a truncated approximation. Numerical comparison with dense computation is documented in the Supplementary Information.

### Binary labels

After representation construction had been fixed, the original seven day-of-week labels were collapsed as follows:

- labels 1-5: weekday;
- labels 6-7: weekend.

### Final representation

- Observations: 440 days
- Relational units: 958 traffic sensors
- Matrix dimension: 958 x 958
- Exact operator rank: 143
- Binary task: weekend versus weekday

## 6. Single-cell transcriptomics — systemic lupus erythematosus

### Source

Single-cell transcriptomic data were obtained from the public peripheral-blood mononuclear cell atlas of systemic lupus erythematosus reported by Perez et al. (2022).

The processed dataset used in this study corresponds to the publicly available CELLxGENE representation:

- Dataset: `218acb0f-9f2f-4f76-b90b-15a4b7c7f629`
- Version: `c55dc602-d168-4d15-acc1-5de4f2f5d551`

The source contains 1,263,676 cells from 261 donors.

### Analysis sample

The binary system comprised:

- 162 donors with systemic lupus erythematosus;
- 99 healthy controls.

Each donor constituted one observational unit.

### Preprocessing

Raw counts were obtained from the `adata.raw` layer.

Counts were normalized independently for each cell to a total library size of 10^4 and transformed using log(1+x).

Gene selection was performed without using disease labels.

Genes were ranked by donor-balanced variance in an unsupervised audit. The top 336 genes were retained for the final representation.

The dimensionality was selected through an additional unsupervised numerical and stability audit, as described in the Supplementary Information.

### Representation construction

For each donor, the selected genes constituted the relational units.

Pairwise gene relations were quantified as Pearson correlations across the donor's cells.

The resulting 336 x 336 matrix was symmetrized and its diagonal was set to unity.

Donors containing constant or numerically invalid selected genes were excluded by construction. Matrices failing the predefined positive-semidefinite tolerance of 10^-8 were likewise rejected.

### Final representation

- Observations: 261 donors
- Relational units: 336 genes
- Matrix dimension: 336 x 336
- Binary task: systemic lupus erythematosus versus healthy controls

## Data redistribution

The empirical datasets are not bundled with this repository.

Users should obtain the source data from the original repositories and comply with the corresponding access conditions and licences.

The synthetic toy dataset generated by `examples/toy_example/generate_toy_data.py` is intended solely to test installation and demonstrate the computational workflow. It is not derived from any of the empirical datasets and is not part of the scientific analyses reported in the manuscript.
