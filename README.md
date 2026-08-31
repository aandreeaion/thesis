# Master's Thesis: Argumentation Strategies in Political News Discourse

This repository contains the code, notebooks, figures, and selected results for my Master's thesis in Digital Humanities at the University of Groningen.

The project investigates argumentative units in politically oriented news discourse using transformer-based token classification. A ModernBERT model is trained and evaluated on the Webis-Editorial-16 corpus and subsequently applied to additional political news corpora for corpus-scale analysis.

The computational workflow combines corpus preprocessing, model training and hyperparameter optimisation, model evaluation, calibration, manual error analysis, large-scale prediction, descriptive analysis, statistical testing, and sequential analysis of argumentative-unit patterns.

---

## Research overview

The project is organised around two connected objectives:

1. developing and evaluating a transformer-based classifier for argumentative units in news discourse; and
2. using the resulting predictions to investigate how argumentative units vary across political orientation, topic, source, and local discourse sequence.

The model is developed using Webis-Editorial-16 and evaluated both in-domain and on a manually annotated 45-article AllSides validation sample.

The trained model is then used to obtain corpus-scale predictions for the substantive analyses conducted on the AllSides same-events corpus and NLPCSS20.

---

## Repository structure

```text
thesis/
│
├── figures/
│   └── Figures generated for the corpus analyses and thesis
│
├── jobs/
│   └── SLURM job scripts used on the Hábrók HPC cluster
│
├── notebooks/
│   ├── Corpus preprocessing
│   ├── Corpus description
│   ├── Descriptive analysis
│   ├── Sequence analysis
│   └── Error analysis
│
├── results/
│   ├── Model evaluation results
│   ├── Error-analysis summaries
│   ├── Corpus descriptive statistics
│   ├── Statistical tests
│   └── Selected analysis outputs
│
├── scripts/
│   ├── Data preprocessing
│   ├── Model training
│   ├── Hyperparameter optimisation
│   ├── Calibration
│   ├── Evaluation
│   ├── Prediction
│   └── Error analysis
│
├── .gitignore
└── README.md
```

Large datasets, trained models, checkpoints, corpus-wide prediction files, virtual environments, caches, and cluster logs are intentionally excluded from Git.

---

# Requirements

## Software

To run the project, the following software is required:

- Git
- Python 3
- `pip`
- Python virtual environments (`venv`)
- Jupyter Notebook or JupyterLab

GPU access is strongly recommended for training and corpus-scale ModernBERT inference.

The original experiments were run primarily on the University of Groningen **Hábrók HPC cluster** using SLURM.

The descriptive and statistical notebooks can generally be run on a normal computer once their input data and prediction files are available.

---

## Python environment

A Python virtual environment should be used rather than installing the packages globally.

After cloning the repository, create a virtual environment from the project root:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

---

## Python dependencies

The project uses packages from the Python NLP, machine-learning, data-analysis, and scientific-computing ecosystem.

Important dependencies include packages such as:

```text
torch
transformers
tokenizers
optuna
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
jupyter
notebook
ipykernel
```

Additional packages may be required by individual analysis notebooks.

If a `requirements.txt` file is included in the repository, install the environment with:

```bash
pip install -r requirements.txt
```

If reproducing the project from the original Hábrók environment, the package versions used in that environment should be preserved where possible.

To record the packages from an existing working environment:

```bash
pip freeze > requirements.txt
```

The exact Python and package versions are particularly important for `torch`, `transformers`, and GPU/CUDA compatibility.

---

# Jupyter setup

After activating the virtual environment, install Jupyter if necessary:

```bash
pip install jupyter ipykernel
```

Register the project's virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name thesis --display-name "Python (thesis)"
```

Start JupyterLab with:

```bash
jupyter lab
```

or classic Jupyter Notebook with:

```bash
jupyter notebook
```

When opening one of the notebooks, select:

```text
Python (thesis)
```

as the kernel.

When working through VS Code, the same `.venv` environment can be selected through the notebook kernel/interpreter selector.

---

# Data

## Important

The datasets used for the thesis are **not stored in this Git repository**.

They are excluded because some files are large and some originate from external datasets for which redistribution through this repository may not be appropriate.

The local project therefore contains a `data/` directory that is ignored by Git.

The research uses data derived from or associated with:

- Webis-Editorial-16;
- the manually annotated 45-article AllSides validation sample;
- the AllSides same-events corpus; and
- NLPCSS20.

Anyone attempting full reproduction must obtain the relevant original datasets and prepare them using the preprocessing code provided in this repository.

---

## Local data directory

The expected local project structure includes a data directory:

```text
thesis/
├── data/
├── figures/
├── jobs/
├── notebooks/
├── outputs/
├── results/
└── scripts/
```

`data/` exists in the working version of the project but is ignored by Git.

Some scripts and notebooks use paths based on the original Hábrók directory structure. When running the repository on another machine, these paths may therefore need to be changed to point to the local copies of the datasets.

---

# Large files excluded from Git

The following directories are deliberately excluded from normal Git version control:

```text
data/
outputs/
results/model_predictions/
.venv/
```

## `data/`

Contains source and processed corpora used by the project.

## `outputs/`

Contains large outputs produced during model development, including training runs, checkpoints, and related generated files.

The local working directory contained approximately 20 GB of model outputs at the time the repository was created.

## `results/model_predictions/`

Contains corpus-level model prediction files used as input for subsequent substantive analyses.

These files are retained separately because the directory is approximately 1.4 GB and is not suitable for ordinary Git version control.

## `.venv/`

Contains the local Python environment. Virtual environments should always be reconstructed rather than committed to Git.

Large model predictions and model outputs have **not been deleted**. They are simply stored separately from the GitHub repository and can be provided separately if required for examination or verification.

---

# Computational workflow

A simplified version of the complete workflow is:

```text
Webis-Editorial-16
        │
        ▼
Corpus reconstruction
        │
        ▼
BIO / ModernBERT tokenisation
        │
        ▼
Hyperparameter optimisation
        │
        ▼
Final ModernBERT training
        │
        ▼
In-domain evaluation
        │
        ├──────────────► Calibration / threshold analysis
        │
        ▼
External AllSides-45 evaluation
        │
        ▼
Manual error analysis
        │
        ▼
Corpus-scale prediction
        │
        ├──────────────► AllSides same-events
        │
        └──────────────► NLPCSS20
                              │
                              ▼
                 Descriptive analysis
                              │
                              ▼
                   Statistical analysis
                              │
                              ▼
                    Sequence analysis
```

---

# 1. Corpus preprocessing

## Webis-Editorial-16 reconstruction

The Webis corpus reconstruction is documented in:

```text
notebooks/reconstruction_Webis16_v2.ipynb
```

The reconstructed data are subsequently converted into ModernBERT token-classification format using:

```text
scripts/webis_to_modernBERT_BIO.py
```

This stage creates the labelled model input used for model development.

---

## AllSides 45-article validation sample

The manually annotated validation sample is documented in:

```text
notebooks/validation_sample.ipynb
```

Annotation preprocessing is handled by:

```text
scripts/prep_annotations.py
```

The annotations are subsequently converted into the token-classification representation used for ModernBERT evaluation with:

```text
scripts/allsides_45_to_modernBERT_BIO.py
```

The validation corpus is used for out-of-domain evaluation and manual error analysis.

---

## NLPCSS20 preprocessing

Preparation and filtering of NLPCSS20 are documented in:

```text
notebooks/prep_nlpcss20.ipynb
notebooks/nlpcss20_predicted_span_filtering.ipynb
```

Corpus description is provided through:

```text
notebooks/nlpcss20_corpus_description.ipynb
```

---

# 2. Model training

The repository contains several scripts reflecting different stages of model experimentation and final training.

Important training scripts include:

```text
scripts/finetune_modernBERT.py
scripts/train_final_modernBERT.py
scripts/train_final_modernBERT_merged.py
scripts/train_final_modernBERT_merged_optuna.py
scripts/train_final_modernBERT_strict.py
```

The repository retains some intermediate experimental scripts in addition to the final training configuration so that the model-development process is documented.

---

# 3. Hyperparameter optimisation

Optuna is used for hyperparameter optimisation.

Relevant scripts include:

```text
scripts/optuna_modernBERT.py
scripts/optuna_modernBERT_relaxed.py
scripts/optuna_modernBERT_relaxed_merged.py
scripts/optuna_modernBERT_relaxed_256.py
scripts/optuna_modernBERT_relaxed_512.py
scripts/optuna_modernBERT_relaxed_1024.py
scripts/optuna_largerBatchSize.py
scripts/optuna_modernBERT_relaxed_merged_larger_batch.py
```

Several of these files correspond to experiments with alternative maximum sequence lengths, evaluation procedures, or batch-size settings.

They are retained to document the development process and should not all be interpreted as separate final models.

---

# 4. Model evaluation

Evaluation code includes:

```text
scripts/evaluate_modelB_per_label.py
scripts/evaluate_merged_per_label.py
scripts/evaluate_out_of_domain.py
scripts/relaxed_F1.py
```

The project distinguishes between strict and relaxed approaches to span-level evaluation where relevant.

Selected evaluation outputs are stored in `results/`, including files such as:

```text
results/webis_test_per_label_results_merged.csv
results/webis_test_per_label_results_merged.json
results/allsides_45_out_of_domain_results.json
results/allsides_45_out_of_domain_per_label_results.csv
```

---

# 5. Calibration

Model calibration and decision-threshold analyses are implemented through:

```text
scripts/temperature_calibration.py
scripts/temperature_calibration_strict.py
scripts/calibration_threshold_f1.py
scripts/calibration_treshold_f1_merged.py
```

Corresponding Hábrók submission scripts are stored in `jobs/`.

---

# 6. External validation and error analysis

The external evaluation uses the manually annotated 45-article AllSides sample.

Relevant scripts include:

```text
scripts/evaluate_out_of_domain.py
scripts/export_allsides_45_error_analysis.py
scripts/error_sample.py
scripts/manual_inspection_ea.py
scripts/sanity_check_45.py
```

The manual error-analysis notebooks include:

```text
notebooks/allsides_45_manual_error_analysis_summary.ipynb
notebooks/ea_summary.ipynb
```

Selected error-analysis results are stored under:

```text
results/error_analysis_summary/
results/error_analysis_summary_final/
```

The repository also contains the manually coded and exported error-analysis tables used during this stage.

---

# 7. Corpus-scale prediction

After model development and evaluation, the trained model is applied to the substantive corpora.

## AllSides same-events

Prediction script:

```text
scripts/predict_allsides_same_event.py
```

Hábrók submission script:

```text
jobs/run_predict_allsides_same_events.sh
```

## NLPCSS20

Prediction script:

```text
scripts/predict_nlpcss-20.py
```

Hábrók submission script:

```text
jobs/run_predict_nlpcss20.sh
```

The complete corpus-level prediction files are stored locally in:

```text
results/model_predictions/
```

and are not included in the Git repository because of their size.

---

# 8. AllSides same-events analysis

Important notebooks include:

```text
notebooks/allsides_same_events_descriptive_statistics.ipynb
notebooks/allsides_same_events_final_analysis.ipynb
notebooks/allsides_same_events_sequence_analysis.ipynb
```

These notebooks are used for corpus description and substantive comparisons involving variables such as political orientation, argumentative-unit composition, span density, topic, outlet, and sequential organisation.

Selected summary outputs are available in:

```text
results/allsides_same_events_descriptive_stats/
results/allsides_same_events_sequence_statistical_tests.csv
results/same_event_composition_statistical_tests.csv
results/same_event_density_statistical_test.csv
```

---

# 9. NLPCSS20 analysis

Important notebooks include:

```text
notebooks/nlpcss20_corpus_description.ipynb
notebooks/nlpcss20_descriptive_analysis_final.ipynb
notebooks/nlpcss20_sequence_analysis.ipynb
```

Additional notebooks document earlier stages of the analysis and are retained for transparency:

```text
notebooks/nlpcss20_descriptive_analysis.ipynb
notebooks/nlpcss20_descriptive_statistics.ipynb
```

Selected statistical outputs are stored under:

```text
results/analysis/nlpcss20_descriptive_analysis/
results/nlpcss20_sequence_statistical_tests.csv
```

---

# Figures

The `figures/` directory contains visualisations generated for the thesis analyses.

These include figures related to:

- political orientation;
- argumentative-unit composition;
- argumentative-unit density;
- topic;
- news source/outlet;
- article length;
- transition matrices; and
- sequence patterns.

Most thesis figures are stored as PDF files, with PNG versions retained for selected visualisations.

---

# Results directory

Unlike the excluded large model outputs, the tracked `results/` directory contains relatively small and interpretable outputs that support the results reported in the thesis.

These include:

```text
Model evaluation metrics
Per-label evaluation results
Out-of-domain evaluation results
Manual error-analysis summaries
Corpus descriptive statistics
Statistical test results
Sequence-analysis results
```

These files make it possible to inspect many of the reported quantitative results without rerunning model training.

---

# Running the project on Hábrók

The original computational experiments were performed using the University of Groningen Hábrók HPC cluster.

After logging into Hábrók and activating the Python environment:

```bash
source .venv/bin/activate
```

jobs can be submitted through SLURM, for example:

```bash
sbatch jobs/run_train_final_modernbert_merged.sh
```

or:

```bash
sbatch jobs/run_predict_nlpcss20.sh
```

The `jobs/` directory contains submission scripts for:

- Optuna experiments;
- final model training;
- strict and relaxed evaluation;
- calibration;
- external evaluation; and
- corpus-scale prediction.

SLURM configuration, GPU availability, partitions, memory limits, and module names are specific to the HPC environment and may need to be changed when running the project on another cluster.

---

# Running without Hábrók

Hábrók is not required for inspecting the repository or running most analysis notebooks.

After cloning the repository, creating the virtual environment, installing the dependencies, and obtaining the necessary data, start Jupyter:

```bash
jupyter lab
```

Analysis notebooks can then be executed interactively.

Model training and corpus-wide ModernBERT inference can technically be run outside Hábrók, but a CUDA-capable GPU is strongly recommended.

The SLURM `.sh` files are specific to an HPC environment. On a local machine, the corresponding Python script can instead be executed directly according to the configuration contained in the relevant job file.

---

# Cloning the repository

Clone the repository with:

```bash
git clone https://github.com/aandreeaion/thesis.git
```

Enter the project:

```bash
cd thesis
```

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then register the Jupyter kernel:

```bash
python -m ipykernel install --user --name thesis --display-name "Python (thesis)"
```

and launch:

```bash
jupyter lab
```

---

# Reproducing the complete analysis

Full end-to-end reproduction requires more than cloning this repository because the underlying corpora and large generated model files are not distributed through normal Git.

A complete reproduction broadly requires:

1. obtaining the original corpora;
2. placing them in the appropriate local `data/` structure;
3. reconstructing and preprocessing Webis-Editorial-16;
4. producing the ModernBERT token-classification data;
5. reproducing the model-development experiments;
6. training the final model;
7. performing calibration and evaluation;
8. preparing the manually annotated AllSides validation corpus;
9. running the external evaluation;
10. generating predictions for the AllSides same-events and NLPCSS20 corpora;
11. running the descriptive and statistical analysis notebooks; and
12. running the sequence-analysis notebooks.

For inspection of the thesis results, complete retraining is not necessary because selected final evaluation and statistical outputs are included in the repository.

---

# Reproducibility and file availability

This GitHub repository is intended to preserve the computational component of the thesis while avoiding the use of Git for very large generated files or externally sourced datasets.

The following are retained separately:

```text
Raw datasets
Processed datasets
Model checkpoints
Training outputs
Full corpus-level predictions
Other large intermediate files
```

These files are not missing from the original research environment; they are excluded from the Git repository because of file size and/or redistribution considerations.

Large model prediction files and other retained research artefacts can be provided separately if required for thesis examination or verification.

---

# Notes on intermediate files

The repository was created from the working research project at the end of the thesis process.

As a result, some scripts and notebooks corresponding to intermediate experiments are retained alongside the final versions. These files document the development and checking process but are not necessarily required to reproduce the final reported results.

Files containing names such as alternative sequence lengths, strict/relaxed variants, or alternative optimisation settings generally correspond to experiments performed during model development.

The notebooks and scripts identified above as the main analysis files are the most relevant starting points for understanding the final workflow.

---

# Citation

If using material from this repository, please cite the associated Master's thesis.

**Author:** Andreea Gabriela Ion  
**Programme:** MA Digital Humanities  
**Institution:** University of Groningen  
**Year:** 2026

The final thesis title and institutional repository link can be added here once the thesis has been deposited.

---

# Author

**Andreea Gabriela Ion**  
MA Digital Humanities  
University of Groningen
