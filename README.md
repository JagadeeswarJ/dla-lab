# Deep Learning Applications (DLA) Lab Repository

A complete, self-contained repository for the **Deep Learning Applications (DLA) Laboratory** curriculum (VNR VJIET / JNTUH / Academic Semester 7).

This repository is pre-configured to run out-of-the-box on any clean, newly installed system (Windows, Linux, or macOS) with **Python 3.12**.

> [!IMPORTANT]
> **Datasets are ignored in git via `.gitignore`** to keep the git repository lightweight.
> When cloning this repo onto a new computer, run **`python setup_datasets.py`** once. This single command will automatically download and unpack all 5 lab datasets directly into the codebase.

---

## Table of Contents

1. [Quick Start (3 Steps for Exam)](#quick-start-3-steps-for-exam)
2. [Prerequisites & Python 3.12 Installation](#prerequisites--python-312-installation)
3. [Environment Setup & Package Installation](#environment-setup--package-installation)
4. [Getting the Datasets](#getting-the-datasets)
   - [Method 1: One-Command Automatic Setup (Recommended)](#method-1-one-command-automatic-setup-recommended)
   - [Method 2: Manual Dataset Downloads & Links](#method-2-manual-dataset-downloads--links)
5. [How to Run the Code (Step-by-Step)](#how-to-run-the-code-step-by-step)
   - [Running with VS Code](#running-with-vs-code-recommended)
   - [Running with Jupyter Notebook](#running-with-jupyter-notebook)
   - [Running as Headless Python Scripts](#running-as-headless-python-scripts)
6. [Lab Experiments Breakdown](#lab-experiments-breakdown)
   - [Notebook 1: DLA_W_1_to_4.ipynb](#notebook-1-dla_w_1_to_4ipynb)
   - [Notebook 2: DLA_W_6_to_8.ipynb](#notebook-2-dla_w_6_to_8ipynb)
7. [Dependencies & requirement.txt](#dependencies--requirementtxt)
8. [Exam Tips & Troubleshooting](#exam-tips--troubleshooting)

---

## Quick Start (3 Steps for Exam)

If you are on a new computer during an exam, run these 3 commands in PowerShell or Command Prompt inside the `dla-lab` folder:

```bash
# 1. (Optional but recommended) Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows CMD use: venv\Scripts\activate.bat

# 2. Install all required Python packages at once
pip install -r requirement.txt

# 3. Download and prepare all 5 lab datasets in one go
python setup_datasets.py
```

Now open either notebook in VS Code or Jupyter and click **"Run All"**!

---

## Prerequisites & Python 3.12 Installation

### Step 1: Install Python 3.12 via winget (PowerShell)

If Python 3.12 is not already installed on your system, install it directly using Windows Package Manager (`winget`) in PowerShell:

```powershell
winget install -e --id Python.Python.3.12
```

> [!TIP]
> After the installation completes, restart PowerShell (or refresh your environment variables in the current session with `$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`) so that `python` and `pip` are accessible from your PATH.

### Step 2: Verify Python and pip

Verify the installation by running:

```powershell
python --version
pip --version
```

*Expected output:*
```text
Python 3.12.x
pip 24.x.x (or newer)
```

---

## Environment Setup & Package Installation

### Step 1: Open Terminal in the Codebase Directory

Open **PowerShell** or **Command Prompt** (or terminal in VS Code) and navigate to the project directory:

```bash
cd path\to\dla-lab
```

*(On Windows, you can also open the folder in File Explorer, type `cmd` or `powershell` in the address bar, and press Enter).*

---

### Step 2: Create & Activate a Virtual Environment

Creating an isolated virtual environment ensures clean, reproducible package versions:

- **Create the environment:**
  ```bash
  python -m venv venv
  ```

- **Activate the environment:**
  - **Windows PowerShell:**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
    *(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then re-run).*
  - **Windows Command Prompt (cmd):**
    ```cmd
    venv\Scripts\activate.bat
    ```
  - **Linux / macOS:**
    ```bash
    source venv/bin/activate
    ```

When active, your prompt will show `(venv)`.

---

### Step 3: Install All Dependencies at Once

Install all required packages with a single command:

```bash
pip install -r requirement.txt
```

*(You can also use `pip install -r requirements.txt`).*

This installs:
- **`tensorflow`** (Deep Learning engine & Keras models: Conv1D, Conv2D, LSTM)
- **`opencv-python`** (Image preprocessing & color conversions)
- **`scikit-learn`** (ML models, preprocessing, metrics & train-test split)
- **`pandas`** & **`numpy`** (Data manipulation and arrays)
- **`matplotlib`** & **`seaborn`** (Plotting and visualizations)
- **`pretty_midi`** (MIDI note sequence extraction)
- **`ipykernel`** & **`notebook`** (Jupyter notebook environment)

---

## Getting the Datasets

Because dataset archives are large (hundreds of MBs), they are ignored by git via `.gitignore`. You can obtain all required datasets using either of the two methods below:

### Method 1: One-Command Automatic Setup (Recommended)

Run the included verification and download script:

```bash
python setup_datasets.py
```

This script checks for existing files and automatically downloads and extracts only what is missing:
1. Clones the Udacity Self-Driving Car dataset into `data/`.
2. Downloads and extracts the UCI HAR dataset into `har_data/uci-har-dataset-master/`.
3. Downloads and extracts the Quora Question Pairs dataset as `train.csv`.
4. Clones the Nottingham folk tunes dataset into `nottingham-dataset/`.
5. Downloads and extracts the MAESTRO MIDI dataset into `maestro/`.

*Expected output:*
```text
============================================================
DLA LAB DATASET VERIFICATION & SETUP
============================================================
[DATASET SETUP] Self-driving car dataset ('data/') is present.
[DATASET SETUP] UCI HAR dataset ('har_data/uci-har-dataset-master') is present.
[DATASET SETUP] Quora Question Pairs dataset ('train.csv') is present.
[DATASET SETUP] Nottingham ABC folk music dataset ('nottingham-dataset/') is present.
[DATASET SETUP] MAESTRO MIDI dataset ('maestro/') is present.
============================================================
[DATASET SETUP] ALL DATASETS ARE FULLY READY FOR EXAM EXECUTION!
============================================================
```

---

### Method 2: Manual Dataset Downloads & Links

If you prefer to download datasets manually or if a specific network blocks Python downloads, here are the direct links and destination folders:

| Dataset | URL / Command | Destination Folder |
|---|---|---|
| **Udacity Self-Driving Car** | `git clone --depth 1 https://github.com/Aman-py/data.git` | `data/` (contains `driving_log.csv` and `IMG/`) |
| **UCI HAR Dataset** | [Download Zip](https://github.com/greenglobal/uci-har-dataset/archive/refs/heads/master.zip) | Extract into `har_data/uci-har-dataset-master/` |
| **Quora Question Pairs** | [Download train.csv.zip](https://raw.githubusercontent.com/Koorimikiran369/Quora-Question-Pairing/main/train.csv.zip) | Extract into root folder as `train.csv` |
| **Nottingham ABC Tunes** | `git clone --depth 1 https://github.com/jukedeck/nottingham-dataset.git` | `nottingham-dataset/` |
| **MAESTRO MIDI** | [Download MIDI Zip](https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip) | Extract into `maestro/` |

---

## How to Run the Code (Step-by-Step)

### Running with VS Code (Recommended)

1. Open VS Code: `File -> Open Folder...` and select the `dla-lab` directory.
2. In the left file explorer, open either notebook:
   - **`DLA_W_1_to_4.ipynb`** (Weeks 1 to 4 + Week 13)
   - **`DLA_W_6_to_8.ipynb`** (Weeks 6 to 8)
3. In the upper-right corner of the open notebook editor, click **Select Kernel**.
4. Click **Python Environments...** and choose your Python 3.12 environment (or `(venv)`).
5. Click **Run All** (or use `Shift + Enter` to run each cell sequentially).

---

### Running with Jupyter Notebook

1. In your activated terminal, launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Your browser will open the Jupyter dashboard at `http://localhost:8888/tree`.
3. Click on **`DLA_W_1_to_4.ipynb`** or **`DLA_W_6_to_8.ipynb`**.
4. Go to menu: `Cell -> Run All` (or press `Shift + Enter` cell-by-cell).

---

### Running as Headless Python Scripts

If an examiner asks you to run the notebook non-interactively or export it to an HTML/PDF report:

- **Execute a notebook from the command line:**
  ```bash
  jupyter nbconvert --to notebook --execute DLA_W_6_to_8.ipynb --output DLA_W_6_to_8_executed.ipynb
  ```

- **Export executed notebook to HTML:**
  ```bash
  jupyter nbconvert --to html DLA_W_6_to_8.ipynb
  ```

---

## Lab Experiments Breakdown

### Notebook 1: `DLA_W_1_to_4.ipynb`

#### **Week 1: Data Ingestion & Exploration**
- **Objective:** Load the Udacity Self-Driving Car simulator driving log dataset (`data/driving_log.csv`).
- **Telemetry Columns:** `['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']`.
- **Path Cleaning:** Strips machine-specific directory prefixes using `ntpath.split()` to reference local image files in `data/IMG/`.

#### **Week 2: Steering Angle Analysis & Dataset Balancing**
- **Objective:** Address extreme class imbalance (most driving samples have steering angle $\approx 0.0$).
- **Technique:**
  1. Bin steering angles into 25 histogram bins.
  2. Set a sample threshold of 500 per bin.
  3. Randomly shuffle and remove excess samples beyond the threshold to prevent straight-driving bias.
  4. Train-test split (80% train, 20% test).
  5. Compute a **Mean Baseline MSE** as a benchmark for the CNN.

#### **Week 3: Image Preprocessing & NVIDIA CNN Model**
- **Image Preprocessing Pipeline (`imagePreprocessing`):**
  1. **Region of Interest (ROI) Cropping:** Crop out the sky and car hood `img[60:135, :, :]`.
  2. **Color Space Conversion:** Convert BGR to YUV (`cv2.COLOR_BGR2YUV`), as specified by NVIDIA's autonomous vehicle paper.
  3. **Gaussian Blur:** Filter out high-frequency sensor noise (`cv2.GaussianBlur` with $(3,3)$ kernel).
  4. **Resize:** Rescale image to $(66, 200, 3)$ matching the NVIDIA CNN input dimensions.
  5. **Normalization:** Normalize pixel intensities to $[0.0, 1.0]$.
- **NVIDIA CNN Architecture:**
  - 3 Conv2D layers with $5\times 5$ kernels, stride $(2, 2)$, ELU activation.
  - 2 Conv2D layers with $3\times 3$ kernels, stride $(1, 1)$, ELU activation.
  - Dropout ($0.5$).
  - Flatten + 4 Fully Connected (Dense) layers: 100 $\rightarrow$ 50 $\rightarrow$ 10 $\rightarrow$ 1 output (steering angle regression).
  - Optimizer: `Adam(learning_rate=1e-3)`, Loss: `Mean Squared Error (MSE)`.

#### **Week 4: Generator Training, Evaluation & Prediction**
- **Batch Generator:** Infinite Python generator `batch_generator` with on-the-fly image preprocessing and batch shuffling.
- **Model Training:** `model.fit()` with `steps_per_epoch` and `validation_steps`.
- **Evaluation:**
  - Plotting training vs validation loss curves over epochs.
  - Comparison of Actual Steering vs. Predicted Steering angles on the test set.
  - Performance verification: NVIDIA CNN Test MSE is significantly lower than the Mean Baseline MSE.

#### **Week 13: NLP Duplicate Question Detection & LSTM Music Generation**
- **Part 1 (Quora Question Pairs Duplicate Detection):**
  - Dataset: `train.csv` (`question1`, `question2`, `is_duplicate`).
  - Feature Engineering: Question lengths, length differences, word counts, common words, and Jaccard similarity.
- **Part 2 (Character-Level LSTM Music Generator):**
  - Dataset: Nottingham folk tunes in ABC notation (`nottingham-dataset/`).
  - Sequence Modeling: Character vocabulary mapping, windowed sequence extraction (`SEQ_LEN = 40`).
  - Architecture: `Embedding(vocab_size, 32)` $\rightarrow$ `LSTM(64, return_sequences=True)` $\rightarrow$ `TimeDistributed(Dense(vocab_size, softmax))`.
  - Generative Sampling: Seed prompt generation of new ABC musical melodies.
- **Part 3 (MAESTRO MIDI Note Extraction):**
  - Dataset: `maestro/` MIDI files.
  - Tool: `pretty_midi.PrettyMIDI` parses instruments, extracts start times and note pitch numbers $(0–127)$, and prepares numerical pitch arrays.

---

### Notebook 2: `DLA_W_6_to_8.ipynb`

#### **Week 6: UCI Human Activity Recognition (HAR) Ingestion & Preprocessing**
- **Dataset:** Human Activity Recognition Using Smartphones (UCI HAR).
- **Features:** 561-dimensional feature vectors representing triaxial acceleration and angular velocity recorded at 50 Hz from a Samsung Galaxy S II smartphone.
- **Activities (6 Classes):**
  1. `WALKING`
  2. `WALKING_UPSTAIRS`
  3. `WALKING_DOWNSTAIRS`
  4. `SITTING`
  5. `STANDING`
  6. `LAYING`
- **Data Quality Checks:** Missing values check (`isnull()`), duplicate rows check, infinite values check.
- **Preprocessing:**
  - Feature standardization using `StandardScaler` (zero mean, unit variance).
  - Target label zero-indexing: $y \in [1, 6] \rightarrow y \in [0, 5]$.

#### **Week 7: Exploratory Data Analysis (EDA)**
- **Visualizations:**
  - Categorical distribution bar chart of human activities.
  - Percentage pie chart showing activity proportions.
  - Kernel Density Estimation (KDE) and histogram of sensor features.

#### **Week 8: Machine Learning & Deep Learning Classifiers**
- **Classical Machine Learning Models:**
  1. **Logistic Regression** (`max_iter=2000`)
  2. **Random Forest Classifier** (`n_estimators=100`, `random_state=42`)
  3. **Support Vector Machine (SVM)** (`kernel='rbf'`)
  - Evaluation: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, and `classification_report`.
- **Deep Learning Architectures:**
  1. **1D Convolutional Neural Network (CNN):**
     - `Input(shape=(561, 1))`
     - `Conv1D(64, 3, activation='relu')` $\rightarrow$ `MaxPooling1D(2)` $\rightarrow$ `Dropout(0.3)`
     - `Conv1D(128, 3, activation='relu')` $\rightarrow$ `MaxPooling1D(2)` $\rightarrow$ `Flatten()`
     - `Dense(128, activation='relu')` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(6, activation='softmax')`
  2. **Long Short-Term Memory (LSTM):**
     - `Input(shape=(561, 1))`
     - `LSTM(64)` $\rightarrow$ `Dropout(0.3)`
     - `Dense(64, activation='relu')` $\rightarrow$ `Dense(6, activation='softmax')`
- **Callbacks & Metrics:**
  - `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)`.
  - Confusion Matrix heatmap plotted using Seaborn.

---

## Dependencies & requirement.txt

The complete dependencies file [requirement.txt](file:///e:/079_data/VNR/SEM_7/LABS/DLA/dla-lab/requirement.txt) contains:

```text
tensorflow>=2.16.0
opencv-python>=4.8.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
pretty_midi>=0.2.10
ipykernel>=6.29.0
notebook>=7.0.0
```

To install everything in one go:
```bash
pip install -r requirement.txt
```

---

## Exam Tips & Troubleshooting

### 1. Training Speed During the Exam
- The NVIDIA CNN and UCI HAR models run quickly on modern CPUs.
- If you have an exam time limit and need to demonstrate execution to an invigilator quickly, you can optionally reduce `epochs` in `model.fit()` (e.g. from 30 epochs to 3–5 epochs) for rapid completion without modifying the model architecture.

### 2. PowerShell Script Execution Policy
If PowerShell says `running scripts is disabled on this system` when activating `venv`:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. TensorFlow oneDNN / DirectML Notice
You may see log messages such as `oneDNN custom operations are on` or `TensorFlow GPU support is not available on native Windows`. These are normal informational notices from TensorFlow on Windows CPU and do **not** affect execution or model training results.

### 4. Running Without Internet
If the exam lab computer does not have internet access:
- Run `python setup_datasets.py` before disconnecting, or copy the already-extracted dataset folders (`data/`, `har_data/`, `train.csv`, `nottingham-dataset/`, `maestro/`) via a USB drive.
- Both notebooks automatically detect existing local folders and skip network downloads.

### 5. Verification Checklist Before Starting the Exam
- [x] Python 3.12 installed and added to PATH (`python --version`)
- [x] Virtual environment created and activated (`(venv)`)
- [x] `pip install -r requirement.txt` completed
- [x] `python setup_datasets.py` passes all checks
- [x] Notebook kernel set to Python 3.12 (`venv`)
- [x] "Run All" executes from top to bottom with zero errors!
