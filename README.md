# Intelligent Reading Comprehension and Quiz Generation System
## AI Lab Project — BS(CS) Spring 2026 | FAST NUCES Islamabad

---

## Project Overview
An AI-powered Reading Comprehension and Quiz Generation System built on the RACE dataset.
The system generates questions, verifies answers, creates distractors, and provides hints.

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Dataset
Place your `train.csv` file inside:
```
data/raw/train.csv
```
The preprocessing script will automatically split it 80/10/10 into train/dev/test.

### 3. Run Preprocessing
```bash
python src/preprocessing.py
```

### 4. Train Model A
```bash
python src/model_a_train.py
```

### 5. Train Model B
```bash
python src/model_b_train.py
```

### 6. Run the App
```bash
streamlit run ui/app.py
```

---

## Project Structure
```
race_rc_project/
├── data/
│   ├── raw/                  # Place train.csv here
│   └── processed/            # Auto-generated splits & features
├── models/
│   ├── model_a/traditional/  # Saved Model A pkl files
│   └── model_b/traditional/  # Saved Model B pkl files
├── src/
│   ├── preprocessing.py      # Dataset loading, cleaning, splitting, vectorization
│   ├── model_a_train.py      # Model A training script
│   ├── model_b_train.py      # Model B training script
│   ├── inference.py          # Unified inference API
│   └── evaluate.py           # Metric computation
├── ui/
│   └── app.py                # Streamlit app (4 screens)
├── notebooks/
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── experiments.ipynb     # Experiment tracking
├── tests/
│   └── test_inference.py     # Unit tests
├── requirements.txt
└── README.md
```

---

## Model Details

### Model A — Question & Answer Generator/Verifier
- Logistic Regression + TF-IDF / OHE
- SVM + TF-IDF / OHE
- Naive Bayes
- K-Means Clustering (Unsupervised)
- GMM (Unsupervised)
- Label Propagation (Semi-Supervised)
- Soft Voting Ensemble

### Model B — Distractor & Hint Generator
- Random Forest Distractor Ranker
- Logistic Regression Hint Scorer
- Cosine Similarity + OHE features

---

## Dataset
RACE (ReAding Comprehension from Examinations) — Lai et al., 2017
- ~87,866 samples used (train.csv)
- Split: 80% train / 10% dev / 10% test
- Columns: id, article, question, A, B, C, D, answer
