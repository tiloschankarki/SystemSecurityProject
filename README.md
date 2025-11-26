
# **IoT Malware Detection — System Security Project**

Machine Learning Models + Scenario-Based Evaluation + Streamlit Dashboard

---

## **Project Overview**

This project builds an IoT intrusion-detection system using supervised machine learning.
We focus on **scenario-based evaluation**, where models are trained on a subset of CTU IoT captures and tested on entirely different real-world scenarios.

This prevents dataset leakage and measures whether the model can generalize to **new attacks**, **new devices**, and **new environments**.

The project includes:

* Decision Tree (baseline + tuned)
* Random Forest (baseline + tuned)
* Full preprocessing pipeline
* Scenario-based data split (realistic security evaluation)
* Streamlit dashboard for evaluation

---

## **Folder Structure**

```
SystemSecurityProject/
│
├── data/
│   ├── raw/                # ORIGINAL large CTU files (ignored in Git)
│   ├── intermediate/       # Cleaned scenario CSVs used for training
│   └── processed/          # Scenario train/test splits (ignored in Git)
│
├── notebooks/              # Jupyter notebooks (EDA, debugging)
├── reports/
│   └── roc_curves/         # ROC curve PNGs
│
├── saved_models/           # .pkl models (Git-ignored)
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess_scenario_split.py
│   ├── models/
│   │   ├── decision_tree_scenario.py
│   │   ├── random_forest_scenario.py
│   │   └── evaluate.py
│   └── dashboard/
│       └── app.py          # Streamlit dashboard
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## **Installation & Setup**

### **1. Clone the repository**

```
git clone https://github.com/your-username/SystemSecurityProject.git
cd SystemSecurityProject
```

### **2. Create and activate virtual environment**

```
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

### **3. Install dependencies**

```
pip install -r requirements.txt
```

### **4. Download intermediate CSV files**

Because raw CTU data is several GB, we provide **processed intermediate CSVs** via Google Drive:

 *Google Drive Link Here*

Download them into:

```
data/intermediate/
```

---

## ** Data Preprocessing (Scenario-Based)**

We use a **realistic scenario split** designed to mimic real intrusion detection conditions.

* **Train on scenarios 3-1, 8-1, 20-1, 21-1**
* **Test on unseen scenarios:** Somfy-01, 34-1, 42-1, 44-1, Honeypot 4-1, Honeypot 5-1

Run preprocessing:

```
python src/preprocessing/preprocess_scenario_split.py
```

This generates:

```
data/processed/X_train_scenario.csv
data/processed/X_test_scenario.csv
data/processed/y_train_scenario.csv
data/processed/y_test_scenario.csv
```

---

## **Models Included**

### **1. Decision Tree**

* Baseline
* Tuned via GridSearchCV
* Evaluates accuracy, precision, recall, F1, confusion matrix, ROC

Run:

```
python src/models/decision_tree_scenario.py
```

### **2. Random Forest**

* Baseline
* Tuned
* More robust generalization

Run:

```
python src/models/random_forest_scenario.py
```

### **3. Evaluation Utility**

Supports:

* Metrics
* ROC plots
* Confusion matrices

Run:

```
python src/models/evaluate.py
```

---

## **Streamlit Dashboard**

The dashboard visualizes:

* Model comparison
* Confusion matrices
* ROC curves
* Feature importance
* Threshold slider
* Optional CSV upload for user evaluation

Run the dashboard:

```
streamlit run src/dashboard/app.py
```

---

## **What’s in `.gitignore`**

We ignore all large files:

```
# Raw CTU data (1–6 GB total)
data/raw/*

# Generated outputs
data/processed/*
saved_models/*

# Virtual environment
.venv/

# Python cache
__pycache__/
*.pyc
```

---

## ** Why Scenario-Based Splitting?**

Most student projects incorrectly:

* Merge all scenarios into one file
* Randomly split train/test
* Get 99–100% accuracy (data leakage)

This is **not realistic** and does **not** represent real IoT intrusion detection.

Our scenario split ensures:

* Training sees only certain devices and attacks
* Testing contains **entirely unseen** attacks, devices, and environments
* Results reflect **true generalization**

This is closer to real-world IDS deployment.

---

## **Summary of Findings**

| Model                    | Accuracy | Precision   | Recall | Notes                        |
| ------------------------ | -------- | ----------- | ------ | ---------------------------- |
| Decision Tree (baseline) | ~0.90    | Good        | Good   | Lightweight but unstable     |
| Decision Tree (tuned)    | ~0.90    | Same        | Same   | Depth limited—no improvement |
| Random Forest (baseline) | Higher   | More robust | Better | Handles variance better      |
| Random Forest (tuned)    | Best     | Best        | Best   | Recommended model            |

---
## **Notes**

* Raw CTU logs are **NOT** included (size limitations)
* Use the Google Drive intermediate file set to run the project
* Project is reproducible end-to-end

---

