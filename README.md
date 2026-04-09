# 💳 Credit Card Fraud Detection — Exploratory Data Analysis

> **A data analytics project on highly imbalanced financial transaction data using Python.**

---

## 📌 Project Overview

This project performs an in-depth **Exploratory Data Analysis (EDA)** on a real-world credit card transaction dataset to uncover patterns associated with fraudulent activity. The dataset contains **284,807 transactions**, of which only **492 (~0.17%)** are fraudulent — making this a classic **highly imbalanced classification problem**.

The goal is to understand the data's structure, identify meaningful signals, and engineer features that could support downstream fraud detection models.

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Records** | 284,807 transactions |
| **Features** | 31 columns (Time, V1–V28, Amount, Class) |
| **Target** | `Class` — 0 = Normal, 1 = Fraud |
| **Missing Values** | None |
| **Duplicates** | 1,081 removed → 283,726 clean records |

> **Note:** Features V1–V28 are the result of PCA transformation to protect user privacy. Only `Time`, `Amount`, and `Class` retain their original meaning.

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualizations |
| Seaborn | Statistical charts |
| Google Colab | Cloud-based notebook environment |

---

## 📁 Repository Structure

credit-card-fraud-detection/
│
├── creditcard.csv                  # Dataset (not included in repo due to size)
├── fraud_detection_eda.ipynb       # Main Jupyter/Colab notebook
└── README.md                       # This file
```

---
## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/)
2. Upload `creditcard.csv` when prompted
3. Run all cells sequentially (`Runtime → Run All`)

### Option 2: Local (Jupyter Notebook)
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# Launch notebook
jupyter notebook fraud_detection_eda.ipynb
```

> Replace the file upload cell with: `df = pd.read_csv('creditcard.csv')`

---

## 📊 Key Analysis Steps

### 1. Data Loading & Inspection
- Loaded dataset with `pd.read_csv()`
- Verified shape: **(284,807 rows × 31 columns)**
- Confirmed **zero missing values** across all features
- Identified and removed **1,081 duplicate rows**

### 2. Class Distribution
```
Class 0 (Normal):  284,315 transactions  (99.83%)
Class 1 (Fraud):       492 transactions   (0.17%)
```
This extreme imbalance means accuracy alone is a misleading metric — a model predicting "no fraud" 100% of the time would still achieve 99.83% accuracy.

### 3. Feature Engineering
Two new features were created to enrich the analysis:

- **`Hour`** — Derived from `Time` to capture the hour of day the transaction occurred
- **`Amount_Bin`** — Categorical bucketing of transaction amounts:
  - `0–50`, `50–100`, `100–500`, `500–1000`, `1000–5000`, `5000+`
- **`Log_Amount`** — Log-transformed `Amount` to reduce right-skew

### 4. Exploratory Visualizations

| Chart | Insight |
|---|---|
| Class Distribution Bar Chart | Visualizes the severe class imbalance |
| Amount Distribution by Class | Fraud transactions are often smaller amounts |
| Box Plot (Amount × Class) | Fraudulent amounts have slightly higher median |
| Fraud Transactions by Hour | Fraud peaks around **Hour 11** |
| Fraud by Amount Range | Most fraud occurs in the **0–50** range |
| Correlation Heatmap | Explores relationships across all 31 features |

### 5. Statistical Summary

| Class | Count | Mean Amount | Total Amount |
|---|---|---|---|
| Normal | 283,253 | $88.41 | $25,043,410 |
| Fraud | 473 | $123.87 | $58,591 |

> After deduplication: 283,726 total records; 473 fraud cases remain.

---

## 💡 Key Findings

1. **Severe Class Imbalance** — Only 0.17% of transactions are fraudulent. Any predictive model must account for this, using techniques like SMOTE, weighted loss functions, or anomaly detection.

2. **Fraud Peaks at Hour 11** — Fraudulent transactions are more concentrated around midday hours, which could reflect patterns in when legitimate cardholders are less likely to notice unauthorized activity.

3. **Small-Amount Fraud is Common** — The `0–50` range has the highest fraud count. This aligns with real-world behavior where fraudsters test cards with small charges before escalating.

4. **Fraudulent Transactions Have Higher Average Amounts** — Mean fraud amount ($123.87) is higher than normal ($88.41), suggesting some fraud events involve larger purchases.

5. **PCA Features (V1–V28) Hide Raw Signals** — These features cannot be directly interpreted, but their distributions differ between classes and are valuable for model training.

---

## 📈 Potential Next Steps

This EDA lays the groundwork for building a full fraud detection pipeline:

- **Modeling:** Logistic Regression, Random Forest, XGBoost, Isolation Forest
- **Evaluation Metrics:** Precision, Recall, F1-Score, ROC-AUC (not accuracy)
- **Imbalance Handling:** SMOTE oversampling, undersampling, class weights
- **Risk Scoring:** Classify transactions as Low / Medium / High Risk
- **Deployment:** Flask/FastAPI endpoint for real-time fraud scoring

---

## 🧠 What I Learned

- How to perform thorough **data quality checks** (missing values, duplicates, dtypes)
- The importance of **class imbalance awareness** in real-world ML problems
- How to use **feature engineering** (time extraction, log transforms, binning) to create meaningful signals
- How to use **multiple chart types** to tell a cohesive data story
- Why **standard accuracy is insufficient** for fraud detection tasks

---

## 👤 Author

**[Basir Ahammed Mandal]**
- GitHub: [@github.com/basirahamed002](https://github.com/github.com/basirahamed002)
- LinkedIn: [https://www.linkedin.com/in/basir-ahammed-mandal-b75935268/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

*Dataset originally published on Kaggle by the Machine Learning Group of ULB (Université Libre de Bruxelles).*


