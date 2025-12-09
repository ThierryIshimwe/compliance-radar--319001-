
---

## **Compliance Radar — Departmental Risk Detection**

**Team Members:

Mariavittoria Giurato (310091)

Thierry Ishimwe (321371)

Dimitar Ivanov (315691)
---

## **Section 1 — Introduction**

This project develops *Compliance Radar*, a machine learning system designed to identify departments that exhibit elevated compliance risk.
Using the dataset **org_compliance_data.db**, which contains operational metrics, audit scores, reporting behavior, engagement indicators, and organizational metadata, our goal is to:

* Detect high-risk departments
* Understand the operational and behavioral factors behind risk
* Provide interpretable insights that help compliance teams prioritize interventions

This is a **binary classification task**, where each department is labeled:

* **1 — High-risk**, if its `dept_id` appears in the high-risk table
* **0 — Normal-risk**, otherwise

The purpose goes beyond prediction: **the model must strengthen internal accountability processes** by surfacing risk drivers, outlier patterns, and governance weaknesses.

---

## **Section 2 — Methods**

### **2.1 Data Loading & EDA**

We loaded and explored all database tables. Our EDA highlighted:

* 682 departments in total, 200 high-risk (29%)
* 25 numeric features, 12 categorical features
* Important numeric patterns:

  * High-risk departments had:

    * ↑ overall risk
    * ↑ reporting gaps
    * ↑ exposure (financial + operational)
    * ↓ compliance scores
* Strong correlations (Top Positive):

  * `violations_past_3years` (0.71)
  * `risk_exposure_operational` (0.64)
  * `risk_exposure_financial` (0.58)
* Strong Negative:

  * `improvement_commitment` (–0.41)
  * `interdept_collaboration_score` (–0.24)

We used targeted visualizations (distributions, scatter plots, correlation heatmaps) while keeping the notebook lightweight as required.

---

### **2.2 Preprocessing**

We applied a systematic cleaning pipeline:

| Step                   | Description                                                                      |
| ---------------------- | -------------------------------------------------------------------------------- |
| Missing handling       | `"None"`, `"NA"`, blanks → `np.nan`; categorical → `"Unknown"`; numeric → median |
| Category normalization | Standardized team_size, reporting_structure, divisions                           |
| Duplicates             | Removed 27 conflicting `dept_id` rows, kept clean version                        |
| Type conversion        | All categorical → `category`; numerics → float/int                               |
| Integrity checks       | Missing → 0; duplicates → 0                                                      |

---

### **2.3 Feature Engineering**

We built **expert-driven features** based on compliance logic:

| Feature               | Meaning                                   |
| --------------------- | ----------------------------------------- |
| `avg_audit_score`     | Overall audit performance                 |
| `audit_consistency`   | Stability of audit scores                 |
| `exposure_index`      | Combined financial + operational exposure |
| `reporting_severity`  | Reporting gaps × exposure                 |
| `risk_intensity`      | Risk relative to compliance               |
| `training_efficiency` | Training hours adjusted for compliance    |
| `unknown_count_cat`   | Governance/reporting opacity              |

We also created:

* **Winsorized features** for stability
* **Interaction features** from EDA (risk × gaps, compliance × gaps)

Final feature matrix: **61 columns**

---

### **2.4 Modeling Overview**

We tested **three models** using the required experimental design:

1. **Logistic Regression (scaled)**
2. **Random Forest (unscaled tree pipeline)**
3. **MLP Neural Network (scaled, small architecture)**

Split:

* **70% Train**
* **15% Validation**
* **15% Test (final use only)**

Performance metrics:

* Accuracy
* Precision
* Recall (priority)
* F1
* ROC AUC

---

## **Section 3 — Experimental Design**

### **Baselines**

We first trained each model with default parameters.

| Model               | AUC (Val) |
| ------------------- | --------- |
| Logistic Regression | 0.9877    |
| Random Forest       | 0.9973    |
| MLP Neural Network  | 0.9927    |

Random Forest performed best → chosen for tuning.

---

### **Hyperparameter Tuning**

We used **RandomizedSearchCV**, tuning:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `bootstrap`

Tuned Model Comparison:

| Model                     | Accuracy  | Precision | Recall    | F1        | AUC       |
| ------------------------- | --------- | --------- | --------- | --------- | --------- |
| Tuned Logistic Regression | 0.971     | 0.935     | 0.967     | 0.951     | 0.994     |
| Tuned Random Forest       | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| Tuned MLP                 | 0.961     | 0.933     | 0.933     | 0.933     | 0.991     |

**Winner: Tuned Random Forest**

---

## **Section 4 — Results**

### **4.1 Test Set Metrics**

(never used during training)

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **0.9806** |
| Precision | **1.0000** |
| Recall    | **0.9333** |
| F1 Score  | **0.9655** |
| AUC       | **0.9932** |

---

### **4.2 Confusion Matrix**

Insert this image inside `/images/`:

```
images/confusion_matrix.png
```

Interpretation:

* Only **2 false negatives**
* **0 false positives**
* Model is conservative and safe: when it flags risk, it is always correct.

---

### **4.3 Feature Importance (Top drivers)**

Add image:

```
images/feature_importances.png
```

Key insights:

* **compliance_score_final** → strongest predictor
* **overall_risk_w / score** → captures department volatility
* **violations_past_3years** → strong structural signal
* **exposure_index** + reporting features reflect early warning signals
* Several **Unknown-category flags** were non-trivial predictors → incomplete reporting is itself a risk indicator

---

## **Section 5 — Conclusions**

This project delivered a complete ML pipeline capable of:

* Detecting high-risk departments with **high recall**
* Providing explainability via feature importances
* Highlighting early-warning indicators such as reporting gaps, exposure levels, and governance opacity

### **Limitations**

* Some categorical columns are extremely sparse
* Audit and reporting behaviors may be influenced by seasonal factors
* Unknown values may mix genuine ambiguity and missing reporting discipline

### **Future Work**

* Use SHAP for deeper interpretability
* Add temporal modeling if historical data becomes available
* Deploy model as a monitoring dashboard for real-time alerts

---



