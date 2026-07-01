# PlaceMux Task 12 – Production-Grade Binary Classification

## Overview

This project implements a production-ready binary classification pipeline using the Titanic dataset. The objective is to build a reliable machine learning model that not only predicts passenger survival but also produces calibrated probabilities, selects an optimal decision threshold, evaluates stability across multiple folds and data segments, and packages the trained model for future deployment.

The project follows industry-standard machine learning practices such as preprocessing pipelines, hyperparameter tuning, probability calibration, cross-validation, threshold optimization, inference latency measurement, and model serialization.

---

## Objective

The primary objectives of this project are to:

* Train a robust binary classification model.
* Perform hyperparameter tuning to improve performance.
* Calibrate prediction probabilities using Platt Scaling.
* Select a cost-optimal decision threshold.
* Evaluate model stability using Stratified Cross Validation.
* Analyze performance across important data segments.
* Measure inference latency.
* Package the trained model for serving.
* Generate reports, visualizations, and prediction examples.

---

## Dataset

Dataset: Titanic Dataset

Target Variable:

* Survived

  * 0 → Did Not Survive
  * 1 → Survived

The dataset contains passenger demographic and travel-related information used to predict survival.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

---

## Machine Learning Pipeline

The project follows the pipeline below:

1. Dataset Loading
2. Dataset Validation
3. Missing Value Handling
4. Feature Preprocessing
5. Train-Test Split
6. Hyperparameter Tuning using GridSearchCV
7. Random Forest Training
8. Probability Calibration using Platt Scaling
9. Threshold Optimization
10. Cross Validation
11. Segment Analysis
12. Model Evaluation
13. Inference Latency Analysis
14. Live Prediction Demonstration
15. Model Serialization

---

## Model Details

Algorithm:

Random Forest Classifier

Hyperparameter Tuning:

* GridSearchCV
* 5-Fold Cross Validation

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

## Probability Calibration

The trained classifier is calibrated using **Platt Scaling (Sigmoid Calibration)** through `CalibratedClassifierCV`.

Calibration improves the reliability of predicted probabilities so that confidence scores better represent the true likelihood of the positive class.

---

## Threshold Optimization

Instead of using the default classification threshold of **0.50**, the project evaluates multiple thresholds and selects the operating point that maximizes the F1 Score.

The selected threshold is saved for future inference.

---

## Cross Validation

Model stability is evaluated using **5-Fold Stratified Cross Validation**.

The following metrics are computed for every fold:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Results are exported to:

```
outputs/cross_validation_results.csv
```

---

## Segment Analysis

To ensure consistent model behavior, evaluation is performed across important data segments including:

* Gender
* Passenger Class

Metrics are exported to:

```
outputs/segment_analysis.csv
```

---

## Visualizations

The project automatically generates:

* Calibration Curve
* ROC Curve
* Precision-Recall Curve
* Confusion Matrix

These visualizations assist in evaluating classifier performance and probability quality.

---

## Output Files

```
outputs/

calibration_curve.png

confusion_matrix.png

cross_validation_results.csv

evaluation_report.txt

operating_point.txt

precision_recall_curve.png

prediction_examples.csv

roc_curve.png

segment_analysis.csv
```

---

## Saved Model

The trained calibrated classifier is stored inside:

```
models/calibrated_model.pkl
```

The optimal decision threshold is stored inside:

```
models/optimal_threshold.txt
```

These files can be directly used for deployment or future inference.

---

## Live Verification

The project demonstrates live predictions on unseen samples after training.

For every sample, the following are displayed:

* Predicted Probability
* Predicted Class

Prediction examples are also saved to:

```
outputs/prediction_examples.csv
```

---

## Error Handling

The project includes validation and exception handling for:

* Missing dataset
* Empty dataset
* Missing required columns
* Duplicate records
* Model training failures
* Calibration failures
* Model serialization errors

---

## Results

The implemented pipeline successfully performs:

* Binary classification
* Hyperparameter optimization
* Probability calibration
* Decision threshold optimization
* Stable cross-validation evaluation
* Segment-wise performance analysis
* Inference latency measurement
* Model packaging for serving

All generated reports, figures, and serialized models provide reproducible evidence of successful execution.

---

## Future Improvements

Possible future enhancements include:

* XGBoost and LightGBM comparison
* Bayesian Hyperparameter Optimization
* Explainable AI using SHAP
* FastAPI model deployment
* Docker containerization
* Continuous model monitoring
* Automated retraining pipeline

---

## Project Structure

```
placemux-task-12/

│── data/

│── models/

│── outputs/

│── main.py

│── requirements.txt

│── README.md

│── .gitignore
```

---

## Conclusion

This project demonstrates an end-to-end production-oriented binary classification workflow. In addition to achieving strong predictive performance, it emphasizes calibrated probabilities, threshold optimization, model stability, fairness checks across segments, and deployment readiness. The generated outputs, reports, and serialized model make the solution suitable for real-world machine learning workflows.
