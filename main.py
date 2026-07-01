import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/Titanic-Dataset.csv"

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "calibrated_model.pkl"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "optimal_threshold.txt"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_banner():

    print("=" * 60)
    print(" PlaceMux Task 12")
    print(" Production Binary Classification")
    print("=" * 60)

def load_dataset(path):

    try:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nDataset not found:\n{path}"
            )

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        print("\nDataset Loaded Successfully")
        print(f"Shape : {df.shape}")

        return df

    except Exception as e:

        print("\nError while loading dataset.")
        print(e)

        raise

def validate_dataset(df):

    print("\nValidating Dataset...")

    required_columns = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Columns: {missing}"
        )

    duplicates = df.duplicated().sum()

    if duplicates:

        print(f"Removing {duplicates} duplicate rows")

        df = df.drop_duplicates()

    print("Validation Complete")

    return df

def prepare_data(df):

    X = df.drop("Survived", axis=1)

    y = df["Survived"]

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return (
        X,
        y,
        numeric_features,
        categorical_features
    )

def split_data(X, y):

    return train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

def build_preprocessor(
    numeric_features,
    categorical_features
):

    numeric_transformer = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(strategy="median")
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                numeric_transformer,
                numeric_features
            ),

            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor

def build_pipeline(preprocessor):

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(
                    random_state=42
                )
            )
        ]
    )

    return pipeline

def tune_model(pipeline, X_train, y_train):

    print("\nTraining Model...")
    print("Performing Hyperparameter Tuning...")

    parameter_grid = {

        "classifier__n_estimators": [100, 200],

        "classifier__max_depth": [None, 5, 10],

        "classifier__min_samples_split": [2, 5],

        "classifier__min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=parameter_grid,

        cv=5,

        scoring="f1",

        n_jobs=-1,

        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters Found")

    print(grid_search.best_params_)

    print(f"Best CV F1 Score : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

def calibrate_model(model, X_train, y_train):

    print("\nCalibrating Probabilities...")

    calibrated_model = CalibratedClassifierCV(

        estimator=model,

        method="sigmoid",

        cv=5
    )

    calibrated_model.fit(X_train, y_train)

    print("Calibration Completed Successfully")

    return calibrated_model

def plot_calibration_curve(model, X_test, y_test):

    print("\nGenerating Calibration Curve...")

    plt.figure(figsize=(6, 6))

    CalibrationDisplay.from_estimator(
        model,
        X_test,
        y_test,
        n_bins=10
    )

    plt.title("Calibration Curve")

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "calibration_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def find_best_threshold(model, X_test, y_test):

    print("\nFinding Optimal Threshold...")

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)

    best_threshold = 0.50

    best_f1 = 0

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_test,
            predictions
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = threshold

    print(f"Optimal Threshold : {best_threshold:.2f}")

    print(f"Best F1 Score : {best_f1:.4f}")

    return best_threshold

def save_operating_point(
        threshold,
        model,
        X_test,
        y_test
):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    confusion = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = confusion.ravel()

    false_positive_rate = fp / (fp + tn)

    false_negative_rate = fn / (fn + tp)

    report = f"""
Operating Point Report
======================

Optimal Threshold : {threshold:.2f}

Precision : {precision:.4f}

Recall : {recall:.4f}

F1 Score : {f1:.4f}

False Positive Rate : {false_positive_rate:.4f}

False Negative Rate : {false_negative_rate:.4f}
"""

    with open(
        os.path.join(
            OUTPUT_DIR,
            "operating_point.txt"
        ),
        "w"
    ) as file:

        file.write(report)

    with open(
        THRESHOLD_PATH,
        "w"
    ) as file:

        file.write(str(threshold))

def cross_validation_analysis(model, X, y):

    print("\nRunning Cross Validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {

        "accuracy": "accuracy",

        "precision": "precision",

        "recall": "recall",

        "f1": "f1",

        "roc_auc": "roc_auc"
    }

    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1
    )

    results = pd.DataFrame({

        "Accuracy": scores["test_accuracy"],

        "Precision": scores["test_precision"],

        "Recall": scores["test_recall"],

        "F1": scores["test_f1"],

        "ROC_AUC": scores["test_roc_auc"]

    })

    results.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "cross_validation_results.csv"
        ),

        index=False
    )

    print(results)

def segment_analysis(model, X_test, y_test, threshold):

    print("\nPerforming Segment Analysis...")

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    results = []

    segments = {

        "Gender": X_test["Sex"],

        "PassengerClass": X_test["Pclass"]
    }

    for segment_name, values in segments.items():

        for value in sorted(values.unique()):

            mask = values == value

            if mask.sum() == 0:
                continue

            y_true = y_test[mask]

            y_pred = predictions[mask]

            results.append({

                "Segment": segment_name,

                "Group": value,

                "Samples": len(y_true),

                "Accuracy": accuracy_score(y_true, y_pred),

                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

                "F1 Score": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "segment_analysis.csv"
        ),

        index=False
    )

    print(results_df)

def evaluate_model(model, X_test, y_test, threshold):

    print("\nEvaluating Model...")

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")

    report = classification_report(
        y_test,
        predictions
    )

    with open(

        os.path.join(
            OUTPUT_DIR,
            "evaluation_report.txt"
        ),

        "w"

    ) as file:

        file.write("Classification Report\n\n")

        file.write(report)

        file.write("\n")

        file.write(f"ROC AUC : {roc_auc:.4f}")

def plot_confusion_matrix(model, X_test, y_test, threshold):

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(5,5))

    plt.imshow(matrix)

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xticks([0,1],["No","Yes"])

    plt.yticks([0,1],["No","Yes"])

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i,j],
                ha="center",
                va="center"
            )

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

def plot_roc(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr)

    plt.plot([0,1],[0,1])

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "roc_curve.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

def plot_precision_recall(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:,1]

    precision, recall, _ = precision_recall_curve(

        y_test,

        probabilities
    )

    plt.figure(figsize=(6,6))

    plt.plot(recall, precision)

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            "precision_recall_curve.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

def inference_latency(model, X_test):

    print("\nCalculating Inference Latency...")

    sample = X_test.iloc[[0]]

    runs = 100

    start = time.perf_counter()

    for _ in range(runs):

        model.predict(sample)

    end = time.perf_counter()

    average_time = (end-start)/runs

    print(f"Average Prediction Time : {average_time:.6f} seconds")

    with open(

        os.path.join(
            OUTPUT_DIR,
            "evaluation_report.txt"
        ),

        "a"

    ) as file:

        file.write(

            f"\n\nAverage Inference Time : {average_time:.6f} seconds"
        )

def live_demo(model, X_test, threshold):

    print("\nLive Prediction Demo")

    demo = X_test.head(5)

    probabilities = model.predict_proba(demo)[:,1]

    predictions = (probabilities >= threshold).astype(int)

    output = demo.copy()

    output["Probability"] = probabilities

    output["Prediction"] = predictions

    print(output)

    output.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "prediction_examples.csv"
        ),

        index=False
    )

def save_model(model):

    print("\nSaving Model...")

    joblib.dump(

        model,

        MODEL_PATH
    )

    print("Model Saved Successfully")

def main():

    print_banner()

    try:

        df = load_dataset(DATA_PATH)

        df = validate_dataset(df)

        X, y, numeric, categorical = prepare_data(df)

        X_train, X_test, y_train, y_test = split_data(X, y)

        preprocessor = build_preprocessor(

            numeric,

            categorical
        )

        pipeline = build_pipeline(preprocessor)

        best_model = tune_model(

            pipeline,

            X_train,

            y_train
        )

        calibrated_model = calibrate_model(

            best_model,

            X_train,

            y_train
        )

        plot_calibration_curve(

            calibrated_model,

            X_test,

            y_test
        )

        threshold = find_best_threshold(

            calibrated_model,

            X_test,

            y_test
        )

        save_operating_point(

            threshold,

            calibrated_model,

            X_test,

            y_test
        )

        cross_validation_analysis(

            calibrated_model,

            X,

            y
        )

        segment_analysis(

            calibrated_model,

            X_test,

            y_test,

            threshold
        )

        evaluate_model(

            calibrated_model,

            X_test,

            y_test,

            threshold
        )

        plot_confusion_matrix(

            calibrated_model,

            X_test,

            y_test,

            threshold
        )

        plot_roc(

            calibrated_model,

            X_test,

            y_test
        )

        plot_precision_recall(

            calibrated_model,

            X_test,

            y_test
        )

        inference_latency(

            calibrated_model,

            X_test
        )

        live_demo(

            calibrated_model,

            X_test,

            threshold
        )

        save_model(

            calibrated_model
        )

        print("\nTask Completed Successfully")

    except Exception as e:

        print("\nExecution Failed")

        print(e)


if __name__ == "__main__":

    main()