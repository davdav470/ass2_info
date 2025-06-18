import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from data_processor import DatasetPreprocessor
from SimpleBaselineClassifier import SimpleBaselineClassifier

def evaluate_model(name, y_true, y_pred, metrics_dict):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics_dict[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    print(f"\n📈 Ergebnisse für: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

def plot_metrics(metrics_dict):
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())

    for metric in metrics:
        values = [metrics_dict[model][metric] for model in models]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color='skyblue')
        plt.title(f"{metric} of the Different Classifiers")
        plt.ylabel(f"{metric} in Percent")
        plt.xlabel("Different Classifiers")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

def plot_confusion_matrices(classifiers, X_test, y_test):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, clf) in enumerate(classifiers.items()):
        y_pred = clf.predict(X_test.values)
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[idx], cmap='Blues')
        axes[idx].set_title(f"{name}")

    for j in range(len(classifiers), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Confusion Matrices")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, "breast_cancer_wisconsin_diagnostic.zip")
    output_csv_path = os.path.join(base_dir, "dataset.csv")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ ZIP-Datei nicht gefunden: {zip_path}")

    preprocessor = DatasetPreprocessor(zip_file_path=zip_path)
    preprocessor.to_csv(csv_file_path=output_csv_path)

    df = pd.read_csv(output_csv_path)
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Simple Baseline": SimpleBaselineClassifier(strategy='most_frequent')
    }

    metrics_dict = {}

    for name, clf in classifiers.items():
        clf.fit(X_train.values, y_train.values)
        y_pred = clf.predict(X_test.values)
        evaluate_model(name, y_test, y_pred, metrics_dict)

    plot_metrics(metrics_dict)
    plot_confusion_matrices(classifiers, X_test, y_test)

if __name__ == "__main__":
    main()