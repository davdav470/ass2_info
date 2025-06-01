from data_processor import DatasetPreprocessor
from SimpleBaselineClassifier import SimpleBaselineClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Daten vorverarbeiten
    processor = DatasetPreprocessor("wdbc.data", "breast_cancer_cleaned.csv")
    processor.run()

    # Vorverarbeitete Daten laden
    df = pd.read_csv("breast_cancer_cleaned.csv")
    X = df.drop(columns=["Diagnosis"]).values
    y = df["Diagnosis"].values

    # Trainings- und Testdaten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Daten skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Verschiedene Classifier definieren
    classifiers = {
        "Baseline (most_frequent)": SimpleBaselineClassifier(strategy='most_frequent'),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42)
    }

    results = {}
    confusion_matrices = {}

    for name, clf in classifiers.items():
        # Baseline bekommt unskalierte Daten, alle anderen skalierte
        if name == "Baseline (most_frequent)":
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    # Bar Chart für die Metriken
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(classifiers))
    width = 0.2

    plt.figure(figsize=(12, 7))
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, [results[name][metric] for name in classifiers], width, label=metric.capitalize())
    plt.xticks(x + width*1.5, classifiers.keys(), rotation=20)
    plt.ylabel("Score")
    plt.xlabel("Classifier")
    plt.title("Evaluation Metrics for All Classifiers")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion Matrices plotten
    fig, axes = plt.subplots(1, len(classifiers), figsize=(20, 5), constrained_layout=True)
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center', color='red', fontsize=12)
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02)
    plt.suptitle("Confusion Matrices for All Classifiers")
    plt.show()
