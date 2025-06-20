from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from data_processor import DatasetPreprocessor
from simple_baseline_classifier import SimpleBaselineClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def evaluate_model(name, y_true, y_pred, metrics_dict):
    # Calculates and stores the main metrics for a classifier
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics_dict[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

    # Prints the results for the current classifier
    print(f"Results for: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")


def plot_metrics(metrics_dict):
    # Creates bar charts for all metrics and classifiers
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())

    for metric in metrics:
        values = [metrics_dict[model][metric] for model in models]
        plt.figure(figsize=(10, 6))
        plt.bar(models, values, color="skyblue")
        plt.title(f"{metric} of the Different Classifiers")
        plt.ylabel(f"{metric} in Percent")
        plt.xlabel("Different Classifiers")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


def plot_confusion_matrices(classifiers, x_test, y_test):
    # Creates confusion matrix plots for all classifiers
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, clf) in enumerate(classifiers.items()):
        y_pred = clf.predict(x_test.values)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[idx], cmap="Blues")
        axes[idx].set_title(f"{name}")

    # Remove unused subplots
    for j in range(len(classifiers), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Confusion Matrices")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def main():
    # Set base directories and file paths
    base_dir = Path(__file__).resolve().parent
    zip_path = base_dir / "breast_cancer_wisconsin_diagnostic.zip"
    output_csv_path = base_dir / "dataset.csv"

    # Check if the ZIP file exists
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    # Data preprocessing: extract and save as CSV
    preprocessor = DatasetPreprocessor(zip_file_path=str(zip_path))
    preprocessor.to_csv(csv_file_path=str(output_csv_path))

    # Load CSV and extract features/target variable
    df = pd.read_csv(output_csv_path)
    x = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Split into training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Simple Baseline": SimpleBaselineClassifier(strategy="most_frequent"),
    }

    metrics_dict = {}

    # Train and evaluate all classifiers
    for name, clf in classifiers.items():
        clf.fit(x_train.values, y_train.values)
        y_pred = clf.predict(x_test.values)
        evaluate_model(name, y_test, y_pred, metrics_dict)

    # Visualize metrics and confusion matrices
    plot_metrics(metrics_dict)
    plot_confusion_matrices(classifiers, x_test, y_test)


if __name__ == "__main__":
    main()
