import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_processor import DatasetPreprocessor
from SimpleBaselineClassifier import SimpleBaselineClassifier
import os


def main():

    # Verzeichnis der aktuellen Datei (main.py)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, "breast_cancer_wisconsin_diagnostic.zip")
    output_csv_path = os.path.join(base_dir, "dataset.csv")  # oder ass2_info/dataset.csv, je nachdem wohin du willst


    # === 2. Daten vorbereiten ===
    preprocessor = DatasetPreprocessor(zip_file_path=zip_path)
    preprocessor.to_csv(csv_file_path=output_csv_path)


    # === 3. CSV laden ===
    df = pd.read_csv(output_csv_path)

    # Annahme: letzte Spalte ist die Zielvariable (target)
    X = df.iloc[:, :-1].values  # alle Features
    y = df.iloc[:, -1].values   # Zielvariable (z. B. diagnosis)

    # === 4. Trainings-/Testdaten Splitten ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === 5. Klassifikator erstellen ===
    clf = SimpleBaselineClassifier(strategy="most_frequent")
    # Alternativen:
    # clf = SimpleBaselineClassifier(strategy="uniform", random_state=42)
    # clf = SimpleBaselineClassifier(strategy="constant", constant=1)

    # === 6. Modell trainieren und vorhersagen ===
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # === 7. Evaluation ===
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"First 10 Predictions: {y_pred[:10]}")

if __name__ == "__main__":
    main()
