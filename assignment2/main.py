import os
import pandas as pd
from data_processor import DatasetPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from SimpleBaselineClassifier import SimpleBaselineClassifier

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, "breast_cancer_wisconsin_diagnostic.zip")
    output_csv_path = os.path.join(base_dir, "dataset.csv")

    # Überprüfen, ob die ZIP-Datei existiert
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ ZIP-Datei nicht gefunden: {zip_path}")
    
    # Vorverarbeitung
    preprocessor = DatasetPreprocessor(zip_file_path=zip_path)
    preprocessor.to_csv(csv_file_path=output_csv_path)

    # CSV laden
    df = pd.read_csv(output_csv_path)

    # Merkmale und Zielvariable trennen
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    # Trainings-/Testdaten splitten
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === Klassifikator 1: Random Forest ===
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)

    # === Klassifikator 2: SimpleBaselineClassifier (most_frequent) ===
    baseline = SimpleBaselineClassifier(strategy='most_frequent')
    baseline.fit(X_train.values, y_train.values)
    y_pred_baseline = baseline.predict(X_test.values)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

    # Ergebnisse ausgeben
    print(f"✅ RandomForest Genauigkeit:        {rf_accuracy:.4f}")
    print(f"📊 SimpleBaseline (most_frequent): {baseline_accuracy:.4f}")

if __name__ == "__main__":
    main()
