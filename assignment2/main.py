import os
from data_processor import DatasetPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# main.py
import os


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, "breast_cancer_wisconsin_diagnostic.zip")
    output_csv_path = os.path.join(base_dir, "dataset.csv")

    # Überprüfen, ob die ZIP-Datei existiert
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ ZIP-Datei nicht gefunden: {zip_path}")
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

    # Klassifikator trainieren
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Vorhersagen und Evaluation
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Genauigkeit: {accuracy:.4f}")

if __name__ == "__main__":
    main()
