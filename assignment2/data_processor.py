import os
import zipfile
import pandas as pd

class DatasetPreprocessor:
    def __init__(self, zip_file_path: str):
        self.zip_file_path = zip_file_path
        self.extracted_path = "temp_extracted"
        self.data_file_name = None
        self.df = None

        self._extract_zip()
        self._find_data_file()
        self._load_data()

    def _extract_zip(self):
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.extracted_path)
        print("📦 ZIP entpackt.")

    def _find_data_file(self):
        files = os.listdir(self.extracted_path)
        print(f"Dateien im ZIP: {files}")
        for f in files:
            if f.endswith(".data"):
                self.data_file_name = f
                break
        if not self.data_file_name:
            raise FileNotFoundError("Keine .data-Datei im ZIP gefunden.")
        print(f"✅ Datenpfad gesetzt: {os.path.join(self.extracted_path, self.data_file_name)}")

    def _load_data(self):
        file_path = os.path.join(self.extracted_path, self.data_file_name)
        # Spaltennamen gemäß UCI-Dokumentation
        column_names = [
            'id', 'diagnosis',
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        self.df = pd.read_csv(file_path, header=None, names=column_names)
        # Entferne die ID-Spalte (nicht relevant für ML)
        self.df.drop("id", axis=1, inplace=True)

    def to_csv(self, csv_file_path: str):
        if self.df is not None:
            self.df.to_csv(csv_file_path, index=False)
            print(f"✅ CSV gespeichert unter: {csv_file_path}")
        else:
            raise ValueError("Daten wurden nicht korrekt geladen.")
