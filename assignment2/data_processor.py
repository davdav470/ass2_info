import zipfile
import os
import pandas as pd

class DatasetPreprocessor:
    def __init__(self, zip_file_path: str):
        self.extract_dir = "temp_extracted"

        # ZIP-Datei entpacken
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)
            print("📁 Dateien im ZIP:", zip_ref.namelist())  # Debug-Ausgabe

            # Suche nach der ersten .csv oder .data Datei
            for name in zip_ref.namelist():
                if name.endswith('.csv') or name.endswith('.data'):
                    self.data_file_path = os.path.join(self.extract_dir, name)
                    break
            else:
                raise FileNotFoundError("❌ Keine .csv oder .data Datei im ZIP gefunden.")

        print(f"✅ Datenpfad gesetzt: {self.data_file_path}")

        # Daten einlesen (ohne Header)
        self.df = pd.read_csv(self.data_file_path, header=None)

        # Spaltennamen setzen laut Dokumentation
        self.df.columns = [
            "id", "diagnosis",
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]

        # Entferne die ID-Spalte, da sie keine sinnvolle Information für ML enthält
        self.df.drop(columns=["id"], inplace=True)

    def to_csv(self, csv_file_path: str):
        self.df.to_csv(csv_file_path, index=False)
