import zipfile
from pathlib import Path
import pandas as pd


class DatasetPreprocessor:
    def __init__(self, zip_file_path: str):
        self.extract_dir = Path("temp_extracted")

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.extract_dir)
            print("Files in ZIP:", zip_ref.namelist())  # Debug output

            # Search for the first .csv or .data file
            for name in zip_ref.namelist():
                if name.endswith((".csv", ".data")):
                    self.data_file_path = self.extract_dir / name
                    break
            else:
                raise FileNotFoundError("No .csv or .data file found in ZIP.")

        print(f"Data path set to: {self.data_file_path}")

        # Read the data file
        self.df = pd.read_csv(self.data_file_path, header=None)

        # Set column names according to documentation
        self.df.columns = [
            "id",
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

        # Remove the ID column, as it is not useful for ML
        self.df.drop(columns=["id"], inplace=True)

        # Convert target variable (diagnosis) to binary: M → 1, B → 0
        self.df["diagnosis"] = self.df["diagnosis"].map({"M": 1, "B": 0})
        if self.df["diagnosis"].isnull().any():
            raise ValueError("Invalid values found in the 'diagnosis' column.")

        # Move the diagnosis column to the end
        diagnosis = self.df.pop("diagnosis")
        self.df["diagnosis"] = diagnosis

    def to_csv(self, csv_file_path: str):
        self.df.to_csv(csv_file_path, index=False)
