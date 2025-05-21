import pandas as pd
import os

class DatasetPreprocessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.column_names = [
            'ID', 'Diagnosis',
            # 30 Merkmale
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
            'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
    
    def load_data(self):
        print(f"Lade Daten aus {self.input_file} ...")
        df = pd.read_csv(self.input_file, header=None, names=self.column_names)
        print("Daten geladen.")
        return df

    def preprocess(self, df):
        print("Verarbeite Daten ...")
        df = df.drop(columns=['ID'])  # ID ist unnötig
        df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})  # 'M' = bösartig (1), 'B' = gutartig (0)
        return df

    def save_to_csv(self, df):
        df.to_csv(self.output_file, index=False)
        print(f"Daten gespeichert in {self.output_file}")

    def run(self):
        df = self.load_data()
        df = self.preprocess(df)
        self.save_to_csv(df)


if __name__ == "__main__":
    processor = DatasetPreprocessor("wdbc.data", "breast_cancer_cleaned.csv")
    processor.run()
