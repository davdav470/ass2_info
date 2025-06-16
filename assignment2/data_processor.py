import os
import pandas as pd
import zipfile
from typing import Optional


class DatasetPreprocessor:
    def __init__(self, zip_file_path: str):
        """
        Extracts ZIP, preprocesses dataset, stores it internally.
        """
        self.zip_file_path = zip_file_path
        self._data: Optional[pd.DataFrame] = None

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall("temp_extracted")

        # Annahme: Datei heißt so im ZIP
        csv_filename = "wdbc.data"  # Passe das an den tatsächlichen Dateinamen an
        file_path = os.path.join("temp_extracted", csv_filename)
        df = pd.read_csv(file_path)

        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        df = df.dropna()

        if 'diagnosis' in df.columns and df.columns[-1] != 'diagnosis':
            target = df['diagnosis']
            df = df.drop(columns=['diagnosis'])
            df['diagnosis'] = target

        self._data = df

    def to_csv(self, csv_file_path: str) -> None:
        if self._data is None:
            raise ValueError("No data loaded.")

        self._data.to_csv(csv_file_path, index=False)