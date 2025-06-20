################################################################################
# Author 1:      Alina Grundner  
# MatNr 1:       12331261
# Author 2:      David Leibold 
# MatNr 2:       12335498
# Author 3:      Lukas Umfahrer
# MatNr 3:       12337160
# File:          SimpleBaselineClassifier.py
# Description: the code for the class SimpleBaselineClassifier
# Comments:    ... comments for the tutors ...
#              ... can be multiline ...
################################################################################
import numpy as np

class SimpleBaselineClassifier:
    def __init__(self, strategy='most_frequent', random_state=None, constant=None):
        # Speichert die gewählte Strategie und ggf. Parameter
        self.strategy = strategy  
        self.random_state = random_state 
        self.constant = constant 

    def __repr__(self):
        # Gibt eine String-Repräsentation des Klassifikators zurück
        return (f"SimpleBaselineClassifier(strategy={self.strategy}, "
                f"random_state={self.random_state}, constant={self.constant})")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._y_train = y_train

        if self.strategy == 'most_frequent':
            # Bestimmt die häufigste Klasse im Training
            values, counts = np.unique(y_train, return_counts=True)
            self._most_frequent = values[np.argmax(counts)]
        elif self.strategy == 'uniform':
            # Speichert alle einzigartigen Klassen und initialisiert Zufallszahlengenerator
            self._unique_values = np.unique(y_train)
            self._rng = np.random.RandomState(self.random_state)
        elif self.strategy == 'constant':
            # Prüft, ob der konstante Wert im Training vorkommt
            if self.constant not in y_train:
                raise ValueError(f"The constant value '{self.constant}' is not in the training labels.")
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Gibt Vorhersagen für die Testdaten entsprechend der Strategie zurück
        n = X_test.shape[0]

        if self.strategy == 'most_frequent':
            # Sagt immer die häufigste Klasse voraus
            return np.full(n, self._most_frequent)
        elif self.strategy == 'uniform':
            # Sagt zufällig eine der Trainingsklassen voraus
            return self._rng.choice(self._unique_values, size=n)
        elif self.strategy == 'constant':
            # Sagt immer den konstanten Wert voraus
            return np.full(n, self.constant)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")
