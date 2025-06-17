import numpy as np

class SimpleBaselineClassifier:
    def __init__(self, strategy='most_frequent', random_state=None, constant=None):
        self.strategy = strategy # Sagt immer den häufigsten Wert in den Trainings-Zielwerten vorher.
        self.random_state = random_state    # Wählt zufällig gleichverteilt einen Wert aus den Zielklassen
        self.constant = constant    # Gibt einen konstanten Wert zurück

    def __repr__(self):
        return (f"SimpleBaselineClassifier(strategy={self.strategy}, "
                f"random_state={self.random_state}, constant={self.constant})")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._y_train = y_train

        if self.strategy == 'most_frequent':
            values, counts = np.unique(y_train, return_counts=True)
            self._most_frequent = values[np.argmax(counts)]
        elif self.strategy == 'uniform':
            self._unique_values = np.unique(y_train)
            self._rng = np.random.RandomState(self.random_state)
        elif self.strategy == 'constant':
            if self.constant not in y_train:
                raise ValueError(f"The constant value '{self.constant}' is not in the training labels.")
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        n = X_test.shape[0]

        if self.strategy == 'most_frequent':
            return np.full(n, self._most_frequent)
        elif self.strategy == 'uniform':
            return self._rng.choice(self._unique_values, size=n)
        elif self.strategy == 'constant':
            return np.full(n, self.constant)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")
