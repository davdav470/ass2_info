import numpy as np

class SimpleBaselineClassifier:
    def __init__(self, strategy='most_frequent', random_state=None, constant=None):
        """
        Initialize the SimpleBaselineClassifier with the given strategy.

        Args:
            strategy (str): Prediction strategy. One of: 'most_frequent', 'uniform', 'constant'.
            random_state (int or None): Seed for random number generator (only for 'uniform').
            constant (int or str or None): Constant value to predict (only for 'constant').
        """
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def __repr__(self):
        return (f"SimpleBaselineClassifier(strategy={self.strategy}, "
                f"random_state={self.random_state}, constant={self.constant})")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the classifier by storing training labels and preparing for prediction.

        Args:
            X_train (np.ndarray): Feature matrix of the training set.
            y_train (np.ndarray): Target vector of the training set.
        """
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
        """
        Predict target values for the test set based on the selected strategy.

        Args:
            X_test (np.ndarray): Feature matrix of the test set.

        Returns:
            np.ndarray: Predicted target values.
        """
        n = X_test.shape[0]

        if self.strategy == 'most_frequent':
            return np.full(n, self._most_frequent)
        elif self.strategy == 'uniform':
            return self._rng.choice(self._unique_values, size=n)
        elif self.strategy == 'constant':
            return np.full(n, self.constant)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")
