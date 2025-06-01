import numpy as np

class SimpleBaselineClassifier:
    def __init__(self, strategy='most_frequent', random_state=None, constant=None):
        if strategy not in ['most_frequent', 'uniform', 'constant']:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def __repr__(self):
        return (f"SimpleBaselineClassifier(strategy={self.strategy}, "
                f"random_state={self.random_state}, constant={self.constant})")

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        self._classes, self._class_counts = np.unique(y_train, return_counts=True)

        if self.strategy == 'most_frequent':
            self._most_frequent_class = self._classes[np.argmax(self._class_counts)]
        elif self.strategy == 'constant':
            if self.constant not in self._classes:
                raise ValueError(f"Constant value '{self.constant}' not found in training data target values.")
        elif self.strategy == 'uniform':
            self._rng = np.random.RandomState(self.random_state)
        return self

    def predict(self, X_test):
        n_samples = X_test.shape[0]

        if self.strategy == 'most_frequent':
            return np.full(n_samples, self._most_frequent_class)
        elif self.strategy == 'uniform':
            return self._rng.choice(self._classes, size=n_samples)
        elif self.strategy == 'constant':
            return np.full(n_samples, self.constant)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
