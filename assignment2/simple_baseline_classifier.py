import numpy as np


class SimpleBaselineClassifier:
    def __init__(
        self,
        strategy: str = "most_frequent",
        random_state: int | None = None,
        constant: int | str | None = None,
    ) -> None:
        # Stores the chosen strategy and optional parameters
        self.strategy = strategy  # 'most_frequent', 'uniform', or 'constant'
        self.random_state = random_state  # For reproducibility in 'uniform'
        self.constant = constant  # For the 'constant' strategy

    def __repr__(self) -> str:
        # Returns a formal string representation of the classifier
        return (
            f"SimpleBaselineClassifier(strategy={self.strategy}, "
            f"random_state={self.random_state}, constant={self.constant})"
        )

    def fit(
        self,
        _x_train: np.ndarray,  # type: ignore[type-arg]
        y_train: np.ndarray,  # type: ignore[type-arg]
    ) -> None:
        # Stores the training target values and prepares the chosen strategy
        self._y_train = y_train

        if self.strategy == "most_frequent":
            # Determines the most frequent class in the training data
            values, counts = np.unique(y_train, return_counts=True)
            self._most_frequent = values[np.argmax(counts)]
        elif self.strategy == "uniform":
            # Stores all unique classes and initializes the random number generator
            self._unique_values = np.unique(y_train)
            self._rng = np.random.RandomState(self.random_state)
        elif self.strategy == "constant":
            # Checks if the constant value is present in the training labels
            if self.constant not in y_train:
                raise ValueError(f"The constant value '{self.constant}' is not in the training labels.")
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

    def predict(self, x_test: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        # Returns predictions for the test data according to the chosen strategy
        n = x_test.shape[0]

        if self.strategy == "most_frequent":
            # Always predicts the most frequent class
            return np.full(n, self._most_frequent)
        if self.strategy == "uniform":
            # Predicts randomly one of the classes seen during training
            return self._rng.choice(self._unique_values, size=n)
        if self.strategy == "constant":
            # Always predicts the constant value
            return np.full(n, self.constant)
        raise ValueError(f"Unknown strategy '{self.strategy}'")
