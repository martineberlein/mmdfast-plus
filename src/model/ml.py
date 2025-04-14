import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseClassifierModel:
    """
    Base class providing common functionalities for training and evaluating classifiers.

    This class offers methods to load data, split and standardize it, train a model,
    evaluate its accuracy, and extract correct/mispredicted test samples.

    Subclasses must assign a specific classifier instance to `self.model` in their
    constructors.
    """

    def __init__(self):
        self.model = None  # To be set by subclasses
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.X_test_not_transformed = None  # copy of unscaled test data
        self.y_train = None
        self.y_test = None
        logger.info("BaseClassifierModel initialized.")

    @staticmethod
    def _load_data(csv_path):
        """
        Load dataset from a CSV file.

        Parameters:
            csv_path (str): The path to the CSV file.
        Returns:
            tuple:
                X (pd.DataFrame): Feature columns.
                y (pd.Series): Target values. The CSV must have an "Outcome" column.
        Raises:
            RuntimeError: If the file cannot be read or the "Outcome" column is missing.
        """
        try:
            logger.info("Loading data from file: %s", csv_path)
            df = pd.read_csv(csv_path)
            if "Outcome" not in df.columns:
                raise ValueError("CSV data must contain an 'Outcome' column.")
            X = df.drop("Outcome", axis=1)
            y = df["Outcome"]
            logger.info("Data loaded successfully. Dataset shape: %s", df.shape)
            return X, y
        except Exception as e:
            logger.error("Data loading error: %s", e)
            raise RuntimeError(f"Data loading error: {e}")

    def split_and_prepare_data(self, csv_path, test_size=0.3):
        """
        Load the dataset from CSV, split it into training and test sets, and standardize the features.

        Parameters:
            csv_path (str): Path to the CSV file.
            test_size (float): Fraction of data used for testing (default is 0.3).
        """
        logger.info("Starting data splitting and preprocessing.")
        X, y = self._load_data(csv_path)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )
        logger.info("Data split complete. Train shape: %s, Test shape: %s",
                    self.X_train.shape, self.X_test.shape)

        # Save original test data before scaling
        self.X_test_not_transformed = self.X_test.copy()

        # Standardize features using a StandardScaler (fit on training and transform both sets)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        logger.info("Data scaling complete.")

        # Convert numpy arrays back to DataFrames preserving feature names
        feature_names = pd.read_csv(csv_path).drop("Outcome", axis=1).columns
        self.X_train = pd.DataFrame(self.X_train, columns=feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
        logger.info("Converted arrays back to DataFrame format.")

    def train(self):
        """
        Train the classifier using the training data.
        """
        if self.model is None:
            logger.error("No model instance available. Please set a valid classifier in the subclass.")
            raise ValueError("Classifier model is not initialized.")
        logger.info("Training model: %s", type(self.model).__name__)
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training complete.")

    def evaluate(self):
        """
        Evaluate the trained classifier on the test set and print its accuracy.
        """
        logger.info("Evaluating model performance.")
        y_pred = self.model.predict(self.X_test)
        accuracy = (self.y_test == y_pred).mean()
        logger.info("Accuracy: %.2f%%", accuracy * 100)
        print(f"Accuracy: {accuracy:.2%}")

    def get_correct_wrong(self):
        """
        Obtain test samples with correct and mispredicted predictions.

        Returns:
            tuple:
                correct_predictions (pd.DataFrame): Samples that were predicted correctly.
                false_predictions (pd.DataFrame): Samples that were mispredicted.
        """
        logger.info("Extracting correct and mispredicted samples.")
        y_pred = self.model.predict(self.X_test)
        X_test_copy = self.X_test.copy()
        original_dtypes = self.X_test.dtypes

        # Reverse the scaling to recover the original feature values
        X_test_inversed = self.scaler.inverse_transform(X_test_copy)
        X_test_inversed = pd.DataFrame(X_test_inversed, columns=X_test_copy.columns)
        for col in X_test_inversed.columns:
            X_test_inversed[col] = X_test_inversed[col].astype(original_dtypes[col])

        # Append actual and predicted outcomes
        X_test_inversed["Actual"] = self.y_test.values
        X_test_inversed["Predicted"] = y_pred

        # Split DataFrame into correct and mispredicted samples
        correct_predictions = X_test_inversed[X_test_inversed["Actual"] == X_test_inversed["Predicted"]]
        false_predictions = X_test_inversed[X_test_inversed["Actual"] != X_test_inversed["Predicted"]]
        logger.info("Correct predictions: %d, Mispredictions: %d",
                    correct_predictions.shape[0], false_predictions.shape[0])
        return correct_predictions, false_predictions

    def get_mispredicted_indices(self):
        """
        Retrieve the indices of mispredicted samples in the test set.

        Returns:
            pd.Index: Indices where the classifier prediction is incorrect.
        """
        logger.info("Fetching mispredicted sample indices.")
        y_pred = self.model.predict(self.X_test)
        mispredicted = self.X_test.index[self.y_test != y_pred]
        logger.info("Total mispredicted samples: %d", len(mispredicted))
        return mispredicted

    def show_feature_importances(self):
        """
        Plot feature importances if the underlying classifier supports it.

        Logs a warning if the classifier does not implement feature_importances_.
        """
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("%s does not support feature importances.", type(self.model).__name__)
            return

        logger.info("Plotting feature importances for %s", type(self.model).__name__)
        importances = self.model.feature_importances_
        features = self.X_train.columns
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importances")
        plt.show()


class RandomForestModel(BaseClassifierModel):
    """
    A classifier model using the RandomForestClassifier.
    """

    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(random_state=0)
        logger.info("Initialized RandomForestClassifier.")


class DecisionTreeModel(BaseClassifierModel):
    """
    A classifier model using the DecisionTreeClassifier.
    """

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(random_state=0)
        logger.info("Initialized DecisionTreeClassifier.")


class SVMModel(BaseClassifierModel):
    """
    A classifier model using Support Vector Machine (SVC).
    """

    def __init__(self):
        super().__init__()
        self.model = SVC()
        logger.info("Initialized SVM (SVC).")


if __name__ == "__main__":
    csv_path = "../../data/diabetes.csv"

    logger.info("Starting Random Forest model training...")
    rf = RandomForestModel()
    rf.split_and_prepare_data(csv_path, test_size=0.3)
    rf.train()
    rf.evaluate()
    rf_correct, rf_wrong = rf.get_correct_wrong()
    print("Random Forest - Correct Predictions (first few rows):")
    print(rf_correct.head())
    print("\nRandom Forest - Mis-Predictions (first few rows):")
    print(rf_wrong.head())

    logger.info("Starting Decision Tree model training...")
    dt = DecisionTreeModel()
    dt.split_and_prepare_data(csv_path, test_size=0.3)
    dt.train()
    dt.evaluate()
    dt_correct, dt_wrong = dt.get_correct_wrong()
    print("Decision Tree - Correct Predictions (first few rows):")
    print(dt_correct.head())
    print("\nDecision Tree - Mis-Predictions (first few rows):")
    print(dt_wrong.head())

    logger.info("Starting SVM model training...")
    svm = SVMModel()
    svm.split_and_prepare_data(csv_path, test_size=0.3)
    svm.train()
    svm.evaluate()
    svm_correct, svm_wrong = svm.get_correct_wrong()
    print("SVM - Correct Predictions (first few rows):")
    print(svm_correct.head())
    print("\nSVM - Mis-Predictions (first few rows):")
    print(svm_wrong.head())

    # Optionally, retrieve mispredicted indices for one of the models
    rf_mispred_indices = rf.get_mispredicted_indices()
    logger.info("Random Forest - Mispredicted indices: %s", list(rf_mispred_indices))
    print("\nRandom Forest - Indices of mispredicted samples:")
    print(rf_mispred_indices)
