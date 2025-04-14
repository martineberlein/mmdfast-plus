import os
import logging
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from model.ml import SVMModel, RandomForestModel, DecisionTreeModel

# Configure logging for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Reducer:
    """
    A class for feature reduction based on feature importance provided by
    a Random Forest classifier.

    The Reducer uses a fitted RandomForestClassifier to compute the importance
    of each feature, and then selects a subset of features based on one of these strategies:
    """

    def __init__(self, top_n=None, threshold=None, random_state=0):
        """
        Initialize the Reducer.
        """
        self.top_n = top_n
        self.threshold = threshold
        self.random_state = random_state
        self.selected_features = None
        self.rf = RandomForestClassifier(random_state=self.random_state)
        logger.info("Reducer initialized with top_n=%s, threshold=%s, random_state=%d",
                    self.top_n, self.threshold, self.random_state)

    def fit(self, X, y):
        """
        Fit the Reducer using the Random Forest classifier to calculate feature importance.
        After fitting, the attribute `selected_features` contains the names of the features to be retained.
        """
        logger.info("Fitting the Reducer using RandomForestClassifier.")
        self.rf.fit(X, y)
        importances = pd.Series(self.rf.feature_importances_, index=X.columns)
        logger.info("Feature importance computed:\n%s", importances.sort_values(ascending=False))

        # Determine which features to select based on the provided criteria.
        if self.top_n is not None:
            # Select top_n features with highest importances.
            self.selected_features = importances.sort_values(ascending=False).head(self.top_n).index.tolist()
            logger.info("Selected top %d features: %s", self.top_n, self.selected_features)
        elif self.threshold is not None:
            # Select features with importance greater or equal to the threshold.
            self.selected_features = importances[importances >= self.threshold].index.tolist()
            logger.info("Selected features with importance >= %.3f: %s", self.threshold, self.selected_features)
        else:
            # Default selection: features with importance greater than or equal to mean importance.
            mean_importance = importances.mean()
            self.selected_features = importances[importances >= mean_importance].index.tolist()
            logger.info("No top_n or threshold provided. Using mean importance (%.3f). Selected features: %s",
                        mean_importance, self.selected_features)

        if not self.selected_features:
            logger.warning("No features were selected. Check your top_n or threshold parameters.")
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by reducing it to only the selected features.
        """
        if self.selected_features is None:
            logger.error("The reducer has not been fitted yet. Please call fit() before transform().")
            raise ValueError("Reducer is not fitted yet.")
        logger.info("Reducing data to the selected features: %s", self.selected_features)
        return X[self.selected_features]

    def fit_transform(self, X, y):
        """
        Fit the reducer to the data and transform X to only include the selected features.
        Returns:
            pd.DataFrame: Reduced DataFrame with only the selected features.
        """
        logger.info("Performing fit_transform on the data.")
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    csv_path = "../../data/diabetes.csv"

    logger.info("=== Random Forest Model Training ===")
    rf = RandomForestModel()
    rf.split_and_prepare_data(csv_path, test_size=0.3)
    rf.train()
    rf.evaluate()
    rf_correct, rf_wrong = rf.get_correct_wrong()
    print("Random Forest - Mis-Predictions (first few rows):")
    print(rf_wrong.head())

    rf_wrong_features = rf_wrong.drop(columns=["Actual", "Predicted"])
    rf_wrong_target = rf_wrong["Actual"]
    reducer_rf = Reducer(top_n=3, random_state=42)
    rf_wrong_reduced = reducer_rf.fit_transform(rf_wrong_features, rf_wrong_target)
    print("\nRandom Forest - Reduced mispredictions (first few rows):")
    print(rf_wrong_reduced.head())
    print("Selected features:", reducer_rf.selected_features)