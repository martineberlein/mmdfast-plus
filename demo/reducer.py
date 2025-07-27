from mmdfast.logger import LOOGER
from mmdfast.reduction import Reducer

from model.ml import RandomForestModel


if __name__ == "__main__":
    csv_path = "../data/diabetes.csv"

    rf = RandomForestModel()
    rf.split_and_prepare_data(csv_path, test_size=0.3)
    rf.train()
    rf.evaluate()
    rf_correct, rf_wrong = rf.get_correct_wrong()

    rf_wrong_features = rf_wrong.drop(columns=["Actual", "Predicted"])
    rf_wrong_target = rf_wrong["Actual"]

    reducer_rf = Reducer(top_n=3, random_state=42)
    rf_wrong_reduced = reducer_rf.fit_transform(rf_wrong_features, rf_wrong_target)
    print("\nRandom Forest - Reduced mispredictions (first few rows):")
    print(rf_wrong_reduced.head())
    print("Selected features:", reducer_rf.selected_features)