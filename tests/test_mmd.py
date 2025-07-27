import unittest
import pandas as pd

from mmd.diagnoser import discover
from model.ml import RandomForestModel

from mmdfast.utils import classify_columns
from mmdfast.reduction import Reducer

from util import RESOURCES_PATH


class TestMMD(unittest.TestCase):

    def test_mmd(self):

        df_path = RESOURCES_PATH / "diabetes.csv"
        df = pd.read_csv(df_path)

        # Classify columns in the DataFrame
        column_types = classify_columns(df.drop(columns=["Outcome"]))
        print("Column types:", column_types)

        model = RandomForestModel()
        model.split_and_prepare_data(df_path, test_size=0.2)
        model.train()
        model.evaluate()

        mispredictions = model.get_mispredicted_dataframe()

        df_mis = pd.concat([model.X_test, mispredictions], axis=1)
        result = discover(
            df_mis,
            ("misprediction", True),
            relevant_attributes=column_types,
            coverage=0.6,
            allow_disjunctions=True,
        )

        result_ = result.dataframe()
        print(result_[['ruleset', 'precision', 'recall']])

    def test_mmd_fr(self):
        df_path = RESOURCES_PATH / "diabetes.csv"
        df = pd.read_csv(df_path)

        # Classify columns in the DataFrame
        column_types = classify_columns(df.drop(columns=["Outcome"]))
        print("Column types:", column_types)

        model = RandomForestModel()
        model.split_and_prepare_data(df_path, test_size=0.2)
        model.train()
        model.evaluate()

        reducer_rf = Reducer(top_n=3, random_state=42)
        rf_wrong_reduced = reducer_rf.fit_transform(model.X_test, model.y_test)
        influential_features = reducer_rf.selected_features
        print("Selected features:", influential_features)

        df_reduced = model.X_test.loc[:, influential_features]
        column_types_reduced = {i: j for i, j in column_types.items() if i in influential_features}
        mispredictions = model.get_mispredicted_dataframe()

        df_mis = pd.concat([df_reduced, mispredictions], axis=1)
        result = discover(
            df_mis,
            ("misprediction", True),
            relevant_attributes=column_types_reduced,
            coverage=0.6,
            allow_disjunctions=True,
        )

        result_ = result.dataframe()
        print(result_[['ruleset', 'precision', 'recall']])


if __name__ == "__main__":
    unittest.main()
