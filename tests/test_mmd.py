import unittest
import pandas as pd

from mmd.diagnoser import discover
from model.ml import RandomForestModel
from mmdfast.utils import classify_columns

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
        model_correct, model_wrong = model.get_correct_wrong()

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


if __name__ == "__main__":
    unittest.main()
