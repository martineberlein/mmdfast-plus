import pandas as pd
from pathlib import Path

from model.ml import RandomForestModel
from mmdfast.reduction import Reducer



RESOURCES_PATH = Path(__file__).parent.parent / "data_refactored"
SE_SUBJECTS = [
    "BRCTP",
    "java",
    "php",
    "python",
    "ruby",
]

KAGGLE_SUBJECTS = [
    "bank",
    "hotel_bookings",
    "job_change",
    # "spam",
    "water_potability"
]


def load_dataframe(subject: str, suffix="csv"):
    df_path = RESOURCES_PATH / f"{subject}.{suffix}"
    return pd.read_csv(df_path)


def safe_series_to_file(series, series_name, prefix="feature_imp", suffix="csv"):
    file_name = f"{series_name}_{prefix}.{suffix}"

    # Turn Series into a DataFrame with proper column names
    df = series.rename("importance").rename_axis("feature").reset_index()

    df.to_csv(file_name, index=False)  # index is now a column
    return file_name


if __name__ == "__main__":

    for subject in ["python"]:
        df = load_dataframe(subject)
        # df = df.sample(frac=0.03, random_state=12345678).reset_index(drop=True)

        model = RandomForestModel()
        model.split_and_prepare_data(df, test_size=0.5)
        model.train()
        model.evaluate()

        mispredictions_label = model.get_mispredicted_dataframe()

        reducer_rf = Reducer(top_n=3, random_state=42)
        importances = reducer_rf.fit(model.X_test, mispredictions_label)

        safe_series_to_file(importances, subject)
