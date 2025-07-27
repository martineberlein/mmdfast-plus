from typing import Dict
import pandas as pd
import numpy as np


def test_mispredict(XData, yData, model):
    inputs = XData.values
    ground_truth = yData.values.ravel()

    # predict label for test inputs
    predictions = model.predict(inputs)

    # check for mispredictions
    mispredictions = np.where(ground_truth == predictions, 0, 1)

    # convert to pandas dataframe
    df_mispredictions = pd.DataFrame({'misprediction': mispredictions})

    return df_mispredictions



def classify_columns(
    df: pd.DataFrame,
    discrete_threshold: int = 5
) -> Dict[str, str]:
    """
    Classify each column in df as:
      - 'D' (Discrete): categorical/text or numeric with few distinct values
      - 'I' (Int): integer dtype with many distinct values
      - 'C' (Continuous): float dtype with many distinct values

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns you want to classify.
    discrete_threshold : int, optional
        The maximum number of unique values for a numeric column
        to still be considered discrete (default: 20).

    Returns
    -------
    Dict[str, str]
        Mapping from column names to one of 'D', 'I', or 'C'.
    """
    result: Dict[str, str] = {}

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)

        # 1) object or categorical → discrete
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            result[col] = 'D'

        # 2) integer dtype
        elif pd.api.types.is_integer_dtype(series):
            if nunique <= discrete_threshold:
                result[col] = 'D'
            else:
                result[col] = 'I'

        # 3) float dtype
        elif pd.api.types.is_float_dtype(series):
            if nunique <= discrete_threshold:
                result[col] = 'D'
            else:
                result[col] = 'C'

        # 4) all others (e.g. datetime, bool) → treat as discrete by default
        else:
            result[col] = 'D'

    return result
