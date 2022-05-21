import pandas as pd

from grouped_time_split import GroupTimeSeriesSplit


# https://github.com/labdmitriy/ml-lab/blob/master/ml_lab/model_selection.py
# https://www.kaggle.com/code/jorijnsmit/found-the-holy-grail-grouptimeseriessplit/notebook
def split_dataset(
    df_x: pd.DataFrame, groups, test_size: int, splits: int = 5
) -> tuple[tuple, tuple]:
    splitter = GroupTimeSeriesSplit(test_size=test_size, n_splits=splits)

    return splitter.split(df_x, groups=groups)
