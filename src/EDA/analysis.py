import pandas as pd
from pandas import DatetimeIndex


def describe_dataset(df: pd.DataFrame):
    columns = df.columns.tolist()
    print(columns)
    if 'id' in columns:
        columns.remove('id')
    print()
    print(f'length {len(df)}')
    print(df[columns].describe())


def get_missing_dates(series: pd.Series) -> DatetimeIndex:
    dates = pd.to_datetime(series.unique())
    range = pd.date_range(start=min(dates), end=max(dates))
    return range.difference(other=dates)


def check_missing_dates(series: pd.Series):
    diff = get_missing_dates(series)
    print(f'Dates range from {min(series)} until {max(series)}  with {len(diff)} missing days.')
    if len(diff) > 0:
        print(diff)
