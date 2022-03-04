import pandas as pd


def describe_dataset(df: pd.DataFrame):
    columns = df.columns.tolist()
    print(columns)
    if 'id' in columns:
        columns.remove('id')
    print()
    print(f'length {len(df)}')
    print(df[columns].describe())


def check_dates(series: pd.Series):
    dates = pd.to_datetime(series.value_counts().keys())
    range = pd.date_range(start=min(dates), end=max(dates))
    diff = range.difference(other=dates)
    print(f'Dates range from {min(dates)} until {max(dates)}  with {len(diff)} missing days.')
    if len(diff) > 0:
        print(diff)
