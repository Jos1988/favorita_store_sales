import pandas as pd
from matplotlib import pyplot as plt

from EDA.analysis import check_dates, describe_dataset
from definitions import HOLIDAY_EVENTS_FILE


print('Loading holiday data...')
holiday_df = pd.read_csv(HOLIDAY_EVENTS_FILE)
describe_dataset(holiday_df)
print('Each row represents a holiday.')
print()

check_dates(holiday_df['date'])
dates = pd.to_datetime(holiday_df['date'].value_counts().keys())
range = pd.date_range(start=min(dates), end=max(dates))

is_holiday = []
for date in range:
    if date in dates:
        is_holiday.append(1)
        continue
    is_holiday.append(0)

plt.plot(range, is_holiday)
plt.show()
plt.close()

print(f'{round(is_holiday.count(1)/ len(is_holiday), 2)}% is a holiday.')

print('transferred holidays:')
print(holiday_df['transferred'].value_counts())
print(holiday_df['transferred'].value_counts(normalize=True))
print(f"has nans: {holiday_df['transferred'].hasnans}")

print()
print(f"There are {len(holiday_df['type'].unique())} types")
type_freqs = holiday_df['type'].value_counts()
print(type_freqs.keys().sort_values())
print(f'type occurs {set(type_freqs.values)} times')
print(f"has nans: {holiday_df['type'].hasnans}")

