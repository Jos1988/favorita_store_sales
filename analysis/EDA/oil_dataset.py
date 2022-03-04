import pandas as pd
from matplotlib import pyplot as plt

from definitions import OIL_FILE
from EDA.analysis import describe_dataset, check_dates


print('Loading oil data...')
oil_df = pd.read_csv(OIL_FILE)
describe_dataset(oil_df)
print('Each row represents the oil price on one day.')
print()

print()
dates_freqs = oil_df['date'].value_counts()
print(f'Each date occurs {set(dates_freqs.values)} times')
print(f"has nans: {oil_df['date'].hasnans}")

print()
check_dates(oil_df['date'])

print()
print(F"Price ranges from {oil_df['dcoilwtico'].min()} to {oil_df['dcoilwtico'].max()} with {len(oil_df['dcoilwtico'].unique())} unique values.")
print(oil_df['dcoilwtico'].describe())

print('')
plt.plot(oil_df['date'], oil_df['dcoilwtico'])
plt.show()
plt.close()
after_drop_prices = oil_df[oil_df['dcoilwtico'] < 70.0]

print('note the prices drop after the ' + after_drop_prices['date'].values[0] + 'from around 100 too around 40!')
print(f"has nans: {oil_df['dcoilwtico'].hasnans}")
print(f"number of nans: {oil_df['dcoilwtico'].isna().sum()}")
