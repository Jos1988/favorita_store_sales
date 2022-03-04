import pandas as pd
from matplotlib import pyplot as plt

from definitions import TRAIN_FILE
from EDA.analysis import describe_dataset, check_dates


print('Loading train data...')
train_df = pd.read_csv(TRAIN_FILE)
describe_dataset(train_df)
print('Each row represents the daily sales of one product type in one store.')
print()

print(f"Ids range from {min(train_df['id'])} to {max(train_df['id'])}")

print()
dates_freqs = train_df['date'].value_counts()
print(f'Each date occurs {set(dates_freqs.values)} times')
print(f"has nans: {train_df['date'].hasnans}")

print()
check_dates(train_df['date'])

print()
print(f"There are {len(train_df['store_nbr'].unique())} store numbers")
store_freqs = train_df['store_nbr'].value_counts()
print(store_freqs.keys().sort_values())
print(f'Each store nbr occurs {set(store_freqs.values)} times')
print(f"has nans: {train_df['store_nbr'].hasnans}")

print()
print(f"There are {len(train_df['family'].unique())} product families")
family_freqs = train_df['family'].value_counts()
print(family_freqs.keys().sort_values())
print(f'Each family occurs {set(family_freqs.values)} times')
print(f"has nans: {train_df['family'].hasnans}")

print()
print(F"on promotion ranges from {train_df['onpromotion'].min()} to {train_df['onpromotion'].max()} with {len(train_df['onpromotion'].unique())} unique values.")
print(f"The lowest values occur most (top 5 %)L")
print(train_df['onpromotion'].value_counts(normalize=True)[:5] * 100)
train_df['onpromotion'].hist(bins=100)
plt.title('on promotion value freqs')
plt.show()
print(f"has nans: {train_df['onpromotion'].hasnans}")

print()
print(F"Sales ranges from {train_df['sales'].min()} to {train_df['sales'].max()} with {len(train_df['sales'].unique())} unique values.")
print(f"The lowest values occur most (top 5 %)L")
print(train_df['sales'].value_counts(normalize=True)[:5] * 100)
train_df['sales'].hist(bins=100)
plt.title('sales value freqs')
plt.show()
print(f"has nans: {train_df['sales'].hasnans}")

