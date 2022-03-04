import pandas as pd
from matplotlib import pyplot as plt

from EDA.analysis import describe_dataset, check_dates

from definitions import TRANSACTIONS_FILE


df = pd.read_csv(TRANSACTIONS_FILE)
describe_dataset(df)

print()
print('Dataset contains the number of transactions by store by day.')

df.transactions.hist(bins=100)
plt.show()

print(df.transactions.describe())

print(f'has nans: {df.transactions.hasnans}')

print()
dates_freqs = df['date'].value_counts()
print(f'Each date occurs {set(dates_freqs.values)} times')
print(f"has nans: {df['date'].hasnans}")

print()
check_dates(df['date'])

print('note not every day / store is present in the dataset. probably because 0 transactions are not recorded.')

for store_nbr, store_grp in df.groupby('store_nbr'):
    print()
    print(f'store number: {store_nbr}')
    check_dates(store_grp['date'])
