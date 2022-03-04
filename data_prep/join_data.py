import pandas as pd
from definitions import TRAIN_FILE, STORES_FILE, OIL_FILE, HOLIDAY_EVENTS_FILE, TRANSACTIONS_FILE, CACHE_FOLDER

print('Loading train data...')
train_df = pd.read_csv(TRAIN_FILE)
print('Loading stores data...')
stores_df = pd.read_csv(STORES_FILE)
print('Loading oil data...')
oil_df = pd.read_csv(OIL_FILE)
print('Loading holiday data...')
holiday_df = pd.read_csv(HOLIDAY_EVENTS_FILE)
print('Loading transaction data...')
transaction_df = pd.read_csv(TRANSACTIONS_FILE)

data = train_df.merge(stores_df, on='store_nbr', how='left')
print('Missing dates in oil data so oil has a lot of nans (no oil prices in the weekend.)')
data = data.merge(oil_df, on='date', how='left')

print('add holiday data')
data = data.merge(holiday_df, on='date', how='left')

print('Missing data in transaction data a lot of nans')
data = data.merge(transaction_df, on=('date', 'store_nbr'), how='left')

print('date to datetime')
data['date'] = pd.to_datetime(data['date'])

print('add weekdays')
data['weekday'] = data['date'].dt.day_name()

data = data.rename(columns={'type_x': 'store_type', 'type_y': 'holiday_type', 'transactions': 'total_transactions', 'locale': 'hld_locale',
                            'locale_name': 'hld_locale_name', 'description': 'hld_description', 'transferred': 'hld_transferred'})

print('check if data is missing because it is weekend.')
for day, group in data.groupby('weekday'):
    print()
    print(day)
    print(f'oil nans: {group.dcoilwtico.isna().sum() / len(group)}')
    print(f'transactions nans: {group.total_transactions.isna().sum() / len(group)}')

print('Oil data missing on the weekends.')

data[100_000:110_000].to_csv(CACHE_FOLDER + '/data_cache_sample.csv')
data.to_pickle(CACHE_FOLDER + '/data_cache.pkl')


