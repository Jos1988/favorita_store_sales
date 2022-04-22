import numpy as np
import pandas as pd

from definitions import TRAIN_FILE


print('Loading train data...')
train_df = pd.read_csv(TRAIN_FILE)
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['store_nbr'] = train_df['store_nbr'].astype('category')

# remove outliers.
train_df['sales'] = train_df['sales'].clip(upper=2500)
train_df['onpromotion'] = train_df['onpromotion'].clip(upper=2500)

# data_by_store = []
# for store_nbr, store_df in train_df[['date', 'store_nbr', 'sales', 'onpromotion']].groupby('store_nbr'):
#     store_df = store_df.groupby('date').sum()
#     store_df = store_df[['sales', 'onpromotion']].rename(columns={'sales': 'sales_' + str(store_nbr), 'onpromotion': 'onpromotion_' + str(store_nbr)})
#     data_by_store.append(store_df)
#
# grouped_by_store = pd.concat(data_by_store, axis=1)
# grouped_by_store[grouped_by_store < 1] = np.NAN
#
# # correlate onpromotion and sales per product family, try shifted.
# # filter dates after august 2014 as no promotions where recorded before mid 2014.
# promo_sales = grouped_by_store.loc[grouped_by_store.index > '2014-08-01']
# # drop columns with a lot of missing data, these are stores that weren't open during the the full period.
# na_thresh = len(promo_sales) * 0.9
# promo_sales = promo_sales.dropna(thresh=na_thresh, axis=1)
#
# for shift_n in range(0, 31):
#     promo_sales_correlation_results = []
#     for column_name in promo_sales.columns:
#         column_type, store_nbr = column_name.split('_')
#         if column_type == 'onpromotion':
#             continue
#
#         # Shift sales figures n days into the future.
#         shifted_sales = promo_sales[column_name].shift(shift_n)[shift_n:]
#         # cut promo figure to size of shifted sales figures.
#         cut_promo = promo_sales[shift_n:]['onpromotion_' + store_nbr]
#         # correlate current promo figures with future sales figures.
#         promo_sales_correlation = shifted_sales.corr(cut_promo)
#         # print(f'correlation for store {store_nbr: >3} is {promo_sales_correlation:>5.2f}') # uncomment to see every result.
#         promo_sales_correlation_results.append(promo_sales_correlation)
#
#     print(f'average_correlation: {sum(promo_sales_correlation_results)/len(promo_sales_correlation_results):.2f} (shifted {shift_n})')


data_by_store = []
for store_nbr, store_df in train_df[['date', 'family', 'sales', 'onpromotion']].groupby('family'):
    raw = store_df
    store_df = store_df.groupby('date').sum()
    store_df = store_df[['sales', 'onpromotion']].rename(columns={'sales': 'sales_' + str(store_nbr), 'onpromotion': 'onpromotion_' + str(store_nbr)})
    data_by_store.append(store_df)

grouped_by_store = pd.concat(data_by_store, axis=1)
grouped_by_store[grouped_by_store < 1] = np.NAN

# correlate onpromotion and sales per product family, try shifted.
# filter dates after august 2014 as no promotions where recorded before mid 2014.
promo_sales = grouped_by_store.loc[grouped_by_store.index > '2014-08-01']
# drop columns with a lot of missing data, these are stores that weren't open during the full period.
# na_thresh = len(promo_sales) * 0.9
# promo_sales = promo_sales.dropna(thresh=na_thresh, axis=1)
promo_sales = promo_sales.fillna(0)

for shift_n in range(0, 31):
    promo_sales_correlation_results = []
    for column_name in promo_sales.columns:
        column_type, store_nbr = column_name.split('_')
        if column_type == 'onpromotion':
            continue

        # Shift sales figures n days into the future.
        shifted_sales = promo_sales[column_name].shift(shift_n)[shift_n:]
        # cut promo figure to size of shifted sales figures.
        cut_promo = promo_sales[shift_n:]['onpromotion_' + store_nbr]
        # correlate current promo figures with future sales figures.
        promo_sales_correlation = shifted_sales.corr(cut_promo)
        if np.isnan(promo_sales_correlation):
            continue
        # print(f'correlation for store {store_nbr: >3} is {promo_sales_correlation:>5.2f}') # uncomment to see every result.
        promo_sales_correlation_results.append(abs(promo_sales_correlation))

    print(f'average_correlation: {sum(promo_sales_correlation_results)/len(promo_sales_correlation_results):.2f} (shifted {shift_n})')
