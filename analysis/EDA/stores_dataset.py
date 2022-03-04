import pandas as pd
from matplotlib import pyplot as plt

from definitions import STORES_FILE
from EDA.analysis import describe_dataset


print('Loading stores data...')
stores_df = pd.read_csv(STORES_FILE)
describe_dataset(stores_df)
print('Each row represents a store.')
print()

print(f"Store numbers range from {min(stores_df['store_nbr'])} to {max(stores_df['store_nbr'])}")

print()
print(f'# there are {len(stores_df.city.unique())} different cities.')
city_vc = stores_df['city'].value_counts(normalize=True)
print('top 3')
print(f"{round(city_vc.values[0]*100, 2)}% of stores are in {city_vc.keys()[0]}")
print(f"{round(city_vc.values[1]*100, 2)}% of stores are in {city_vc.keys()[1]}")
print(f"{round(city_vc.values[2]*100, 2)}% of stores are in {city_vc.keys()[2]}")
print(f"has nans: {stores_df['city'].hasnans}")
stores_df['city'].hist()
plt.title('cities')
plt.xticks(rotation=60)
plt.show()

print()
print(f'# there are {len(stores_df.state.unique())} different states.')
state_vc = stores_df['state'].value_counts(normalize=True)
print('top 3')
print(f"{round(state_vc.values[0]*100, 2)}% of stores are in {state_vc.keys()[0]}")
print(f"{round(state_vc.values[1]*100, 2)}% of stores are in {state_vc.keys()[1]}")
print(f"{round(state_vc.values[2]*100, 2)}% of stores are in {state_vc.keys()[2]}")
print(f"has nans: {stores_df['state'].hasnans}")
stores_df['state'].hist()
plt.title('states')
plt.xticks(rotation=60)
plt.show()

print()
print(f'# there are {len(stores_df.type.unique())} different types.')
type_vc = stores_df['type'].value_counts(normalize=True)
print('top 3')
print(f"{round(type_vc.values[0]*100, 2)}% of stores are in {type_vc.keys()[0]}")
print(f"{round(type_vc.values[1]*100, 2)}% of stores are in {type_vc.keys()[1]}")
print(f"{round(type_vc.values[2]*100, 2)}% of stores are in {type_vc.keys()[2]}")
print(f"has nans: {stores_df['type'].hasnans}")
stores_df['type'].hist()
plt.title('types')
plt.xticks(rotation=60)
plt.show()


print()
print(f'# there are {len(stores_df.cluster.unique())} different clusters.')
cluster_vc = stores_df['cluster'].value_counts(normalize=True)
print('top 3')
print(f"{round(cluster_vc.values[0]*100, 2)}% of stores are in {cluster_vc.keys()[0]}")
print(f"{round(cluster_vc.values[1]*100, 2)}% of stores are in {cluster_vc.keys()[1]}")
print(f"{round(cluster_vc.values[2]*100, 2)}% of stores are in {cluster_vc.keys()[2]}")
print(f"has nans: {stores_df['cluster'].hasnans}")
stores_df['cluster'].hist()
plt.title('clusters')
plt.xticks(rotation=60)
plt.show()
