import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import Grouper
from definitions import CACHE_FOLDER


def process_interval_groups(interval_groups, do_pad: bool = True):
    max_len_interval_group = max([len(group.values.flatten()) for name, group in interval_groups])

    flat_padded_interval_groups = {}
    for name, group in interval_groups:
        interval_sales = group.values.flatten()
        if do_pad:
            interval_sales = np.pad(interval_sales, (0, max_len_interval_group - len(interval_sales)))
        flat_padded_interval_groups[str(name)] = interval_sales

    return pd.DataFrame(flat_padded_interval_groups)


def plot_by_interval(interval: str, type: str = 'line', show_nth: int = None, do_pad: bool = True):
    interval_groups = series.groupby(Grouper(freq=interval))
    interval_groups = [(name, group) for name, group in interval_groups]
    processed_interval_groups = process_interval_groups(interval_groups, do_pad=do_pad)
    mean_interval = processed_interval_groups.mean(axis=1)

    if show_nth:
        processed_interval_groups = process_interval_groups(interval_groups[::show_nth], do_pad=do_pad)

    processed_interval_groups['mean'] = mean_interval
    if type == 'line':
        processed_interval_groups.plot(subplots=True, legend=False, title='sales plotted by ' + interval, figsize=(5, len(processed_interval_groups.columns)/2))
    elif type == 'box':
        processed_interval_groups.boxplot(rot=-45)
        plt.suptitle('Sales in boxplots by ' + interval)
    elif type == 'map':
        plt.matshow(processed_interval_groups.T, interpolation=None, aspect='auto')
        plt.suptitle('Sales heatmap, every row is a(n) ' + interval)

    plt.show()
    plt.close()


# https://machinelearningmastery.com/time-series-data-visualization-with-python/
data: pd.DataFrame = pd.read_pickle(CACHE_FOLDER + '/data_cache.pkl')
data['k_sales'] = data['sales'] / 1000
series: pd.DataFrame = data[['date', 'k_sales']].groupby('date').sum()

# clip outliers
series = series.clip(upper=1500, lower=200)

print('# draw line diagram')
# series.plot(style='k.', title='daily overall sales')
# plt.show()
# plt.close()

print('# plot values distribution')
# series.hist(bins=100)
# plt.show()
# plt.close()

print('# Create line diagrams to show trends.')
plot_by_interval('A')
print('# Sales peak in the middle and at the end of the year. holidays?')
plot_by_interval('M', show_nth=4)
print('# Sales descend gradually during the month, (see average)')
plot_by_interval('W', show_nth=10)
print('# Sales peek during the weekends.')

print('# Create boxplots to show trends.')
# plot_by_interval('A', type='box', do_pad=True)
# plot_by_interval('M', type='box', show_nth=4, do_pad=True)
# plot_by_interval('W', type='box', show_nth=10, do_pad=True)


print('# Create heatmaps to show trends.')
# plot_by_interval('A', type='map')
# plot_by_interval('M', type='map')
# plot_by_interval('W', type='map')
