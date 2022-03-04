import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWM

from definitions import CACHE_FOLDER

data: pd.DataFrame = pd.read_pickle(CACHE_FOLDER + '/data_cache.pkl')
data['k_sales'] = data['sales'] / 1000
series: pd.DataFrame = data[['date', 'k_sales']].groupby('date').sum()
# series['date'] = pd.to_datetime(series['date'])
# series.set_index('date')

# clip outliers
series = series.clip(lower=200, upper=1500)
#
# print('# draw line diagram')
# plt.figure(figsize=[25, 5])
# plt.plot(series, 'black', alpha=0.1, label='daily sales')
# for w in [7, 50]:
#     ma = series.rolling(window=w).mean()
#     plt.plot(ma, label='MA_' + str(w) + '_days')
#
# plt.title('moving averages.')
# plt.legend()
# plt.show()
# plt.close()
#
# print('# lagging')
# # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=[25, 15])
# fig, [[ax1L, ax1R], [ax2L, ax2R], [ax3L, ax3R], [ax4L, ax4R], [ax5L, ax5R]] = plt.subplots(5, 2, figsize=[25, 15], gridspec_kw={'width_ratios': [4, 1]})
# fig.suptitle('lag')
# ax1L.plot(series)
# ax1L.title.set_text('series data')
# ax1R.scatter(series.sample(n=1000, random_state=1), series.sample(n=1000, random_state=1), s=1)
#
# def shift(x, kwargs: dict):
#     new_index = x.name + pd.DateOffset(**kwargs)
#     if new_index not in series.index:
#         return None
#
#     return series.loc[new_index]
#
#
# lag_1 = series - series.shift(periods=1)
# ax2L.plot(lag_1)
# ax2L.title.set_text('lag 1 day')
# ax2R.scatter(lag_1.sample(n=1000, random_state=1), series.sample(n=1000, random_state=1), s=1)
#
# print('the line becomes much smoother when lagging data a week, suggesting that weekly trends play a major role.')
# lag_W = series - series.apply(lambda x: shift(x, {'weeks': 1}), axis=1)
# ax3L.plot(lag_W)
# ax3L.title.set_text('lag one week')
# ax3R.scatter(lag_W.sample(n=1000, random_state=1), series.sample(n=1000, random_state=1), s=1)
#
# lag_M = series - series.apply(lambda x: shift(x, {'months': 1}), axis=1)
# ax4L.plot(lag_M)
# ax4L.title.set_text('lag one month')
# ax4R.scatter(lag_M.sample(n=1000, random_state=1), series.sample(n=1000, random_state=1), s=1)
#
# lag_Y = series - series.apply(lambda x: shift(x, {'years': 1}), axis=1)
# ax5L.plot(lag_Y)
# ax5L.title.set_text('lag one year')
# ax5R.scatter(lag_Y.sample(n=1000, random_state=1), series.sample(n=1000, random_state=1), s=1)
#
# plt.show()
# plt.close()
#
# print('exponential weighted average.')
# plt.figure(figsize=[25, 5])
# plt.plot(series, 'black', alpha=0.1, label='daily sales')
# for a in [0.01, 0.03, 0.1]:
#     esm = series.ewm(alpha=a).mean()
#     plt.plot(esm, label='alfa=' + str(a))
#
# plt.title('Exp weighted average.')
# plt.legend()
# plt.show()
# plt.close()

print('# Holt Winters exponential smoothing')
# https://towardsdatascience.com/holt-winters-exponential-smoothing-d703072c0572

series.index.freq = 'MS' #does not work

# split dataset at 2016-12-31 | 2017-01-01
train = series.iloc[:1457]
test = series.iloc[1457:]

holts_winters_model = HWM(train, trend='add', seasonal='mul', seasonal_periods=364)
holts_winters_fit = holts_winters_model.fit()

print(holts_winters_fit.summary())
holts_winters_forecast = holts_winters_fit.forecast(len(test))

#Replace integer index with datetime index.
holts_winters_forecast.index = test.index

fig = plt.figure(figsize=[20, 5])
fig.suptitle('Holt Winters 365 day forecast')
past, = plt.plot(train.index, train, 'b.-', label='Sales History')
future, = plt.plot(test.index, test, 'r.-', label='Real Sales')
predicted_future, = plt.plot(holts_winters_forecast.index, holts_winters_forecast, 'g.-', label='Forecast')
plt.legend(handles=[past, future, predicted_future])
plt.show()
plt.close()

fig = plt.figure(figsize=[20, 5])
fig.suptitle('Holt Winters 365 day forecast (only forecast)')
future, = plt.plot(test.index, test, 'r.-', label='Real Sales')
predicted_future, = plt.plot(holts_winters_forecast.index, holts_winters_forecast, 'g.-', label='Forecast')
plt.legend(handles=[future, predicted_future])
plt.show()
plt.close()


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# import numpy
# import matplotlib.pyplot as plt
# import pandas
# import math
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
#
# # fix random seed for reproducibility
# numpy.random.seed(7)
#
# # load the dataset
# dataframe = series
# dataset = dataframe.values
# dataset = dataset.astype('float32')
#
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
#
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print(len(train), len(test))
#
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)
#
# # reshape into X=t and Y=t+1
# look_back = 1
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
#
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
#
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
