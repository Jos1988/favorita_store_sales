# Store Sales - Time Series Forecasting
Personal side project for working with time series data.

## Getting started
### 1. Get the data
Download the project data files to the `/data` folder. Either manually from the [Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) website or using the Kaggle API: `kaggle competitions download -c store-sales-time-series-forecasting`.

### 2. Install dependencies
Install dependencies from `/requirements.txt` (generate using `pip freeze`) or install manually as needed.

### 3. Prepare data
todo: integrate this step in scripts.

Run `data_prep/join_data.py` to create the required datasets. 

### 4. Add src
Add the `/src` folder to your python path (in Pycharm right click on folder > Mark Directory as > Source Root)

### 5. Run code
Run scripts from following folders:
- `/analysis/EDA` (exploratory data analysis)
- `/analysis/timeseries` (analysing the data as a time series)
- `/forecasting` (various forecasting methods)

## Kaggle description: Use machine learning to predict grocery sales

> link: https://www.kaggle.com/c/store-sales-time-series-forecasting
> 
> Kaggle API ref: store-sales-time-series-forecasting

### Goal of the Competition
In this “getting started” competition, you’ll use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer.

Specifically, you'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores. You'll practice your machine learning skills with an approachable training dataset of dates, store, and item information, promotions, and unit sales.

> ### Get Started
> We highly recommend the [Time Series course](https://www.kaggle.com/learn/time-series), which walks you through how to make your first submission. The lessons in this course are inspired by winning solutions from past Kaggle time series forecasting competitions.

### Context
Forecasts aren’t just for meteorologists. Governments forecast economic growth. Scientists attempt to predict the future population. And businesses forecast product demand—a common task of professional data scientists. Forecasts are especially relevant to brick-and-mortar grocery stores, which must dance delicately with how much inventory to buy. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leading to lost revenue and upset customers. More accurate forecasting, thanks to machine learning, could help ensure retailers please customers by having just enough of the right products at the right time.

Current subjective forecasting methods for retail have little data to back them up and are unlikely to be automated. The problem becomes even more complex as retailers add new locations with unique needs, new products, ever-transitioning seasonal tastes, and unpredictable product marketing.

### Potential Impact
If successful, you'll have flexed some new skills in a real world example. For grocery stores, more accurate forecasting can decrease food waste related to overstocking and improve customer satisfaction. The results of this ongoing competition, over time, might even ensure your local store has exactly what you need the next time you shop.