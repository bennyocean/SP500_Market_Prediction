# Introduction

This project is focused on predicting market movements with an emphasis on the S&P 500 index, utilizing Python and various data science techniques to uncover insights and forecast market trends. The project involves fetching real-time market data, cleaning, and preprocessing this data for analysis, and employing visualization techniques to identify patterns and trends.

The detailed analysis can be found in the [SP500Market_Prediction](./).

This market prediction endeavor was conducted as part of an ongoing exploration into financial market analysis, aimed at leveraging machine learning for better understanding market dynamics.

# Background

As a data scientist with a keen interest in finance, this project served as a platform to deepen my expertise in financial data analysis. The project offered an opportunity to enhance skills in data manipulation, time series analysis, and predictive modeling within the financial markets context.

The data was sourced from Yahoo Finance, utilizing the `yfinance` library for Python.

# Tools I Used

The project made use of several key tools to facilitate the analysis and prediction of market trends:

- **Python**
- **Jupyter Notebook**
- **Pandas** for data manipulation and analysis
- **Matplotlib** and **Seaborn** for data visualization
- **RandomForestClassifier** for modelling

# The Analysis

### 1. Data Retrieval and Cleaning
The market data for the S&P 500 was fetched using `yfinance`, followed by preprocessing steps such as setting the datetime format for the index and cleaning the data for any anomalies or outliers.

### 2. Exploratory Data Analysis (EDA)
Visual analysis was conducted to examine the closing prices over time, identifying potential patterns or trends that could inform the predictive models.

### 3. Model Fitting

In the predictive modeling phase, I selected the ```RandomForestClassifier``` due to its robustness and ability to handle non-linear relationships within the S&P 500 market data. This model is particularly suited for financial time series data, as it can effectively manage the complexities and inherent volatility of market prices, offering insightful predictions on market trends and movements.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
```

### 4. Prediction & Evaluation

In the prediction phase, I leveraged a custom predict function to fit the ```RandomForestClassifier``` model on the training dataset and then predict market movements on the test dataset. The backtest function further facilitates a rolling window approach to testing our model, allowing us to iteratively train and predict over different segments of the data. This method is crucial for evaluating the model's performance in a realistic, time-consistent manner, closely mimicking how predictions would be made in real-time financial market analysis.

```python
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
```

```python
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)
```

```python
precision_score(predictions["Target"], predictions["Predictions"])
```

The precision score resulted in 0.5228013029315961.


### 5. Adding New Features & Re-Evaluation

For adding new features, I utilized rolling windows to generate new predictors, capturing both short-term and long-term market trends. This approach involved calculating rolling averages and trends over specified horizons, enhancing our model's ability to discern patterns in the S&P 500's movements by incorporating a broader context of historical data.

```python
horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]
```

The inclusion of new features derived from rolling windows improved the model's precision score from 0.5228 to 0.5390, indicating a slight enhancement in its performance on the S&P 500 market data.

# Conclusion

### What I have learned

## What I Have Learned

This project has enhanced my understanding and skills in several key areas:
- **Financial Data**: Learned the complexities of financial data, including volatility and economic cycles.
- **Time Series Analysis**: Improved handling of time series data and the significance of temporal features.
- **Random Forests**: Valued RandomForestClassifier for its robustness and effectiveness with stock market data.
- **Feature Engineering**: Realized the impact of rolling windows in capturing market trends to boost model accuracy.
- **Model Evaluation**: Grasped the importance of backtesting for realistic assessment of model performance.
- **Precision in Predictions**: Acknowledged that slight improvements in precision can significantly influence financial decisions.

### Insights and Findings
A summary of key insights and findings from the analysis, highlighting any significant patterns, trends, or anomalies identified in the market data, and how these inform the predictive modeling approach.

### Closing Thoughts
Reflecting on the project, the process of analyzing and predicting market trends underscores the importance of rigorous data analysis and the potential of machine learning in financial forecasting. The insights gained from this project contribute to a deeper understanding of market dynamics, offering valuable perspectives for future analysis in financial markets.





For feature engineering, I utilized rolling windows to generate new predictors, capturing both short-term and long-term market trends. This approach involved calculating rolling averages and trends over specified horizons, enhancing our model's ability to discern patterns in the S&P 500's movements by incorporating a broader context of historical data.
