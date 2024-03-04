import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

#Delete irrelevant columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Add the following day's price as a column
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# If the price has gone up, represent target as a 1, if it has gone down, represent target as 0
sp500["Target"] = (sp500["Tomorrow"]) > sp500["Close"].astype(int)

# Remove data before 1990, to only train the model on most recent/relevant data
sp500 = sp500.loc["1990-01-01":].copy()

# Train many individual decision trees with randomized params, and average the result to avoid overfitting
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Split up training and testing data, last 100 observations will be test data
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Train the model based off the following predictors (features)
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

print(f'Initial Accuracy: {precision_score(test["Target"], preds)}') ## "Initial Accuracy: 0.5588235294117647"

# Add more features (columns) to increase accuracy

# Define periods of days of which we want to compare
horizons = [2,5,60,250,1000]
new_predictors = []

# Loop through each time period of interest
for horizon in horizons:
    # Calculate the rolling average up until the current period
    rolling_averages = sp500.rolling(horizon).mean()
    
    # Add the rolling average as a ratio to the current day close
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    # Add the trend of the number of days the stock has increased over the current period
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]

# Remove observations with empty columns unless the empty column is "Tomorrow"
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

# Redefine the test and training data with the added features
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Retrain the model with the new predictors (features)
predictors += new_predictors
model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

print(f'Final Accuracy: {precision_score(test["Target"], preds)}') ## "Final Accuracy: 0.5757575757575758"