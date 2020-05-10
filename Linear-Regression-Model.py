# =============================================================================
#                          IMPORT RELEVANT MODULES
# Import the following modules (Download is necessary so just pip install)
# Math is a module with advanced mathematics functions
# Pandas is a data science module
# Numpy is another data science module that is faster than pandas
# Pickle lets us save the trained model
# Matplotlib is a data visualization library
# =============================================================================
import math
import pandas as pd #Dataframes
import numpy as np #Objects for multi-dimensional arrays
from sklearn import preprocessing, model_selection, linear_model
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

# =============================================================================
#                             PREPARE THE DATA
# Then we define our dataframe so lets use Tesla Motors (TSLA) as an example
# We then define which columns we want to keep since some are useless
# At this point, no relationships have been established and we need to define
#them since simple ML algorithms like linear regression don't 'discover' it
# HML is the high minus low percent which you can think of as the daily
#volatility of the stock
# PC is the daily percent change of the stock, which is just (old - new)/old
# Redefine our dataframe to only include relevant information, again
# NEVER, NEVER, NEVER delete data. You can do one of two things:
#1. First do forward fill (ffill) then backward fill (bfill).
#Look at a graph and it will be clear why the order is important
#2. You could do df.fillna(-999999, inplace=True) and it will be treated as an 
#outlier. In this context, this method is better because you aren't creating
#data that isn't there (as in method 1)
# =============================================================================
df = pd.read_csv("TSLA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)
df =  df[['Open','High', 'Low', 'Adj Close']]
df['HML'] = (df['High'] - df['Low']) / df['Adj Close']
df['PC'] = (df['Adj Close'] - df['Open']) / df['Open']
df =  df[['Open','Adj Close', 'HML', 'PC']]
df.fillna(-999999, inplace = True)


# =============================================================================
#                               TRAIN THE MODEL
# Define the column data to be predicted
# Forecast_out is the amount of days by which we will be forecasting out by 5 
#days in this case
# The label column will store the data we will check our prediction against
# Define the input data (everything other than the 'label' column) and drop the
#last three rows
# Preprocessing ensures the data is randomized, normalized and unbiased
# Define a variable to store those 3 rows so we can check our prediction
# Define the output data (the 'Adj. Close' column)
# Just memorize the last line. 30% of the data is for testing
# =============================================================================
predict_column = 'Adj Close'
forecast_out = int(math.floor(0.02*len(df)))
df['label'] = df[predict_column].shift(-forecast_out)

x = np.array(df.drop(['label'], axis = 1))
x = x[: -forecast_out:]
x = preprocessing.scale(x)
prediction_checker = x[-forecast_out: ]

df.dropna(inplace = True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)


# =============================================================================
#                               TEST THE MODEL
# Creates an instance of the least squares linear regression algorithm
# The fit method fits the linear model with the trainging data for the x and y
#variables
# Test the accuracy. This is based on the test data. Accuracy isn't really a
#great measure of how good your model is since its based on existing data
# Drastically improves time-efficiency. Basically stores the model info so it
#doesn't have to be computed again. Remember to train bi-weekly (minimum)
# =============================================================================
LR = linear_model.LinearRegression()
LR.fit(x_train, y_train)

accuracy = LR.score(x_test, y_test)

print("The accuracy is " + str(accuracy * 100) + '\n')

with open("linear-regression.pickle", "wb") as f:
    pickle.dump(LR, f)
pickleIn = open("linear-regression.pickle", "rb")
LR = pickle.load(pickleIn)


# =============================================================================
#                             PREDICT & VISUALIZE
# The 'forecast' variable contains the forecast data for the stock in question
# The rest of the code is just plotting the data
# =============================================================================
forecast = LR.predict(prediction_checker)
for i in range(len(forecast)):
    print(str(i+1) + ") " + str(forecast[i]))
    
style.use("ggplot")

df["Forecast"] = np.nan

lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
nextUnix = lastUnix + 86400

for i in forecast:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += 86400
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df["Adj Close"].plot()
df["Forecast"].plot()
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.gcf().set_size_inches(18.5, 10.5, forward=True)
