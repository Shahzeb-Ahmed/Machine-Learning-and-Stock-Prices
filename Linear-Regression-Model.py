import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import quandl, math, datetime, pickle
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


# =============================================================================
# First we import the data and get our dataframe in order
# =============================================================================
df = quandl.get("WIKI/GOOGL")       #Only goes till 2018 but its good enough

df = df[["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume"]]

df["Percent Volatility"] = (df["Adj. High"] - df["Adj. Close"])/df["Adj. Close"]

df["Percent Change"] = (df["Adj. Close"] - df["Adj. Open"])/df["Adj. Open"]

df = df[["Adj. Close", "Percent Volatility", "Percent Change", "Adj. Volume"]]

df.fillna(-99999, inplace = True)       #Thus nan will just become outliers and not really affect the model


# =============================================================================
# Now we get our training and testing sets ready
# =============================================================================
forecastCol = "Adj. Close"

forecastOut = int(math.ceil(0.01 * len(df)))     #Gives 35 --> 35 nan generated

df["label"] = df[forecastCol].shift(-forecastOut)       #What we want our model to think is right. Also the negative sign is so the data is shifted up

X = np.array(df.drop(["label"], 1))
X = preprocessing.scale(X)      #Remember to scale with everything else. In essence, this ensures the data is randomized, normalized and unbiased
X = X[: -forecastOut]
XLately = X[-forecastOut: ]     #Stuff we're actually predicting against

df.dropna(inplace = True)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)    #20% of the data is for testing


# =============================================================================
# Finally, we test the data. Once our classifier is ready, save it
# =============================================================================
classifier = LinearRegression()     #Seemed to work best out of the possible options

classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)

print(accuracy)

with open("linear-regression.pickle", "wb") as f:
    pickle.dump(classifier, f)

pickleIn = open("linear-regression.pickle", "rb")
classifier = pickle.load(pickleIn)      #Drastically improves time-efficiency. Remember to train bi-weekly, at least.


# =============================================================================
# Now lets predict stuff
# =============================================================================
style.use("ggplot")

forecastSet = classifier.predict(XLately)

df["Forecast"] = np.nan

lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
nextUnix = lastUnix + 86400

for i in forecastSet:       #All this does is put dates on the axes when we plot the data
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += 86400
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]        #Need to understand this fully
    
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.gcf().set_size_inches(18.5, 10.5, forward=True)     #Change plot size