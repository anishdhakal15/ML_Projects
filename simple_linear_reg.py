import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import requests
from sklearn import linear_model
from sklearn.metrics import r2_score

path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

def download(url, filename):
    response = requests.get(path)
    with open(filename, "wb") as f:
        f.write(response.content)

download(path, "FuelConsumption.csv")
df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# # We can plot each of these features
# viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# viz.hist()
# plt.show()

# # let's plot each of these features against the Emission
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# # Train data distribution
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# # We can plot the fit line over the data
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")

# # Evaluation
# test_x = np.asanyarray(test[['ENGINESIZE']])
# test_y = np.asanyarray(test[['CO2EMISSIONS']])
# test_y_ = regr.predict(test_x)

# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y , test_y_) )

