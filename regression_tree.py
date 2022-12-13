import io
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import requests





# URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"
# def download(url, filename):
#     response = requests.get(url)
#     with open(filename, "wb") as f:
#         f.write(response.content)

# download(URL,"realstate200.csv")
data = pd.read_csv('realstate200.csv')
# there are rows with missing values which we will deal with in pre-processing
print(data.isna().sum())
data.dropna(inplace=True)
print(data.isna().sum())
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)
regression_tree = DecisionTreeRegressor(criterion = "squared_error")
regression_tree.fit(X_train, Y_train)
regression_tree.score(X_test, Y_test)
print(regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("$",(prediction - Y_test).abs().mean()*1000)
# using mae criterion
regression_tree = DecisionTreeRegressor(criterion = "absolute_error")
regression_tree.fit(X_train, Y_train)
print(regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("$",(prediction - Y_test).abs().mean()*1000)