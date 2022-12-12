import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import requests
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
k = 4

# onetime 
def download(url, filename):
    response = requests.get(path)
    with open(filename, "wb") as f:
        f.write(response.content)

download(path, 'teleCust1000t.csv')

df = pd.read_csv('teleCust1000t.csv')
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df['custcat'].values
# Normalizing data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

yhat = neigh.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# # calculate the accuracy of KNN for different values of k
# Ks = 10
# mean_acc = np.zeros((Ks-1))
# std_acc = np.zeros((Ks-1))

# for n in range(1,Ks):
    
#     #Train Model and Predict  
#     neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
#     yhat=neigh.predict(X_test)
#     mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
#     std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


# # Plot the model accuracy for a different number of neighbors
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
# plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Neighbors (K)')
# plt.tight_layout()
# plt.show()