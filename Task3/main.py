import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

dataset = pd.read_table("Task3\\ddos-train-labeled.netflow",delimiter='\t')
dataset_test = pd.read_table("Task3\\ddos-test-01.netflow", delimiter='\t')

X_train = dataset.iloc[:,1:]
X_test = dataset_test.iloc[:,1:]
Y_test = dataset_test.iloc[:,0]

# One hot encoding the Protocol feature
temp_df = pd.get_dummies(X_train.iloc[:,3])
for i in temp_df.columns:
    X_train[i] = temp_df[i]

temp_df = pd.get_dummies(X_test.iloc[:,3])
for i in temp_df.columns:
    X_test[i] = temp_df[i]

# Selecting particular features
X_train = X_train.iloc[:,[0,3,4,5,6,7]]
X_test = X_test.iloc[:,[0,3,4,5,6,7]]

# Normalizing training value for better numerical and graphical representation
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
X_test = ss_train.fit_transform(X_test)

# K-Nearest Neighbors
from sklearn.cluster import KMeans
models = KMeans(n_clusters = 2, init="random")

# Fit the classifier
models.fit(X_train)

# Making predictions
result = models.predict(X_test).tolist()
label = Y_test.to_list()

# Calculating result
accurate = 0

for i in range(len(result)):
    if result[i] == 0 and label[i] == 1:
        accurate += 1
    elif result[i] == 1 and label[i] == 0:
        accurate += 1

print(str(accurate/len(result) * 100) + "%")