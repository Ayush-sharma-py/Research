import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score


dataset = pd.read_table("Task 1/ts2-ddos-train-both-labeled.netflow",delimiter='\t')
dataset_training = pd.read_table("Task 1/ts2-ddos-test01.netflow",delimiter='\t')

y_train = dataset.iloc[:,0] # Getting the training labels
X_train = dataset.iloc[:,1:]

y_test = dataset_training.iloc[:,0] # Getting the testing labels
X_test = dataset_training.iloc[:,1:]

df2 = pd.get_dummies(X_train.iloc[:,4])
df2.head()

# Normalizing training value for better numerical and graphical representation
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()


accuracy, precision, recall = {}, {}, {}
precision2, recall2 = {}, {}

# Custom implementation of recall function
def recallCustom(predictions, y_test):
    num_tp = 0
    num_fp = 0
    for i in range(0,len(y_test)):
        if(y_test[i] == 1 and predictions[i] == 1):
            num_tp += 1
        elif(y_test[i] == 0 and predictions[i] == 1):
            num_fp += 1
    return num_tp/float(num_tp + num_fp)

# Custom implementation of precision function
def precisionCustom(predictions, y_test):
    num_tp = 0
    num_fn = 0
    for i in range(0,len(y_test)):
        if(y_test[i] == 1 and predictions[i] == 1):
            num_tp += 1
        elif(y_test[i] == 1 and predictions[i] == 0):
            num_fn += 1
    return num_tp/float(num_tp + num_fn)

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train, y_train)
    
    # Make predictions
    predictions = models[key].predict(X_test)

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    precision2[key] = precisionCustom(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    recall2[key] = recallCustom(predictions, y_test)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Precision2', 'Recall', 'Recall2'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Precision2'] = precision2.values()
df_model['Recall'] = recall.values()
df_model['Recall2'] = recall2.values()

print(df_model)

df_model.plot(xlabel = "Algorithm", ylabel = "Performance", kind = 'bar', title = "Model Comparision")
plt.tight_layout()
plt.show()