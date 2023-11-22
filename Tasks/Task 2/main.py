import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

dataset = pd.read_table("Task 2/ddos-train-labeled.netflow",delimiter='\t')
dataset_training = pd.read_table("Task 2/ddos-test-01.netflow",delimiter='\t')
dataset_training2 = pd.read_table("Task 2/ddos-test-02.netflow",delimiter='\t')
dataset_training3 = pd.read_table("Task 2/ddos-test-03.netflow",delimiter='\t')
dataset_training4 = pd.read_table("Task 2/ddos-test-04.netflow",delimiter='\t')


y_train = dataset.iloc[:,0] # Getting the training labels
X_train = dataset.iloc[:,1:]

# One hot encoding the Protocol feature
temp_df = pd.get_dummies(X_train.iloc[:,3])
for i in temp_df.columns:
    X_train[i] = temp_df[i]

# Selecting particular features
X_train = X_train.iloc[:,[0,3,4,5,6,7]]

# Normalizing training value for better numerical and graphical representation
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)
# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models = KNeighborsClassifier()

# Fit the classifier
models.fit(X_train, y_train)
accuracy, precision, recall = [], [], []

data_testing = [dataset_training, dataset_training2, dataset_training3, dataset_training4]

# Generating the accuracy metrics for all the test datasets
for i in data_testing:
    y_test = i.iloc[:,0] # Getting the testing labels
    X_test = i.iloc[:,1:]


    temp_df = pd.get_dummies(X_test.iloc[:,3])
    for i in temp_df.columns:
        X_test[i] = temp_df[i]

    X_test = X_test.iloc[:,[0,3,4,5,6,7]]

    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    # Make predictions
    predictions = models.predict(X_test)

    # Calculate metrics
    accuracy.append(accuracy_score(predictions, y_test))
    precision.append(precision_score(predictions, y_test))
    recall.append(recall_score(predictions, y_test))

dataset = ["Test1", "Test2", "Test3", "Test4"]
df = pd.DataFrame({"Accuracy": accuracy, "Precision": precision, "Recall": recall}, index = dataset)
print(df)

# Plotting the graph
df.plot(xlabel = "Dataset", ylabel = "Performance", kind = 'bar', title = "K Nearest Neighbor")
plt.legend(loc = 1)
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()