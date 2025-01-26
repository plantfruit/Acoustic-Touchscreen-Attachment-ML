import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# FILENAMES
regX1_cols1to94 = 'Data/5x5_1_regX_1_cols1to94.txt'
regX1_cols1to94_labels = 'Data/5x5_1_regX_1_cols1to94_labels.txt'


# Select filename
featureFile = regX1_cols1to94
labelFile = regX1_cols1to94_labels

X = np.loadtxt(featureFile)
y = np.loadtxt(labelFile)

# Perform test-train split

# Dataset Parameters
num_labels = 10
files_per_label = 10
rows_per_file = 10 

# Train-test split: First 80 rows/train, last 20 rows/test per label
train_indices = []
test_indices = []


for label in range(1, num_labels + 1):
    # Get all rows for this label
    label_rows = np.where(y == label)[0]

    # Split the indices: first 80 for training, last 20 for testing
    train_indices.extend(label_rows[:80])
    test_indices.extend(label_rows[80:])

    # Reversed order
    #train_indices.extend(label_rows[100:])
    #test_indices.extend(label_rows[:100])
    
    # Split the indices: 
    # First 20 rows and last 60 rows for training
    #train_indices.extend(label_rows[:50])
    #train_indices.extend(label_rows[100:])
    # 2nd set of 20 rows for testing
    #test_indices.extend(label_rows[50:100])

    # Split the indices: 
    # First 20 rows and last 60 rows for testing
    #test_indices.extend(label_rows[:50])
    #test_indices.extend(label_rows[100:])
    # 2nd set of 20 rows for training
    #train_indices.extend(label_rows[50:100])


# Convert to arrays for indexing
train_indices = np.array(train_indices)
test_indices = np.array(test_indices)
#print(train_indices)
#print(test_indices)

# Split the dataset
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))


# Linear regression 
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model
y_pred = model.predict(X_test)  # Predict
print(y_pred)
print(np.shape(y_pred))
print(y_test)
print(np.shape(y_test))

# Evaluate 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R^2:", r2)


