import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
from decimal import Decimal as D

# FILENAMES
regX1_cols1to94 = 'Data/5x5_1_regX_1_cols1to94.txt'
regX1_cols1to94_labels = 'Data/5x5_1_regX_1_cols1to94_labels.txt'
regX1_10points_0_3mm = 'Data/regression_10point_0_3res.txt' # Label file
# Full FFT spectrums (not selected columns)
regX1_1 = 'Data/5x5_regX_1.txt'
regX1_2 = 'Data/5x5_regX_2.txt'
regX1_3 = 'Data/5x5_regX_3.txt'
# Selected areas from FFT
regY1_2_cols8to30 = 'Data/5x5_1_regY_2_cols8to30.txt'
regY1_3_cols1to46 = 'Data/5x5_1_regY_3_cols1to46.txt'
regY1_3_cols137to140 = 'Data/5x5_1_regY_3_cols137to140.txt'
regY1_3_cols1to45_137to140 = 'Data/5x5_1_regY_3_cols1to46_cols137to140.txt'
# Full FFT spectrum
regY1_1 = 'Data/5x5_regY_1.txt'
regY1_2 = 'Data/5x5_regY_2.txt'
regY1_3 = 'Data/5x5_regY_3.txt'
# New balloon, hard force (full FFT spectrum)
regX2_1 = 'Data/5x5_2_regX_1.txt'
regX2_2 = 'Data/5x5_2_regX_2.txt'
regX2_3 = 'Data/5x5_2_regX_3.txt'
regY2_1 = 'Data/5x5_2_regY_1.txt'
regY2_2 = 'Data/5x5_2_regY_2.txt'
regY2_3 = 'Data/5x5_2_regY_3.txt'

regY2_3_col88 = 'Data/5x5_2_regX_3_col88.txt'
regX1_1_cols57to75 = 'Data/5x5_2_regX_1_cols57to75.txt'
regX1_1_col67 = 'Data/5x5_2_regX_1_col67.txt'

tube1D_res05 = 'Data/1Dtube_05res.txt'
tube1D_res05_labels = 'Data/1Dtube_05res_labels.txt'


# Select filename
featureFile = tube1D_res05
labelFile = tube1D_res05_labels

X = np.loadtxt(featureFile)
y = np.loadtxt(labelFile)

if X.ndim == 1:
    X = X.reshape(-1, 1)

# Perform test-train split

# Dataset Parameters
num_labels = 17
files_per_label = 10
rows_per_file = 10 



#for seed in range(1, 101):
# Train-test split: First 80 rows/train, last 20 rows/test per label
train_indices = []
test_indices = []

for label in range(1, num_labels + 1):
    # Get all rows for this label
    label_rows = np.where(y == round(label * 0.5 + 0.5, 1))[0]
    #label_rows = np.where(y == round(label, 1))[0]
    #label_rows = np.where(y == round(label * 0.3, 1))[0]
    #print(round(label * 0.3))

    # Shuffle the rows
    seed = 75
    np.random.seed(seed)    
    # Reshape the array into 10 groups of 10 values
    groups = label_rows.reshape(10, 10)  # Shape: (10, 10)
    # Shuffle the groups along the first axis
    np.random.shuffle(groups)
    # Flatten the shuffled groups back into a 1D array
    shuffled_data = groups.flatten()
    
    # Split the indices: first 80 for training, last 20 for testing
    train_indices.extend(shuffled_data[:80])
    test_indices.extend(shuffled_data[80:])

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

# Linear regression 
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model
y_pred = model.predict(X_test)  # Predict
#print(y_pred)
#print(np.shape(y_pred))

# Evaluate 
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)
#print("seed:", seed)


# Save regression results to an Excel file
import xlsxwriter

# Create a workbook and add a worksheet
workbook = xlsxwriter.Workbook("RegressionResults.xlsx")
worksheet = workbook.add_worksheet()

# Write data into columns (no headers)
for row, (value1, value2) in enumerate(zip(y_test, y_pred)):
    worksheet.write(row, 0, value1)  # Write into column A (index 0)
    worksheet.write(row, 1, value2)  # Write into column B (index 1)

# Close the workbook
workbook.close()
