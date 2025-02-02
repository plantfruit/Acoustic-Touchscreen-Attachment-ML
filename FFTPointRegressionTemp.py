import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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


# Select filename
featureFile = regY1_2
labelFile = regX1_10points_0_3mm

featureFileTest = regY2_2
labelFileTest = regX1_10points_0_3mm

X_train = np.loadtxt(featureFile)
y_train = np.loadtxt(labelFile)
X_test = np.loadtxt(featureFileTest)
y_test = np.loadtxt(labelFileTest)

# Dataset Parameters
num_labels = 10
files_per_label = 10
rows_per_file = 10 




# Linear regression 
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model
y_pred = model.predict(X_test)  # Predict
print(y_pred)
print(np.shape(y_pred))

# Evaluate 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R^2:", r2)

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
