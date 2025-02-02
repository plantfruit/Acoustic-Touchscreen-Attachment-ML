import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
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

BGwhite_vol1 = 'Data/BGwhite_vol1.txt'
BGwhite_vol2 = 'Data/BGwhite_vol2.txt'
BGwhite_vol3 = 'Data/BGwhite_vol3.txt'
grid3x3_labels  = 'Data/3x3_labels.txt'

# Select filename
featureFile = BGwhite_vol3
labelFile = grid3x3_labels

featureTest = BGwhite_vol2
labelTest = grid3x3_labels

X = np.loadtxt(featureFile)
y = np.loadtxt(labelFile)
X_test = np.loadtxt(featureTest)
y_test = np.loadtxt(labelTest)
X_train = X
y_train = y

# Dataset Parameters
internalSplit = True
num_labels = 9
files_per_label = 10
rows_per_file = 10 

# Classification begins here
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = cross_val_predict(model, X, y, cv = 10)  # Predict
print(y_pred)
print(np.shape(y_pred))

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Print the accuracy for each fold
cv_scores = cross_val_score(model, X, y, cv = 10)
print(f"Accuracy for each fold: {cv_scores}")

# Generate the confusion matrix with fixed size
all_labels = np.arange(1, num_labels + 1)  # All possible labels from 1 to 25
cm = confusion_matrix(y_test, y_pred, labels=all_labels)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
plt.title('Confusion Matrix (Fixed Size)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

