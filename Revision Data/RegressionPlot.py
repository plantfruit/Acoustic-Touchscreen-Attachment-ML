import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score

# File names
regX = "2D_regX.txt" # x-axis soft press force, microphone 1
regY = "2D_regY.txt" # y-axis hard press force, microphone 3

# Parameters
fileName = regX
debugMode = False
labelFontsize = 32
tickFontsize = 26
errorBarSize = 10 #5
scatterDotSize = 6 # 1.5

# Read in data 
regData = np.loadtxt(fileName)

# Pre-process regression values by averaging all the predictions for each label

# Initialize values for the loop so we have a point of comparison at the start 
oldLabel = regData[0][0]
predictions = []
plotLabels = []
plotPreds = []
plotErrors = []
counter = 0

# Iterate through all predictions
for row in regData:
    newLabel = row[0]
    counter = counter + 1

    # Gradually append prediction values for a certain label until you reach a different label or the end of the array
    # Once you reach end, aggregate predictions, and then find average and error for the group of predictions for this label
    if (oldLabel != newLabel or counter == regData.shape[0] - 1):
        if (counter == regData.shape[0] - 1): # Special case to append last item at end of array
            predictions.append(row[1].item())
            predictions.append(regData[regData.shape[0]-1][1].item())
        # Update final arrays to be plotted with the true value and avg predicted value    
        plotLabels.append(float(oldLabel.item()))
        plotPreds.append(np.average(predictions).item())
        plotErrors.append(np.std(predictions).item())
        if (debugMode):
            print(len(predictions))
        predictions = []
        
    predictions.append(row[1].item()) 
    oldLabel = newLabel

# Verification
if (debugMode):
    print(plotLabels)
    print(plotPreds)
    print(plotErrors)

# Calculate R^2 and RMSE of entire dataset    
RMSE = root_mean_squared_error(plotLabels, plotPreds)
R2 = r2_score(plotLabels, plotPreds)
RMSEstr = "{:.3g}".format(RMSE)
R2str = "{:.3g}".format(R2)

# Calculate trendline values with linear regression
slope, intercept, r_value, _, _ = linregress(plotLabels, plotPreds)
trendline = slope * np.asarray(plotLabels) + intercept  # Compute trendline

# Plot the regression data
plt.figure(figsize=(12, 9)) #(12, 9)
plt.scatter(plotLabels, plotPreds, color='b', linewidths = scatterDotSize)   # Scatter plot
trendlineGraph, = plt.plot(plotLabels, trendline, color='r')  # Trendline
plt.errorbar(plotLabels, plotPreds, yerr= plotErrors, fmt='o', color='b', alpha=0.7, capsize=errorBarSize)
plt.xlabel("Actual location (cm)", fontsize = labelFontsize)
plt.ylabel("Predicted location (cm)", fontsize = labelFontsize)
plt.yticks(fontsize = tickFontsize)
plt.xticks(fontsize = tickFontsize)
text_str = f"R² = {R2str}\nRMSE = {RMSEstr}"
ax = plt.gca()
props = dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8)
ax.annotate(text_str, xy=(1, 0), xycoords='axes fraction',
            xytext=(-10, 10), textcoords='offset points',
            ha='right', va='bottom', fontsize=tickFontsize, bbox=props)
#textBox = plt.text(1.5, 1, text_str, fontsize=tickFontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
plt.show()
