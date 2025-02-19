import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Filenames
D1_05res = ['Final Data/1D_reg.txt']
D2_regX_1 = 'Final Data/2D_regX_1.txt'
D2_regX_2 = 'Final Data/2D_regX_2.txt'
D2_regX_3 = 'Final Data/2D_regX_3.txt'
D2_regY_1 = 'Final Data/2D_regY_1.txt'
D2_regY_2 = 'Final Data/2D_regY_2.txt'
D2_regY_3 = 'Final Data/2D_regY_3.txt'
D2_regXall = 'Final Data/2D_regX.txt'
D2_regYall = 'Final Data/2D_regY.txt'

D1_R2 = [0.932]
D1_RMSE = [0.639]
D2_regX = [D2_regXall, D2_regX_1, D2_regX_2, D2_regX_3]
D2_regY = [D2_regYall, D2_regY_1, D2_regY_2, D2_regY_3]
D2_regX_R2 = [0.992, 0.99, 0.986, 0.979]
D2_regX_RMSE = [0.076, 0.087, 0.1, 0.125]
D2_regY_R2 = [0.985, 0.976, 0.949, 0.978]
D2_regY_RMSE = [0.104, 0.134, 0.195, 0.127]

# Parameters
fileNames = D2_regY
singleFile = False
#file_path = D2_regX
R2 = D2_regY_R2 #[0.932, ]
RMSE =  D2_regX_RMSE #[0.39, ]Y

labelFontsize = 18
tickFontsize = 12

# Color options for multiple lines
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
lineStyles = ['-', '--', '-.', ':']  
legendLines = []
legendText = []

counter = 0
plt.figure(figsize=(8, 6))
for name in fileNames:
    data = np.loadtxt(name) #, delimiter="\t")

    # Extract x and y columns
    x = data[:, 0]
    y = data[:, 1]

    # Group every 20 rows and calculate the mean and standard deviation
    group_size = 20
    num_groups = len(y) // group_size

    x_means = []
    y_means = []
    y_stds = []

    for i in range(num_groups):
        x_group = x[i * group_size:(i + 1) * group_size]
        y_group = y[i * group_size:(i + 1) * group_size]
        
        x_means.append(np.mean(x_group))
        y_means.append(np.mean(y_group))
        y_stds.append(np.std(y_group))

    x_means = np.array(x_means)
    y_means = np.array(y_means)
    y_stds = np.array(y_stds)

    print(x_means)
    print(y_means)
    print(y_stds)

    # Manually enter R² and RMSE 
    manual_r2 = R2[counter]  
    manual_rmse = RMSE[counter]

    # Perform linear regression to get trendline
    slope, intercept, r_value, _, _ = linregress(x_means, y_means)
    trendline = slope * x_means + intercept  # Compute trendline

    if (singleFile):
        plt.scatter(x_means, y_means, color='b')   # Scatter plot
        trendlineGraph, = plt.plot(x_means, trendline, color='r')  # Trendline
        plt.errorbar(x_means, y_means, yerr= y_stds, fmt='o', color='b', alpha=0.7, capsize=5)
    else:
        plt.scatter(x_means, y_means, color=colors[counter % len(colors)])   # Scatter plot
        trendlineGraph, = plt.plot(x_means, trendline, color=colors[counter % len(colors)], linestyle = lineStyles[counter])  # Trendline
        plt.errorbar(x_means, y_means, yerr= y_stds, fmt='o', color=colors[counter % len(colors)], alpha=0.7, capsize=5)

    legendLines.append(trendlineGraph)    

    # Add manually entered R² and RMSE values as text
    text_str = f"R² = {manual_r2}\nRMSE = {manual_rmse}"
    textBox = plt.text(7.5, 1, text_str, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    if (counter == 0):
        legendText.append(f'All Microphones, R² = {R2[counter]}\nRMSE = {RMSE[counter]}')
    else:
        legendText.append(f'Microphone {counter}, R² = {R2[counter]}\nRMSE = {RMSE[counter]}')
    counter = counter + 1

    # Ensure consistent axes
    plt.yticks(fontsize = tickFontsize)
    plt.xticks(fontsize = tickFontsize)
    #plt.yticks(np.arange(0,y_means[len(y_means) - 1] + y_means[0], 1.0))
    plt.ylim(0,y_means[len(y_means) - 1] + y_means[0])
    #plt.xticks(np.arange(0,y_means[len(y_means) - 1] + y_means[0], 1.0))
    plt.xlim(0,y_means[len(y_means) - 1] + y_means[0])

# Add titles and labels
plt.xlabel("Actual Distance (cm)", fontsize = labelFontsize)
plt.ylabel("Predicted Distance (cm)", fontsize = labelFontsize)
if (len(fileNames) > 1):
    plt.legend(handles = legendLines, labels = legendText, loc='best', fontsize = tickFontsize)
        
# Show the plot
plt.savefig('figure1.pdf')
plt.show()






