import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Filenames
D1_05res = 'Final Data/1D_reg.txt'
D2_regX_1 = 'Final Data/2D_regX_1.txt'
D2_regX_2 = 'Final Data/2D_regX_2.txt'
D2_regX_3 = 'Final Data/2D_regX_3.txt'
D2_regY_1 = 'Final Data/2D_regY_1.txt'
D2_regY_2 = 'Final Data/2D_regY_2.txt'
D2_regY_3 = 'Final Data/2D_regY_3.txt'

# Parameters
file_path = D2_regY_1
R2 = 0.976
RMSE =  0.134
data = np.loadtxt(file_path) #, delimiter="\t")

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
manual_r2 = R2  
manual_rmse = RMSE  

# Perform linear regression to get trendline
slope, intercept, r_value, _, _ = linregress(x_means, y_means)
trendline = slope * x_means + intercept  # Compute trendline

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_means, y_means, color='blue')  # Scatter plot
plt.plot(x_means, trendline, color='red')  # Trendline
plt.errorbar(x_means, y_means, yerr= y_stds, fmt='o', color='blue', alpha=0.7, capsize=5)
plt.ylim(0, y_means[len(y_means) - 1] + y_means[0])

# Add titles and labels
plt.xlabel("Actual Distance (cm)")
plt.ylabel("Predicted Distance (cm)")

# Add manually entered R² and RMSE values as text
text_str = f"R² = {manual_r2}\nRMSE = {manual_rmse}"
textBox = plt.text(2.4, 0.6, text_str, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
# Show the plot
plt.show()





