import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Filenames
D1_05res = 'Final Data/

# Load data from the text file (tab-separated)
file_path = 
data = np.loadtxt(file_path, delimiter="\t")

# Extract x and y columns
x = data[:, 0]
y = data[:, 1]

# Perform linear regression
slope, intercept, r_value, _, _ = linregress(x, y)
trendline = slope * x + intercept  # Compute trendline

# Define error bars (Example: random small errors, replace with actual errors if available)
error_y = np.random.uniform(0.5, 1.5, size=len(y))

# Manually enter R² and RMSE (you will need to calculate these separately)
manual_r2 = 0.95  # Example R² value
manual_rmse = 2.3  # Example RMSE value

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label="Data Points")  # Scatter plot
plt.plot(x, trendline, color='red', label="Trendline")  # Trendline
plt.errorbar(x, y, yerr=error_y, fmt='o', color='blue', alpha=0.7, capsize=5

# Add titles and labels
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.title("Scatter Plot with Trendline and Error Bars")
plt.legend()

# Add manually entered R² and RMSE values as text
text_str = f"R² = {manual_r2}\nRMSE = {manual_rmse}"
plt.text(min(x), max(y), text_str, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Show the plot
plt.show()





