import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
g3x3_names = ['All Forces', 'Soft and Medium Force', 'Soft and Hard Force', 'Medium and Hard Force', 'Low Volume', 'Medium Volume', 'High Volume', 'Low and Medium Volume', 'Medium and High Volume', 'Low and High Volume']
g3x3_accuracies = [99, 84.22, 91.11, 84.22, 100, 99.89, 97.4, 97, 99.67, 99.89]

# PARAMETERS
categories = g3x3_names
values = g3x3_accuracies
labelFontSize = 20
textFontSize = 14

# Create the bar chart
plt.figure(figsize=(22, 14))
plt.bar(categories, values, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel("Dataset Categories", fontsize = labelFontSize)
plt.ylabel("Accuracy (%)", fontsize = labelFontSize)

plt.xticks(rotation= -15, ha='left', fontsize = textFontSize)  # Rotate 45 degrees and align right


# Display value labels on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')
    

# Show the grid (optional)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim([60, 105])

# Show the plot
plt.savefig('figure1.pdf')
plt.show()
