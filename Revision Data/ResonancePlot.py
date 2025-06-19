import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.signal import find_peaks

# Filenames
corners1 = 'planar_resonance_1.txt'
corners2 = 'planar_resonance_2.txt'
corners3 = 'planar_resonance_3.txt'

# Parameters
pressData = corners3
ylim = [50, 120] 
xName = 'Frequency (kHz)' #'Time (ms)' #'Frequency (kHz)'
yName = 'Magnitude (dB)' #'Magnitude' #'Magnitude (dB)'
freqWindow = [5, 21] # [2.5, 20] for 1D, [5 21] for 2D
labelFontsize = 32
textFontsize = 26

# Define some figure settings
colors = ['black', 'blue', 'purple']
lineStyles = ['-', '--', ':']
micNum = pressData[len(pressData) - 5] # Use text filename format to grab the legend caption 
legends = [f"No contact (Mic {micNum})", f"Contact point 1 (Mic {micNum})", f"Contact point 2 (Mic {micNum})"]


# Processing begins                  

# Iterate over each file and plot its data
counter = int(micNum) - 1
array1 = np.loadtxt(pressData)
num_rows = array1.shape[0]
print(num_rows)

if (array1.ndim == 1):
    num_rows = 1

plt.figure(figsize=(12, 9))   
# Loop through each pair of rows from the two arrays
for i in range(num_rows):
         

    if (array1.ndim == 1):
        arrayLen = len(array1)
    else:
        arrayLen = len(array1[i])
    x = np.linspace(freqWindow[0], freqWindow[1], arrayLen)            

    # Plot the ith row from both arrays
    if (array1.ndim == 1):
        plt.plot(x, array1,color=colors[counter], linestyle = lineStyles[i]) # Use file name as legend
    else:
        plt.plot(x, array1[i],color=colors[counter], linestyle = lineStyles[i]) # Use file name as legend
        
    # Customization
    plt.axis([None, None, ylim[0], ylim[1]])
    plt.xlabel(xName, fontsize = labelFontsize)
    plt.ylabel(yName, fontsize = labelFontsize)    
    plt.xticks(fontsize = textFontsize)
    plt.yticks(fontsize = textFontsize)
    plt.legend(legends, loc='upper right', fontsize = textFontsize, edgecolor='none', handlelength = 0.8)  # Adjust legend position
    #plt.grid(True)
    
plt.savefig(f'{pressData}' + '.pdf')

plt.show()

