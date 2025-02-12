import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Filenames
D2_pressedTime = ['Final Data/2D_time1.txt', 'Final Data/2D_time2.txt', 'Final Data/2D_time3.txt']
D2_unpressedTime = ['Final Data/2D_time1U.txt', 'Final Data/2D_time2U.txt', 'Final Data/2D_time3U.txt']
D2_pressedFFT = ['Final Data/2D_fft1P.txt', 'Final Data/2D_fft2P.txt', 'Final Data/2D_fft3P.txt']
D2_unpressedFFT = ['Final Data/2D_fft1U.txt', 'Final Data/2D_fft2U.txt', 'Final Data/2D_fft3U.txt']

# Parameters
pressData = D2_pressedFFT
unpressData = D2_unpressedFFT
legends = ["Pressed", "Unpressed"] # List to hold custom legends
ylim = [50, 120] #[-12e3, 8e3]
xName = 'Frequency (kHz)'
yName = 'Magnitude (dB)'
freqWindow = [5, 21]
fftOrTime = True


# Sampling frequency
fs = 48e3  # in Hz       
# Processing begins                  

# Iterate over each file and plot its data
counter = 0        
for file_path in pressData:
    # Initialize figure for the plot
    plt.figure(figsize=(10, 8))

    # Load the data from the text file
    pressLine = np.loadtxt(file_path)
    unpressLine = np.loadtxt(unpressData[counter])
    
    # Ensure the data is 1D
##    if data.ndim != 1:
##        print(f"File {file_path} is not 1 column. Skipping.")
##        continue
    
    # Generate x-axis in seconds
    if (not(fftOrTime)):
        x = np.arange(1, len(pressLine) + 1) / fs * 1e3
    else:
        x = np.linspace(freqWindow[0], freqWindow[1], len(pressLine))

    # Plot the data
    plt.plot(x, pressLine)  # Use file name as legend
    plt.plot(x, unpressLine)

    # Optional: Customize legend names
    legends.append(legends)

    # For copying and pasting elsewhere
    #plt.savefig('figure' + str(counter) + '.pdf')
    
    # Customization
    plt.axis([None, None, ylim[0], ylim[1]])
    plt.xlabel(xName)
    plt.ylabel(yName)
    plt.legend(legends, loc='upper right')  # Adjust legend position
    plt.grid(True)
    plt.show()
    
    counter = counter + 1
