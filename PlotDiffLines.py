import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.signal import find_peaks


# Filenames
D2_pressedTime = ['Final Data/2D_time1.txt', 'Final Data/2D_time2.txt', 'Final Data/2D_time3.txt']
D2_unpressedTime = ['Final Data/2D_time1U.txt', 'Final Data/2D_time2U.txt', 'Final Data/2D_time3U.txt']
D2_pressedFFT = ['Final Data/2D_fft1P.txt', 'Final Data/2D_fft2P.txt', 'Final Data/2D_fft3P.txt']
D2_unpressedFFT = ['Final Data/2D_fft1U.txt', 'Final Data/2D_fft2U.txt', 'Final Data/2D_fft3U.txt']

D2v2_pressedFFT = ['Final Data/2D_v2_fft1P.txt', 'Final Data/2D_v2_fft2P.txt', 'Final Data/2D_v2_fft3P.txt']
D2v2_pressedTime = ['Final Data/2D_v2_time1P.txt', 'Final Data/2D_v2_time2P.txt', 'Final Data/2D_v2_time3P.txt']

D1_pressedTime = ['Final Data/1D_timeP.txt']
D1_unpressedTime = ['Final Data/1D_timeU.txt']
D1_pressedFFT = ['Final Data/1D_fftP.txt']
D1_unpressedFFT = ['Final Data/1D_fftU.txt']

D1obj_pressedTime = ['Final Data/1D_6obj_timeP.txt']
D1obj_unpressedTime = ['Final Data/1D_6obj_timeU.txt']
D1obj_pressedFFT = ['Final Data/1D_6obj_fftP.txt']
D1obj_unpressedFFT = ['Final Data/1D_6obj_fftU.txt']

D2obj_pressedFFT = ['Final Data/2Dobj_1_fftP.txt','Final Data/2Dobj_2_fftP.txt','Final Data/2Dobj_3_fftP.txt']
D2obj_pressedTime = ['Final Data/2Dobj_1_timeP.txt', 'Final Data/2Dobj_2_timeP.txt', 'Final Data/2Dobj_3_timeP.txt']

D2isoIn = 'Noise Isolation/D2_isoIn.txt'
D2isoOut = 'Noise Isolation/D2_isoOut.txt'
D1isoIn = 'Noise Isolation/D1_isoIn.txt'
D1isoOut = 'Noise Isolation/D1_isoOut.txt'

corners1 = 'Final Data/4corners_1.txt'
corners2 = 'Final Data/4corners_2.txt'
corners3 = 'Final Data/4corners_3.txt'



# Parameters
pressData = corners3
unpressData = D1isoIn
legends = ["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Unpressed"]
#legends = ["Outside Sensor", "Inside Sensor"]
#legends = ["Pressed", "Unpressed"] # List to hold custom legends
ylim = [30, 120] #[-60, 60] #[30, 120] #[-12e3, 8e3] # [-33e3, 33e3]
xName = 'Frequency (kHz)' #'Time (ms)' #'Frequency (kHz)'
yName = 'Magnitude (dB)' #'Magnitude' #'Magnitude (dB)'
freqWindow = [5, 21] # [2.5, 20] for 1D, [5 21] for 2D
fftOrTime = True # True - FFT, False - Time domain
plotPeaks = False

labelFontsize = 32
textFontsize = 26

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
lineStyles = ['-', '--', '-.', ':', '-']  

# Sampling frequency
fs = 48e3  # in Hz       
# Processing begins                  

# Iterate over each file and plot its data
counter = 0
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
        plt.plot(x, array1,color=colors[counter % len(colors)], linestyle = lineStyles[counter]) # Use file name as legend
    else:
        plt.plot(x, array1[i],color=colors[counter % len(colors)], linestyle = lineStyles[counter]) # Use file name as legend
        
    # Customization
    plt.axis([None, None, ylim[0], ylim[1]])
    plt.xlabel(xName, fontsize = labelFontsize)
    plt.ylabel(yName, fontsize = labelFontsize)    
    plt.xticks(fontsize = textFontsize)
    plt.yticks(fontsize = textFontsize)
    plt.legend(legends, loc='upper right', fontsize = textFontsize)  # Adjust legend position
    #plt.grid(True)
    counter = counter + 1

    
plt.savefig('figure'+str(i)+'.pdf')

plt.show()

