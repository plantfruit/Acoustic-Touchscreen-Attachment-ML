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

Sym_pressed = ['1D Rows/1cm.txt', '1D Rows/3cm.txt', '1D Rows/7cm.txt', '1D Rows/9cm.txt']
Sym_unpressed = ['1D Rows/9cm.txt', '1D Rows/8cm.txt', '1D Rows/7cm.txt', '1D Rows/6cm.txt']

# Parameters
pressData = Sym_pressed
unpressData = Sym_unpressed
legends = ["1 cm", "3 cm", "7 cm", "9 cm"]
#legends = ["1 cm, 2 cm, 3 cm, 4 cm", "6 cm, 7 cm, 8 cm, 9 cm"] # List to hold custom legends
ylim = [30, 120] #[-60, 60] #[30, 120] # [50, 120] #[-12e3, 8e3] # [-33e3, 33e3]
xName = 'Frequency (kHz)' #'Time (ms)' #'Frequency (kHz)'
yName = 'Magnitude (dB)' #'Magnitude' #'Magnitude (dB)'
freqWindow = [2.5, 20] # [2.5, 20] for 1D, [5 21] for 2D
fftOrTime = True # True - FFT, False - Time domain
plotPeaks = True

labelFontsize = 32
textFontsize = 26

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
lineStyles = ['-', '--', '-.', ':']  

# Sampling frequency
fs = 48e3  # in Hz       
# Processing begins                  


# Initialize figure for the plot
plt.figure(figsize=(12, 9))
# Iterate over each file and plot its data
counter = 0
for file_path in pressData:
    

    # Load the data from the text file
    pressLine = np.loadtxt(file_path)
    unpressLine = np.loadtxt(unpressData[counter])
    print(np.shape(pressLine))
    
    # Ensure the data is 1D
##    if data.ndim != 1:
##        print(f"File {file_path} is not 1 column. Skipping.")
##        continue
    
    # Generate x-axis in seconds
    if (not(fftOrTime)):
        x = np.arange(1, len(pressLine) + 1) / fs * 1e3
        pressSign = np.sign(pressLine)
        unpressSign = np.sign(unpressLine)
        pressLine = 10*np.log10(abs(pressLine)) #10*np.log10(pressLine - min(pressLine) + 1) #pressLine - min(pressLine) + 1
        unpressLine = 10*np.log10(abs(unpressLine)) #10*np.log10(unpressLine - min(unpressLine) + 1) #unpressLine - min(unpressLine) + 1
        pressLine = pressSign * pressLine
        unpressLine = unpressSign * unpressLine
    else:
        x = np.linspace(freqWindow[0], freqWindow[1], len(pressLine))        

    # Plot the data
    plt.plot(x, pressLine, color=colors[counter % len(colors)], linestyle = lineStyles[counter])  # Use file name as legend
    #plt.plot(x, unpressLine, color = 'orange')

    # Optional: Customize legend names
    #legends.append(legends)

    if (plotPeaks):
        markerSize = 100
        pressPeaks, _ = find_peaks(pressLine, prominence = 2)
        unpressPeaks, _ = find_peaks(unpressLine)
        #plt.scatter(x[pressPeaks], pressLine[pressPeaks], color = 'red', s = markerSize)
        #plt.scatter(x[unpressPeaks], unpressLine[unpressPeaks], color = 'green', s = markerSize)
        print(x[pressPeaks])
        print(pressLine[pressPeaks])

    # For copying and pasting elsewhere
    #plt.savefig('figure' + str(counter) + '.pdf')
    
    # Customization
    plt.axis([None, None, ylim[0], ylim[1]])
    plt.xlabel(xName, fontsize = labelFontsize)
    plt.ylabel(yName, fontsize = labelFontsize)    
    plt.xticks(fontsize = textFontsize)
    plt.yticks(fontsize = textFontsize)
    plt.legend(legends)
    #plt.legend(legends, loc='upper right', fontsize = textFontsize)  # Adjust legend position
    #plt.grid(True)
      
    
    counter = counter + 1


plt.savefig('figure'+str(counter)+'.pdf')
plt.show()  
