import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from scipy.signal import coherence, iirnotch, filtfilt


fileName = 'decomposition_output.pkl'
spacing = 1
sample_frequency = 2048
notch_freq = 50.0  
quality_factor = 30.0  
from sklearn.cluster import KMeans

def apply_notch_filter(data, fs, freq, quality_factor):
   b, a = iirnotch(freq, quality_factor, fs)
   return filtfilt(b, a, data)

with open(fileName, 'rb') as file:
    
    output = pickle.load(file)

    num_electrodes = len(output['pulse_trains'])  # Should be 2 based on your description

    # Remove the 7th MU (index 6) from the first electrode
    #output['pulse_trains'][0] = np.delete(output['pulse_trains'][0], 6, axis=0)

    # Remove the 3rd MU (index 2) from the second electrode
    #output['pulse_trains'][1] = np.delete(output['pulse_trains'][1], 2, axis=0)

    # Create a figure with subplots for each electrode (2 rows, 1 column)
    fig, axs = plt.subplots(num_electrodes, 1, figsize=(15, 10))

    ax = axs
            
    mu_data = output['pulse_trains'][0]

    # Number of MUs and length of each MU spike train
    num_mus, train_length = mu_data.shape

    # Loop through each MU and plot the pulse train with spacing
    for j in range(num_mus):
        # Get the pulse train for the current MU
        pulse_train = mu_data[j]

        # Create an array representing the time points, converting to seconds
        time_points = np.arange(train_length) / sample_frequency

        peaks, _ = scipy.signal.find_peaks(np.squeeze(pulse_train), distance = np.round(10000*0.02)+1) # peaks variable holds the indices of all peaks
      
        if len(peaks) > 1:

            kmeans = KMeans(n_clusters = 2, init = 'k-means++',n_init = 1).fit(pulse_train[peaks].reshape(-1,1)) # two classes: 1) spikes 2) noise
            spikes_ind = np.argmax(kmeans.cluster_centers_)
            spikes = peaks[np.where(kmeans.labels_ == spikes_ind)]
            # remove outliers from the spikes cluster with a std-based threshold
            spikes = spikes[pulse_train[spikes] <= np.mean(pulse_train[spikes]) + 3*np.std(pulse_train[spikes])]
        else:
            spikes = peaks

        # Plot the pulse train with an offset, line in blue
        ax.plot(time_points, pulse_train / pulse_train.max() + j * spacing, linewidth=0.5, color='blue')

        # Plot the spikes with markers in orange
        spikes = np.where(pulse_train > 0.7)[0]
        ax.plot(time_points[spikes], pulse_train[spikes] / pulse_train.max() + j * spacing, 'o', color='orange', markersize=3)

    # Set labels and title for each subplot
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('MU Index')
    ax.set_title(f'Pulse Trains of Neurons (Electrode )')
    ax.set_yticks([j * spacing for j in range(num_mus)])
    ax.set_yticklabels([f'MU {j + 1}' for j in range(num_mus)])

    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.close(fig)