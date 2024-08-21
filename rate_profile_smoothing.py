import numpy as np
from scipy.ndimage import gaussian_filter1d
import pickle
import matplotlib.pyplot as plt

fileName = 'decomposition_output.pkl'
with open(fileName, 'rb') as file:
    
    output = pickle.load(file)
    discharge_times= output['discharge_times'][0][0]
    fsamp = 10000
    duration_samples = np.shape(output['pulse_trains'][0])[1]
    bin_size_ms = 800  # Bin size in milliseconds
    bin_size_samples = int(bin_size_ms * fsamp / 1000)  # Bin size in samples
    # Convert discharge times to a spike train
    spike_train = np.zeros(duration_samples)
    spike_train[discharge_times] = 1  # Set spike times to 1

   # Bin the spike train
    binned_spike_train, bin_edges = np.histogram(discharge_times, bins=np.arange(0, duration_samples + bin_size_samples, bin_size_samples))

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(bin_edges[:-1], bins=bin_edges, weights=binned_spike_train, edgecolor='black')
    plt.xlabel("Time (samples)")
    plt.ylabel("Spike Count")
    plt.title("Histogram of Binned Spike Train")
    plt.show()

    # Convolve with a Gaussian kernel to smooth the firing rate
    sigma = 0.1 # Standard deviation of the Gaussian kernel in number of bins (adjust as needed)
    smoothed_firing_rate = gaussian_filter1d(binned_spike_train, sigma=sigma)

    # Convert to firing rate (spikes per second)
    bin_width_sec = bin_size_samples / fsamp
    smoothed_firing_rate = smoothed_firing_rate / bin_width_sec

    # Time vector for plotting
    time_vector = np.arange(len(smoothed_firing_rate)) * bin_width_sec

    # Define the start, stop, and interval
    start_value = 0
    end_value = duration_samples
    interval = bin_size_ms * 10

    # Generate the linspace with the specified interval
    linspace_values = np.arange(start_value, end_value + interval, interval)

    target_signal = output['path']
  

    # Create the figure and axis object
    fig, ax1 = plt.subplots()

    # Plot the first set of data on the first y-axis
    ax1.plot(linspace_values[:-1], smoothed_firing_rate)
    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y1 data', color='g')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(target_signal,'b-')
   

    ax2.plot(output['target'],'r-')
    plt.show()