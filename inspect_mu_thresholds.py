import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from sklearn.cluster import KMeans

fileName = "decomposition_output 2.pkl"
spacing = 1
sample_frequency = 10000
nfirings = 20

with open(fileName, "rb") as file:

    output = pickle.load(file)
    output["mvc"] = 15
    num_electrodes = len(output["pulse_trains"])

    rel_rt = []
    rel_dert = []
    abs_rt = []
    abs_dert = []

    nmus, _ = np.shape(output["pulse_trains"][0])

    for i in range(nmus):

        output["discharge_times"][0][i] = [
            dt
            for dt in output["discharge_times"][0][i]
            if output["fsamp"] <= dt <= 500000 - output["fsamp"]
        ]

        mu_rec = output["discharge_times"][0][i][0:nfirings]
        mu_derec = output["discharge_times"][0][i][-nfirings:]

        rel_rt.append(np.mean(output["target"][mu_rec]))
        rel_dert.append(np.mean(output["target"][mu_derec]))

        abs_rt.append((rel_rt[i] / max(output["target"])) * output["mvc"])
        abs_dert.append((rel_dert[i] / max(output["target"])) * output["mvc"])

    # Sort the recruitment thresholds and get the indices of the sorted order
    sorted_indices = np.argsort(rel_rt)

    # reorder the other threshold based arrays
    rel_rt = np.array(rel_rt)[sorted_indices]
    rel_dert = np.array(rel_dert)[sorted_indices]
    abs_rt = np.array(abs_rt)[sorted_indices]
    abs_dert = np.array(abs_dert)[sorted_indices]

    output["pulse_trains"][0] = np.array(output["pulse_trains"][0])[sorted_indices]
    output["discharge_times"][0] = [
        output["discharge_times"][0][i] for i in sorted_indices
    ]

    print("discharge times cleaned")

    t = np.arange(0, len(output["target"]) / output["fsamp"], 1 / output["fsamp"])

    # Initialize the binary matrix for the raster plot
    raster_plot = np.zeros((nmus, len(t)))

    # Populate the raster plot matrix
    for i, mu_t in enumerate(output["discharge_times"][0]):

        indices = np.array(mu_t).astype(int)
        raster_plot[i, indices] = 1

    # Plotting the figure
    plt.figure(figsize=(10, 6))

    # Plot the raster plot with thinner lines
    for i in range(nmus):
        firing_times = np.where(raster_plot[i] == 1)[0] / sample_frequency
        plt.eventplot(
            firing_times,
            lineoffsets=i + 1,
            colors="C" + str(i % 10),
            linelengths=0.7,
            linewidths=0.5,
        )  # Thinner lines with `linelengths=0.5`

    # Scale the target trajectory to fit over the top of the last motor unit
    scaled_target = (output["target"] / max(output["target"])) * (nmus + 1)

    # Overlay the scaled target trajectory
    plt.plot(t, scaled_target, color="black", linewidth=1, label="Target Trajectory")

    # Labels and other plot adjustments
    plt.xlabel("Time (s)")
    plt.ylabel("Motor Units (n)")
    plt.ylim(-1, nmus + 2)
    plt.xlim(0, max(t))
    plt.legend()
    plt.title("Motor Unit Discharge Times and Target Trajectory")
    plt.show()
