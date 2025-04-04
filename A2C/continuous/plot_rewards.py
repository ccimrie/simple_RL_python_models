import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update(i):
    def smooth_average_sma(data, window_size):
        sma_values = np.zeros(len(data))
        # Calculate the SMA for each point (using a rolling window)
        for i in range(len(data)):
            # Define the window (i.e., the data points to average)
            start = max(0, i - window_size + 1)  # Avoid negative index
            sma_values[i] = np.mean(data[start:i+1])  # Mean of the window
        return sma_values
    ax.clear()
    try:
        values=np.loadtxt('test.txt')
        values_sma=smooth_average_sma(values, 10)
    except Exception as e:
        print("Nothing to load")
    else:
        color=[0.1,0.1,0.7]
        ax.plot(values, color=color, alpha=0.1)
        ax.plot(values_sma, color=color, alpha=1.0)

fig, ax=plt.subplots()
ani=FuncAnimation(fig, update, interval=500, cache_frame_data=False)

plt.show()