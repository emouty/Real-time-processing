from scipy import signal
import matplotlib.pyplot as plt





window = signal.gaussian(100, std=7)
print(window)
plt.plot(window)
plt.title(r"Gaussian window ($\sigma$=7)")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()