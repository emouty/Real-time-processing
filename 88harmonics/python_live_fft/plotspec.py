from numpy import sin, linspace, pi
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange



def plotSpectrum(y, Fs):
    """
    Fs : sampling rate
    y : signal
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y)  # length of the signal
    k = arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(int(n / 2))]  # one side frequency range

    Y = fft(y) / n  # fft computing and normalization
    Y = Y[range(int(n / 2))]

    plot(frq, abs(Y), 'r')  # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    show()

