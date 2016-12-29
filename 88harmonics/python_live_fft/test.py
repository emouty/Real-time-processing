import pyaudio as pa
import numpy as np
from matplotlib import pyplot as plt
from pylab import plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange, signal
import struct
import time
from parabolic import parabolic

soundObject = pa.PyAudio()

nFFT = 1028
BUF_SIZE = 4 * nFFT
FORMAT = pa.paFloat32
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3

fulldata = np.array([])
dry_data = np.array([])


def readstream(inStream):
    """
    read stream depreciated
    :param inStream:
    :return: return right
    """
    numberOfValues = max(stream.get_read_available() / nFFT, 1) * nFFT / 2
    data = inStream.read(int(numberOfValues))
    #print(len(data))
    #y = np.array(struct.unpack("%dh" % (2 * numberOfValues * CHANNELS), data))
    y = np.fromstring(data, dtype=np.float32)
    y_L = y[::2]
    y_R = y[1::2]


    return y_R,

def plotSpectrum(y, Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    :param y:
    :param Fs:
    :return:
    """
    n = len(y)  # length of the signal
    k = arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(int(n / 2))]  # one side frequency range
    y = y * signal.blackmanharris(len(y))
    Y = fft(y) / n  # fft computing and normalization
    Y = Y[range(int(n / 2))]

    plot(frq, abs(Y), 'r')  # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')
    #show()
    return Y,

def findFreq(spec):
    """

    :param spec: full spectrum
    :return: fundamental frequency
    """
    spec = abs(np.array(spec))
    # i : indice of max amp
    i = np.argmax(spec[0])
    i = parabolic(np.log(spec[0]), i)[0]
    return RATE * i / len(spec[0]),

def callback(in_data, frame_count, time_info, flag):
    global b, a, fulldata, dry_data, frames
    audio_data = np.fromstring(in_data, dtype=np.float32)
    #do processing here
    fulldata = np.append(fulldata,audio_data)
    return (audio_data, pa.paContinue)


stream = soundObject.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=BUF_SIZE,
                          stream_callback=callback)

while stream.is_active():
    print("actif")
    time.sleep(2)
    stream.stop_stream()
stream.close()

frames = np.hstack(fulldata)

    #readData = readstream(stream)

    # i = 0
    # frames = []
    #
    # for i in range(0, int(RATE / BUF_SIZE * RECORD_SECONDS)):
    #     data = readstream(stream)
    #     frames.append(data)
    #
    # #Y_R = np.fft.fft(readData, nFFT)
    #

spectrum = plotSpectrum(frames, RATE)


#stream.close()
soundObject.terminate()