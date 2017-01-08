import alsaaudio as a
import random
import struct
import numpy as np
from scipy import fft, signal, ifft
from parabolic import parabolic
from obspy.signal.filter import bandstop, highpass
from bluetooth import *

#alsa variables
BUFFER_SIZE = 1024
CHANNELS = 1
RATE = 44100
FORMAT = a.PCM_FORMAT_S16_LE


# bluetooth config

HOST = "A4:D1:8C:D2:52:39"      # The remote host
PORT =  3              # Server port

#global variables
indiceOfFundamental = 0
harmonic = False
isharmonic = False
washarmonic = False
freq = 440
freqFund = 440
runOnce = False
delaiinit = 10

deviceout = a.PCM()

deviceout.setchannels(CHANNELS)
deviceout.setrate(RATE)
deviceout.setformat(FORMAT)
deviceout.setperiodsize(BUFFER_SIZE)


devicein = a.PCM()

devicein.setchannels(CHANNELS)
devicein.setrate(RATE)
devicein.setformat(FORMAT)
devicein.setperiodsize(BUFFER_SIZE)


def fftBlack(y):
    """
    do fft with blackmanharris window
    :param y: audio in
    :return: freq domain
    """
    n = len(y)  # length of the signal
    y = y * signal.blackmanharris(len(y))
    Y = fft(y) / n  # fft computing and normalization
    Y = Y[range(int(n / 2))]
    return Y,


def findFreq(spec, inFreq = True):
    """

    :param spec: full spectrum
    :param inFreq:
    :return: fundamental frequency
    """
    global indiceOfFundamental


    if not(inFreq):
        spec = abs(np.array(fftBlack(spec)))
    else:
        spec = abs(np.array(spec))

    # i : indice of max
    print("spec in findFreq")
    print(spec)

    try:
        i = np.argmax(spec)
        print("i in findFreq")
        print(i)
        i = parabolic(np.log(spec), i)[0]
        print("i parabolic")
        print(i)
        indiceOfFundamental = int(i)
        return RATE * i / len(spec)
    except IndexError:
        i = np.argmax(spec[0])
        i = parabolic(np.log(spec[0]), i)[0]
        indiceOfFundamental = int(i)
        return RATE * i / len(spec[0])


def amplificationHarmonic(data):
    """
    call findFreq before this method
    This method amplify the value of the
    :param data:
    :return:
    """
    shift = 50
    numberOfPoints = shift*2
    gaussian = signal.gaussian(numberOfPoints, std=7)
    dataFreq = fftBlack(data)
    dataFreq = dataFreq[0]
    print("lenght of data")
    print(len(dataFreq))
    for i in range(numberOfPoints):
        try:
            print("index")
            print(indiceOfFundamental * 2 - shift + i)
            print("before")

            print(dataFreq[indiceOfFundamental*2-shift+i])
            dataFreq[indiceOfFundamental*2-shift+i] += dataFreq[indiceOfFundamental*2-shift+i]*10**(5*gaussian[i])
            print("after")
            print(dataFreq[indiceOfFundamental * 2 - shift + i])
            print("")
        except IndexError:
            print("index error")
    #print(np.real(ifft(dataFreq)))
    result = np.real(ifft(dataFreq))
    return result


def harmonicMode(data, freq):
    """
    suppress fundamental
    :param data: frequence?
    :param freq: center of filter
    :return: filtered data
    """
    #freqMin = int(freq * 0.9)
    #freqMax = int(freq * 1.1)
    #return bandstop(data, freqMin, freqMax, RATE, zerophase=True)
    data = highpass(data, freq, RATE, zerophase=True)
    print("indiceOfFundamental")
    print(indiceOfFundamental)
    return amplificationHarmonic(data)


def encodeData(frequency, harmonicmode):
    """

    :param frequency: int
    :param harmonicmode: boolean
    :return:
    """
    strFreq = str(frequency)
    nbDigits = len(strFreq)
    data=""
    if nbDigits < 6:
        data = (6-nbDigits)*"0"
    if harmonicmode:
        data = data + strFreq + "1"
    else:
        data = data + strFreq + "0"
    return data

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError

if __name__ == '__main__':
    isharmonic = input("isharmonic ? : ")
    isharmonic = str_to_bool(isharmonic)
    while True:
        #send data through bluetooth to MAX/MSP
        #s = BluetoothSocket(RFCOMM)

        #s.connect((HOST, PORT))

        # s.send(encodeData(freq, isharmonic))
        # #print("actif")
        # #time.sleep(1)
        # isharmonic = s.recv(1024)
        print(isharmonic)

        l, data = devicein.read()
        audio_data= np.fromstring(data, dtype='int16')

        spec = abs(np.array(fftBlack(audio_data)))
        freq = findFreq(spec)
        #supposed that harmonic = 1 or 0
        if isharmonic != washarmonic and isharmonic:
            freqFund = freq
            washarmonic = True
        elif isharmonic == washarmonic and isharmonic:
            freqFund = freq // 2
        elif isharmonic != washarmonic and not(isharmonic):
            washarmonic = False

        if harmonic:

            spec = harmonicMode(audio_data, freqFund)

        else:
            spec = audio_data

        deviceout.write(spec.tobytes())
