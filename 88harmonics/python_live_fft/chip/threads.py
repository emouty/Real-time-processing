import sys
from threading import Thread
from queue import LifoQueue, Full, Empty
import struct
import alsaaudio as alsa
import numpy as np
from scipy import fft, signal, ifft
from parabolic import parabolic
from obspy.signal.filter import bandstop, highpass
from bluetooth import *
from time import sleep


#alsa variables
BUFFER_SIZE = 1024
CHANNELS = 1
RATE = 44100
FORMAT = alsa.PCM_FORMAT_S16_LE


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
    #print("spec in findFreq")
    #print(spec)

    try:
        i = np.argmax(spec)
        print("in try i in findFreq")
        print(i)
        i = parabolic(np.log(spec), i)[0]
        print("i parabolic")
        print(i)
        indiceOfFundamental = int(i)
        return RATE * i / len(spec)
    except IndexError:
        i = np.argmax(spec[0])
        print("in except, i in FindFreq: ", i) 
        i = parabolic(np.log(spec[0]), i)[0]
        print("in except, i parabolic : ", i)
        indiceOfFundamental = int(i)
        print("fundamental freq : ", RATE * i / len(spec[0]))
        print("len of spec", len(spec[0]))
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

class Readdata(Thread):

    """Thread si for the reading of data"""

    def __init__(self,
                 data,
                 atype=alsa.PCM_CAPTURE,
                 nbchannels=1,
                 rate=44100,
                 dataformat=alsa.PCM_FORMAT_S16_LE,
                 periodsize=1024):
        Thread.__init__(self)
        self.inp = alsa.PCM(atype)
        self.inp.setchannels(nbchannels)
        self.inp.setrate(rate)
        self.inp.setformat(dataformat)
        self.inp.setperiodsize(periodsize)
        self.data = data
    def run(self):
        """Code a executer pendant l'execution du thread."""
        global RATE, BUFFER_SIZE
        while True:
            l, temp = self.inp.read()
            temp = np.fromstring(temp, dtype='int16')
            while len(temp) < BUFFER_SIZE:
                print("in while, taille : ",len(temp))
                l, temp = self.inp.read()
                temp = np.fromstring(temp, dtype='int16')
            self.data.put(highpass(temp,
                                   20,
                                   RATE,
                                   zerophase=True),
                          True, 0.1)
            print("data read, taille : ", len(temp))
            print("")


class Dataanalysis(Thread):

    """Thread is for the data analysis"""

    def __init__(self,data, handleddata):
        Thread.__init__(self)
        self.data = data
        self.handleddata =handleddata
    def run(self):
        """Code a executer pendant l'execution du thread."""
        global isharmonic, washarmonic, indiceOfFundamental
        while True:
            #send data through bluetooth to MAX/MSP
            #s = BluetoothSocket(RFCOMM)

            #s.connect((HOST, PORT))

            # s.send(encodeData(freq, isharmonic))
            # #print("actif")
            # #time.sleep(1)
            # isharmonic = s.recv(1024)
            print(isharmonic)
            try:
                audio_data = self.data.get(block=True)
                tempiffull = audio_data
            except Full:
                print("data.get is FULL")
                audio_data = tempiffull
            except Empty:
                print("Queue is EMPTY")
                audio_data = tempiffull
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
                    self.handleddata.put(harmonicMode(audio_data, freqFund))

            else:
                #indiceOfFundamental = indiceOfFundamental // 2
                #spec = amplificationHarmonic(audio_data)
                self.handleddata.put(audio_data)


class Writedata(Thread):

    """Thread is for the writting of data"""

    def __init__(self,
                 data,
                 atype=alsa.PCM_PLAYBACK,
                 nbchannels=1,
                 rate=44100,
                 dataformat=alsa.PCM_FORMAT_S16_LE,
                 periodsize=1024):
        Thread.__init__(self)
        self.out = alsa.PCM(atype)
        self.out.setchannels(nbchannels)
        self.out.setrate(rate)
        self.out.setformat(dataformat)
        self.out.setperiodsize(periodsize)
        self.data = data

    def run(self):
        """Code a executer pendant l'execution du thread."""
        global isharmonic, washarmonic, indiceOfFundamental
        while True:
            #send data through bluetooth to MAX/MSP
            #s = BluetoothSocket(RFCOMM)

            #s.connect((HOST, PORT))

            # s.send(encodeData(freq, isharmonic))
            # #print("actif")
            # #time.sleep(1)
            # isharmonic = s.recv(1024)
            try:
                spec = self.data.get()
                #save is used to prevent audio signal do be empty
                #if hendled data hasn't finished
                self.out.write(spec.tobyte())
                save = spec

            except Empty:
                self.out.write(save.tobytes())
            except Full:
                self.out.write(save.tobytes())



if __name__ == '__main__':
    #isharmonic = input("isharmonic ? : ")
    #isharmonic = str_to_bool(isharmonic)
    isharmonic = True
    indata = LifoQueue()
    outdata = LifoQueue()

    #creation of the threads
    readthread = Readdata(data=indata)
    handlethread = Dataanalysis(data=indata,handleddata=outdata)
    writethread = Writedata(data=outdata)

    #start of the threads
    readthread.start()
    handlethread.start()
    sleep(0.15)
    writethread.start()

    #readthread.join()
    #writethread.join()

