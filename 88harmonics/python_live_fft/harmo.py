import sys
from threading import Thread, RLock
from queue import Queue, LifoQueue, Full, Empty
import struct
import alsaaudio as alsa
import numpy as np
from scipy import fft, signal, ifft
from parabolic import parabolic
from obspy.signal.filter import bandstop, highpass, bandpass
from bluetooth import *
from time import sleep
import math
import struct


#alsa variables
BUFFER_SIZE = 1024
CHANNELS = 1
RATE = 44100
FORMAT = alsa.PCM_FORMAT_S16_LE


#bluetooth config

HOST = "A4:D1:8C:D2:52:39"      # The remote host, MAC adress of the bluetooth of your computer
PORT = 3              # Server port

#global variables
indiceOfFundamental = 0
harmonic = False
isharmonic = False
washarmonic = False
freq = 440
freqFund = 440
runOnce = False
delaiinit = 10
lock = RLock()


def sine_wave(frequency=440.0, startat=0, length=2048, framerate=44100, amplitude=0.5):
    return float(amplitude) * np.sin(2.0*math.pi*float(frequency)*(np.arange(start=startat, stop=length)/float(framerate)))


def fftBlack(y):
    """
    do fft with blackmanharris window
    :param y: audio in
    :return: freq domain
    """
    n = len(y)  # length of the signal
    y *= signal.blackmanharris(len(y))
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
        #print("in try i in findFreq", i)
        i = parabolic(np.log(spec), i)[0]
        #print("i parabolic", i)
        indiceOfFundamental = int(i)
        return RATE * i / len(spec)
    except IndexError:
        i = np.argmax(spec[0])
        print("in except, i in FindFreq: ", i) 
        i = parabolic(np.log(spec[0]), i)[0]
        #print("in except, i parabolic : ", i)
        indiceOfFundamental = int(i)
        print("fundamental freq : ", RATE * i / len(spec[0]) / 2)
        #print("len of spec", len(spec[0]))
        return RATE * i / len(spec[0]) / 2


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
    print("lenght of data in amp harmo", len(data))
    for i in range(numberOfPoints):
        try:
            #print("index")
            #print(indiceOfFundamental * 2 - shift + i)
            #print("before")

            #print(dataFreq[indiceOfFundamental*2-shift+i])
            dataFreq[indiceOfFundamental*2-shift+i] += dataFreq[indiceOfFundamental*2-shift+i]*10**(5*gaussian[i])
            #print("after")
            #print(dataFreq[indiceOfFundamental * 2 - shift + i])
            #print("")
        except IndexError:
            print("index error in amp harmo")
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
    #data = bandpass(data, freqMin, freqMax, RATE, zerophase=True)
    print("indiceOfFundamental", indiceOfFundamental)
    return amplificationHarmonic(data)
    #return data


def harmonicMode2(data, freq):
    """
    keep only the first harmonic
    :param data: frequence?
    :param freq: center of filter
    :return: filtered data
    """
    freqMin = int(freq * 0.9)
    freqMax = int(freq * 1.1)
    #return bandstop(data, freqMin, freqMax, RATE, zerophase=True)
    #data = highpass(data, freq, RATE, zerophase=True)
    data = bandpass(data, freqMin, freqMax, RATE, zerophase=True)
    print("indiceOfFundamental", indiceOfFundamental)
    #return amplificationHarmonic(data)
    return data*2


def encodeData(frequency, harmonicmode):
    """
    encode the data in order to send it to MAXMSP
    :param frequency: int
    :param harmonicmode: boolean
    :return:
    """
    # strFreq = str(frequency)
    # nbDigits = len(strFreq)
    # data=""
    # if nbDigits < 6:
    #     data = (5-nbDigits)*"0"
    # if harmonicmode:
    #     data = data + strFreq + "2"
    # else:
    #     data = data + strFreq + "1"
    # return data

    data = [0, 0, 0, 0, 0, 1]

    if harmonicmode:
        data[5] = 2
    for i in range(1,5):
        data[i] = frequency % 10
        frequency -= data[i]
        frequency //= 10
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
            while len(temp) < BUFFER_SIZE*2:
                print("in while, taille : ", len(temp))
                l, temp = self.inp.read()
            self.data.put(highpass(np.fromstring(temp, dtype='int16'),
                                   20,
                                   RATE,
                                   zerophase=True),
                          True, 0.1)
            print("data read, taille : ", len(temp))
            print("")


class Dataanalysis(Thread):

    """Thread is for the data analysis"""

    def __init__(self,data, handleddata, frequencyqueue):
        Thread.__init__(self)
        self.data = data
        self.handleddata = handleddata
        self.frequenqueue = frequencyqueue

    def run(self):
        """Code a executer pendant l'execution du thread."""
        global isharmonic, washarmonic, indiceOfFundamental
        start = 0
        end = 2048
        while True:
            with lock:
                harmonicmode = isharmonic
                print("harmonique mode = ", isharmonic)
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
            #for bluetooth
            self.frequenqueue.put(freq)

            #supposed that harmonic = 1 or 0
            if harmonicmode != washarmonic and harmonicmode:
                freqFund = freq
                washarmonic = True
            elif harmonicmode == washarmonic and harmonicmode:
                freqFund = freq // 2
            elif harmonicmode != washarmonic and not(harmonicmode):
                washarmonic = False
                freqFund = freq // 2
                #not sure
            else:
                freqFund = freq

            if harmonicmode:
                #self.handleddata.put(harmonicMode(audio_data, freqFund))
                #self.handleddata.put(harmonicMode2(audio_data, freqFund*2))
                print(harmonicmode)
            else:
                #indiceOfFundamental = indiceOfFundamental // 2
                #spec = amplificationHarmonic(audio_data)
                #self.handleddata.put(audio_data)
                data = sine_wave(freqFund,start,end)
                self.handleddata.put(data)
                if data[2047] < 10**(-2):
                    start = 0
                    end = 2048
                else:
                    start = end
                    end += 2048




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
        #sleep is used to wait for the first data to be read & handled
        sleep(0.2)

    def run(self):
        """Code a executer pendant l'execution du thread."""
        global indiceOfFundamental
        while True:
            #print("IN RUN OF WRITE")
            try:
                spec = self.data.get()
                #save is used to prevent audio signal do be empty
                #if hendled data hasn't finished

                #self.out.write(spec.tobytes())

                self.out.write(struct.pack('<i', spec))
                save = spec

            except Empty:
                print("write out queue EMPTY")
                #self.out.write(save.tobytes())
                self.out.write(struct.pack('<i', spec))
            except Full:
                print("write out queue FULL")
                #self.out.write(save.tobytes())
                self.out.write(struct.pack('<i', spec))

class Bluetoothdata(Thread):

    """Thread is for the writting of data"""

    def __init__(self,
                 frequencyqueue):
        Thread.__init__(self)
        self.frequencyqueue = frequencyqueue
        self.socket = BluetoothSocket(RFCOMM)
        self.socket.connect((HOST, PORT))

    def run(self):
        """Code a executer pendant l'execution du thread."""
        global isharmonic
        while True:
            #send data through bluetooth to MAX/MSP
            try:
                freq = self.frequencyqueue.get(block=True, timeout=1)
                lastfreqget = freq
                print("SEND DATA")
                with lock:
                    encodeddata = encodeData(freq, isharmonic)
                for i in encodeddata:
                    self.socket.send(i)
            except (Full, Empty):
                encodeddata = encodeData(lastfreqget, isharmonic)
                for i in encodeddata:
                    self.socket.send(i)
                print("no data to send")
                sleep(0.5)
            # except Empty:
            #     #self.socket.send(encodeData(lastfreqget,isharmonic))
            #     print("no data to send")
            #     sleep(0.5)
                
            # #print("actif")
            # #time.sleep(1)
            rawbluetoothdata = self.socket.recv(1024)
            handledbluetoothdata = int.from_bytes(rawbluetoothdata, byteorder='little')
            if handledbluetoothdata == 2:
                with lock:
                    isharmonic = True
            else:
                with lock:
                    isharmonic = False


if __name__ == '__main__':
    #isharmonic = input("isharmonic ? : ")
    #isharmonic = str_to_bool(isharmonic)
    #isharmonic = True

    indata = LifoQueue()
    outdata = LifoQueue()
    freqqueue = Queue()
    #creation of the threads
    readthread = Readdata(data=indata)
    handlethread = Dataanalysis(data=indata,
                                handleddata=outdata,
                                frequencyqueue=freqqueue)
    writethread = Writedata(data=outdata)
    #sendthread = Bluetoothdata(freqqueue)

    #start of the threads
    readthread.start()
    handlethread.start()
    writethread.start()
    #sendthread.start

    #readthread.join()
    #writethread.join()

