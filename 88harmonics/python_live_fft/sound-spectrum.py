#!/usr/bin/env python
# Written by Yu-Jie Lin
# Public Domain
#
# Deps: PyAudio, NumPy, and Matplotlib
# Blog: http://blog.yjl.im/2012/11/frequency-spectrum-of-sound-using.html

import struct
import wave

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

SAVE = 0.0
TITLE = ''
WIDTH = 1280
HEIGHT = 720
FPS = 25.0*4

nFFT = 1028
BUF_SIZE = 4 * nFFT
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


def animate(i, line, stream, wf, MAX_y):
    # Read n*nFFT frames from stream, n > 0
    try:
        N = max(stream.get_read_available() / nFFT, 1) * nFFT
    except (SystemError, OSError):
        stream = pyaudio.PyAudio().open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=BUF_SIZE)
        N = max(stream.get_read_available() / nFFT, 1) * nFFT

    data = stream.read(int(N))
    #if SAVE:
    #    wf.writeframes(data)

    # Unpack data, LRLRLR...
    y = np.array(struct.unpack("%dh" % (N * CHANNELS), data)) / MAX_y
    y_L = y[::2]
    y_R = y[1::2]

    Y_L = np.fft.fft(y_L, nFFT)
    Y_R = np.fft.fft(y_R, nFFT)

    # Sewing FFT of two channels together, DC part uses right channel's
    Y = abs(np.hstack((Y_L[-nFFT / 2:-1], Y_R[:nFFT / 2])))*10

    line.set_ydata(Y)
    return line,


def init(line):
    # This data is a clear frame for animation
    line.set_ydata(np.zeros(nFFT - 1))
    return line,


def main():
    dpi = plt.rcParams['figure.dpi']
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams["figure.figsize"] = (1.0 * WIDTH / dpi, 1.0 * HEIGHT / dpi)

    fig = plt.figure()

    # Frequency range
    x_f = 1.0 * np.arange(-nFFT / 2 + 1, nFFT / 2) / nFFT * RATE
    ax = fig.add_subplot(111, title=TITLE, xlim=(x_f[0], x_f[-1]),
                         ylim=(0, 2 * np.pi * nFFT ** 2 / RATE))
    ax.set_yscale('symlog', linthreshy=nFFT ** 0.5)

    line, = ax.plot(x_f, np.zeros(nFFT - 1))

    # Change x tick labels for left channel
    def change_xlabel(evt):
        labels = [label.get_text().replace(u'\u2212', '')
                  for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        fig.canvas.mpl_disconnect(drawid)

    drawid = fig.canvas.mpl_connect('draw_event', change_xlabel)

    p = pyaudio.PyAudio()
    # Used for normalizing signal. If use paFloat32, then it's already -1..1.
    # Because of saving wave, paInt16 will be easier.
    MAX_y = 2.0 ** (p.get_sample_size(FORMAT) * 8 - 1)

    frames = None
    wf = None
    # if SAVE:
    #     frames = int(FPS * SAVE)
    #     wf = wave.open('temp.wav', 'wb')
    #     wf.setnchannels(CHANNELS)
    #     wf.setsampwidth(p.get_sample_size(FORMAT))
    #     wf.setframerate(RATE)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=BUF_SIZE)

    ani = animation.FuncAnimation(
        fig, animate, frames,
        init_func=lambda: init(line), fargs=(line, stream, wf, MAX_y),
        interval=1000.0 / FPS, blit=False
    )

    if SAVE:
        ani.save('temp.mp4', fps=FPS)
    else:
        plt.show()

    stream.stop_stream()
    stream.close()
    p.terminate()

    if SAVE:
        wf.close()


if __name__ == '__main__':
    soundObj = pyaudio.PyAudio()

    # Learn what your OS+Hardware can do
    defaultCapability = soundObj.get_default_host_api_info()
    print(defaultCapability)

    # See if you can make it do what you want
    isSupported = soundObj.is_format_supported(input_format=pyaudio.paInt16, input_channels=2, rate=44100,
                                               input_device=0)
    print(isSupported)
    main()
