import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import stft
import scipy.signal as signal
import pandas
import pywt


def FFT(data):
    Fs = 2000
    N = 2000
    k = np.arange(2000)
    frq = k * Fs / N
    frq1 = frq[range(int(N / 2))]
    data_f = abs(np.fft.fft(data)) / N
    data_f1 = data_f[range(int(N / 2))]

    plt.plot(frq1, data_f1, 'red')
    plt.xlabel('Freq(hz)')
    plt.ylabel('amplitude')
    plt.title("Freq Spectrum")
    plt.show()

def Filter(data):


    N = 2000

    y = fft(data)
    threadhold = 50
    y[threadhold:(N - threadhold)] = 0  # 滤波器
    data_c = ifft(y)
    return data_c
    pass


def STFT(data):

    fs = 1/60
    window = 'hann'
    # frame长度
    n = 10
    x = np.arange(0,len(data))

    plt.plot(x, data)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title("data")
    plt.show()


    # STFT
    f, t, Z = stft(data, fs=fs, window=window, nperseg=n)
    # 求幅值

    Z = np.abs(Z)
    # 如下图所示
    plt.pcolormesh(t, f, Z, vmin=0, vmax=Z.mean() * 10)
    plt.show()


def DWT(data):
    wavename = 'db5'
    cA, cD = pywt.dwt(data, wavename)

    # cD[np.where(cD<1)] = 0
    ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
    yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component


    x = range(len(data))
    # plt.figure(figsize=(12, 9))
    #
    # plt.subplot(311)
    plt.plot(x, ya)
    plt.title('approximated component')
    plt.show()
    #
    # plt.subplot(313)
    # plt.plot(x,yd)
    # plt.title('detailed component')
    # plt.show()


    return ya

    pass

if __name__ == "__main__":
    DataFrame = pandas.read_csv('../data/code-red.csv')
    DataValue = DataFrame.values
    data = DataValue[:,0]
    # data = (data-np.mean(data))/np.sqrt(np.var(data)+1e-6)
    # STFT(data)
    # DWT(data)
    # STFT(FFT(data))

    # STFT(data)
    data = DWT(data)
    FFT(data)
    # x = range(len(data))
    # plt.plot(x,data)
    # plt.title('before filter')
    # plt.show()
    # b,a = signal.butter(8,0.2,'lowpass')
    # data = signal.filtfilt(b,a,data)
    # plt.plot(x,data)
    # plt.title('after filter')
    # plt.show()