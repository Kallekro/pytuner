import pyaudio
import wave
import struct
import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import argparse

# Constants for recording
CHUNK = 1024 * 2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Only look for peaks in this range
FREQ_DETECT_RANGE = (60, 350)
# Frequencies of strings on standard guitar tuning
TONE_FREQS = [82.41, 110.0, 146.83, 196.0, 246.94, 329.63]
TONES      = [ "E",   "A",   "D",    "G",   "B",    "E"  ]

class SoundManager:
    def __init__(self, show_plots):
        self.p = None
        self.stream = None
        self.frames = []
        self.show_plots = show_plots
        self.setup_analyzation()

    def setup_analyzation(self):
        self.tone_frequency = 0
        self.chosen_peak = 0
        self.x_fft = np.linspace(0, RATE, CHUNK)
        if self.show_plots:
            self.setup_plots()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def analyze_sound(self):
        # read data
        in_data = self.stream.read(CHUNK)
        # unpack data
        data_int = struct.unpack(str(2*CHUNK) + 'B', in_data)
        # wave form: convert to np array, only take every second element and offset by 128
        self.y_raw = np.array(data_int, dtype='b')[::2] + 128
        # smooth sound wave
        self.y_smooth = savgol_filter(data_int, 51, 3)
        # frequency spectrum with fast fourier transform 
        self.y_fft = np.abs(fft(self.y_smooth)[0:CHUNK]) / (128 * CHUNK)
        # peaks of fft
        self.fft_peaks_x = self.find_peaks()
        self.determine_tone_frequency()
        print("Current freq: %f" % self.tone_frequency)
        if self.show_plots:
            self.plot_sound()

    def find_peaks(self):
        peaks = find_peaks(self.y_fft, threshold=0.01, height=0.1)[0]
        # remove peaks not in valid range
        valid_peaks = []
        for peak in peaks:
            if self.x_fft[peak] >= FREQ_DETECT_RANGE[0] and self.x_fft[peak] <= FREQ_DETECT_RANGE[1]:
                valid_peaks.append(peak)
        return np.array(valid_peaks)

    def determine_tone_frequency(self):
        if not len(self.fft_peaks_x):
            return
        threshold = 0.1
        candidate_freq = self.fft_peaks_x[0]
        for i in range(len(self.fft_peaks_x)):
            if self.y_fft[self.fft_peaks_x[i]] - self.y_fft[candidate_freq] > threshold:
                candidate_freq = self.fft_peaks_x[i]
        self.chosen_peak = candidate_freq
        self.tone_frequency = self.x_fft[self.chosen_peak]

    def record_frames(self, seconds):
        self.frames = []
        if seconds == -1:
            print("recording... interrupt to stop recording")
            while 1:
                try:
                    data = self.stream.read(CHUNK)
                    self.frames.append(data)
                except KeyboardInterrupt:
                    break
        else:
            print("recording for %d seconds..." % seconds)
            for _ in range(0, int(RATE / CHUNK * seconds)):
                data = self.stream.read(CHUNK)
                self.frames.append(data)

    def play_back_frames(self):
        play_stream = self.p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True)
        for data in self.frames:
            play_stream.write(data)
        play_stream.stop_stream()
        play_stream.close()

    def save_frames(self, outname):
        wf = wave.open(outname, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def setup_plots(self):
        self.fig,(ax1, ax2) = plt.subplots(2)

        self.li_raw, = ax1.plot(np.arange(0, 2 * CHUNK, 2), np.random.rand(CHUNK), '-', lw=2)
        self.li_fft, = ax2.semilogx(self.x_fft, np.random.rand(CHUNK), '-', lw=2)
        self.li_fft_peaks, = ax2.plot(self.x_fft, np.random.rand(CHUNK), 'o')
        self.li_fft_chosen_peak, = ax2.plot(self.x_fft, np.random.rand(CHUNK), 'o')

        ax1.set_title("Audio Waveform")
        ax1.set_xlim(0, 2 * CHUNK)
        ax1.set_ylim(0, 255)
        plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0,128,255])

        ax2.set_title("Fast Fourier Transform")
        ax2.set_xlim(20, RATE / 2)

        plt.show(block=False)

    def plot_sound(self):
        # plot 1: raw audio wave plot
        self.li_raw.set_ydata(self.y_raw)

        # plot 2: frequency spectrum
        self.li_fft.set_ydata(self.y_fft)
        
        # plot peaks
        self.li_fft_peaks.set_xdata([self.x_fft[i] for i in self.fft_peaks_x])
        self.li_fft_peaks.set_ydata([self.y_fft[i] for i in self.fft_peaks_x])
        
        # plot chosen peak
        self.li_fft_chosen_peak.set_xdata([self.tone_frequency])
        self.li_fft_chosen_peak.set_ydata([self.y_fft[self.chosen_peak]])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tuner")
    parser.add_argument("--plot", "-p", action='store_true', help="Show plots.")
    args = parser.parse_args()
    with SoundManager(args.plot) as sm:
        while 1:
            try:
                sm.analyze_sound()
            except KeyboardInterrupt:
                break