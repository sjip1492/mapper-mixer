# Approach Using sounddevice, replacing pydub/ffplay with a callback-based approach that does real-time mixing.
# This avoids spawning external processes and typically provides more control over how (and when) audio is mixed.
#
# Usage Outline:
# 1. We load all audio files from the CSV into memory as NumPy arrays.
# 2. We open a sounddevice.OutputStream that calls our 'audio_callback' to fill audio buffers in real time.
# 3. We maintain a list of 'voices' that are currently playing. Each voice has:
#    - wave data array
#    - sample rate
#    - current read position in frames
#    - volume
#    - fade_in/out logic (if requested)
#    - loop parameters (time limit, etc.)
# 4. When a new signal arrives, we pick up to N samples (like before), and create new voices for them.
# 5. The audio_callback sums all active voices.
# 6. We remove a voice once it finishes or once it hits its loop_duration.
#
import csv
import math
import glob
import os
import random
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
# save numpy array as npy file
from numpy import asarray
from numpy import save
from numpy import load
################################################################################
# Data structures
################################################################################

class Voice:
    """
    Represents a single active playback.
    """
    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        volume: float,
        fade_in_frames: int,
        fade_out_frames: int,
        loop_enabled: bool,
        loop_duration: float,
        start_time: float,
        randomness_amt = float
    ):
        """
        samples: float32 stereo data, shape (num_frames, 2)
        sample_rate: typically 44100 or 48000
        volume: linear gain for the entire playback
        fade_in_frames: how many frames to fade in
        fade_out_frames: how many frames to fade out
        loop_enabled: whether we keep looping until loop_duration
        loop_duration: how many seconds to keep re-triggering the sample
        start_time: the absolute time we created this voice (time.time())
        randomness_amt: threshold distance for near-closest selection. incomign signals only 0.01-precision, whereas samples are scored at 0.000000001 precision
        """
        self.samples = samples
        self.sample_rate = sample_rate
        self.volume = volume
        self.fade_in_frames = fade_in_frames
        self.fade_out_frames = fade_out_frames
        self.loop_enabled = loop_enabled
        self.loop_duration = loop_duration
        self.start_time = start_time
        self.randomness_amt = randomness_amt

        # internal tracking
        self.current_pos = 0  # current read frame in the sample
        self.done = False     # set True when we finish playback
        self.num_frames = samples.shape[0]
        # if we are looping, we might re-trigger from pos=0 multiple times

    def mix_into_buffer(self, outdata: np.ndarray, frames: int, time_now: float):
        """
        Write up to 'frames' samples into 'outdata'.
        outdata shape is (frames, 2)
        time_now is the current absolute time.
        """
        out_len = len(outdata)
        # We'll fill the buffer from self.samples, applying fade in/out.

        # fade in range is [0..fade_in_frames)
        # fade out range is [num_frames - fade_out_frames..num_frames)
        # if looping, we might have partial fade in/out on each iteration

        # We'll do a loop that writes sample data until we fill outdata.
        # Because out_len might be large, we can read from our sample.
        # If we reach the end of the sample, we check if we can loop.

        idx = 0
        while idx < out_len and not self.done:
            frames_left_in_sample = self.num_frames - self.current_pos
            frames_needed = out_len - idx
            block_size = min(frames_left_in_sample, frames_needed)

            # slice of our wave data
            block = self.samples[self.current_pos : self.current_pos + block_size]

            # compute fade factor
            # We'll apply fade in if global playback frame < fade_in_frames
            # but since we might loop multiple times, we also fade in from pos=0 each time.

            fade_curve = np.ones(block_size, dtype=np.float32)

            # fade in region: [0..fade_in_frames]
            if self.current_pos < self.fade_in_frames:
                # number of frames from the start of sample
                fi_start = self.current_pos
                fi_end = min(self.current_pos + block_size, self.fade_in_frames)
                fade_len = fi_end - fi_start
                # create a linear ramp from (fi_start / fade_in_frames) to (fi_end / fade_in_frames)
                # relative positions in block:
                fade_positions = np.arange(fade_len, dtype=np.float32) + fi_start
                fade_factor = fade_positions / float(self.fade_in_frames)
                fade_curve[:fade_len] *= fade_factor

            # fade out region: [num_frames - fade_out_frames..num_frames]
            fo_start_frame = self.num_frames - self.fade_out_frames
            if fo_start_frame < 0:
                fo_start_frame = 0
            if self.current_pos + block_size > fo_start_frame:
                # portion in fade out region
                overlap_start = max(0, fo_start_frame - self.current_pos)
                for bpos in range(overlap_start, block_size):
                    abs_frame = self.current_pos + bpos
                    if abs_frame >= fo_start_frame:
                        # e.g. alpha from 1.0 down to 0
                        fade_out_pos = abs_frame - fo_start_frame
                        fade_factor = 1.0 - (fade_out_pos / float(self.fade_out_frames))
                        fade_curve[bpos] *= fade_factor

            # multiply block by fade_curve and volume
            # shape (block_size, 2)
            block = block * fade_curve.reshape(-1, 1) * self.volume

            # add to outdata
            outdata[idx : idx + block_size] += block

            # advance
            idx += block_size
            self.current_pos += block_size

            # if we are at the end of the sample, check looping
            if self.current_pos >= self.num_frames:
                # we've hit the end
                if self.loop_enabled:
                    # how long have we been playing?
                    elapsed = time_now - self.start_time
                    if elapsed < self.loop_duration:
                        # reset for next iteration
                        self.current_pos = 0
                    else:
                        # done
                        self.done = True
                else:
                    self.done = True

        if idx < out_len:
            # we've declared 'done', so do nothing for the remainder frames.
            pass

################################################################################
# The real-time mixer class
################################################################################
class RealTimeAudioMixer:
    """
    Manages a callback-based audio stream using sounddevice.
    """
    def __init__(self, samplerate=48000, blocksize=1024, channels=2):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.voices = []  # list of Voice
        self.lock = threading.Lock()

        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=self.channels,
            callback=self.audio_callback,
            dtype=np.float32,
        )

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def add_voice(self, voice: Voice):
        with self.lock:
            self.voices.append(voice)

    def audio_callback(self, outdata, frames, time_info, status):
        """
        This is called repeatedly by sounddevice.
        outdata shape: (frames, channels)
        """
        if status:
            print(f"Audio callback status: {status}")
        # zero the buffer
        outdata[:] = 0.0

        now = time.time()
        with self.lock:
            # mix each voice
            dead_voices = []
            for voice in self.voices:
                if not voice.done:
                    voice.mix_into_buffer(outdata, frames, now)
                if voice.done:
                    dead_voices.append(voice)

            # remove finished voices
            for dv in dead_voices:
                self.voices.remove(dv)


################################################################################
# Our main class that picks samples for each signal.
################################################################################

class AudioSentimentMapper:
    def __init__(
        self,
        csv_path='mapping-society-sounds.csv',
        audio_root='',
        max_layers=10,
        distance_volume_range=10.0,
        fade_in_ms=1000,
        fade_out_ms=1000,
        loop_enabled=False,
        loop_duration=10.0,
        samplerate=44100,
        randomness_amt =0.5
    ):
        """
        Similar arguments as before, but now we do real-time mixing.
        samplerate is used when reading audio or re-sampling.
        """
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.max_layers = max_layers
        self.distance_volume_range = distance_volume_range
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.loop_enabled = loop_enabled
        self.loop_duration = loop_duration
        self.samplerate = samplerate
        self.randomness_amt = randomness_amt

        self.audio_data = []  # list of (filename, val, aro)
        self.loaded_samples = {}  # dict filename -> (np.float32 array shape (N,2), sr)

        # Worker thread for signals
        self.run_thread = True
        self.thread = threading.Thread(target=self.audio_thread_loop, daemon=True)
        self.incoming_signals = []
        self.thread_lock = threading.Lock()

        # Our real-time mixer
        self.mixer = RealTimeAudioMixer(samplerate=self.samplerate)

        self.load_csv()

    def load_csv(self):
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) < 3:
                    continue
                if i < 2:  # skip header lines if needed
                    continue
                filename, valence_str, arousal_str = row[0], row[1], row[2]
                valence = float(valence_str)
                arousal = float(arousal_str)
                self.audio_data.append((filename, valence, arousal))

                if filename not in self.loaded_samples:
                    audio_path = self.find_audio_file(filename)
                    if audio_path is None:
                        print(f"Warning: could not find file '{filename}' in '{self.audio_root}'")
                    else:
                        # load with soundfile
                        data, sr = sf.read(audio_path, always_2d=True)
                        # data shape: (num_frames, channels)
                        # convert to float32
                        data = data.astype(np.float32)

                        # resample if needed
                        if sr != self.samplerate:
                            # for a robust approach, use e.g. librosa or samplerate library. We'll do naive approach.
                            # We'll skip resample for brevity.
                            print(f"[WARN] {filename} has sr={sr}, but we want {self.samplerate}. Audio might play at wrong speed.")

                        # ensure stereo
                        if data.shape[1] == 1:
                            # duplicate channel
                            data = np.repeat(data, 2, axis=1)
                        self.loaded_samples[filename] = (data, sr)

    def find_audio_file(self, filename):
        pattern = os.path.join(self.audio_root, '**', filename)
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return None
        return matches[0]

    def start(self):
        self.mixer.start()
        self.thread.start()

    def stop(self):
        self.run_thread = False
        self.thread.join()
        self.mixer.stop()

    def push_signal(self, valence, arousal, intensity):
        with self.thread_lock:
            self.incoming_signals.append((valence, arousal, intensity))

    def audio_thread_loop(self):
        while self.run_thread:
            signal = None
            with self.thread_lock:
                if self.incoming_signals:
                    signal = self.incoming_signals.pop(0)

            if signal:
                valence, arousal, intensity = signal
                n_to_play = max(1, int(round(self.max_layers * max(0, min(1, intensity)))))
                top_samples = self.find_top_samples_randomness(valence, arousal, n_to_play)

                if not top_samples:
                    continue

                # overall dB in [-30..0]
                overall_db = -30 + 30 * max(0, min(1, intensity))
                overall_gain = 10 ** (overall_db / 20.0)  # linear

                # distance-based offsets
                dists = [d for (_, d) in top_samples]
                dist_min = min(dists)
                dist_max = max(dists)
                dist_range = dist_max - dist_min

                for (filename, dist) in top_samples:
                    loaded = self.loaded_samples.get(filename)
                    if not loaded:
                        continue
                    samples, sr = loaded

                    if dist_range == 0:
                        alpha = 0.0
                    else:
                        alpha = (dist - dist_min) / dist_range

                    # alpha=0 => closest => 0 dB offset, alpha=1 => -distance_volume_range dB
                    dist_db = - self.distance_volume_range * alpha
                    dist_gain = 10 ** (dist_db / 20.0)

                    final_gain = overall_gain * dist_gain

                    # create a Voice
                    fade_in_frames = int((self.fade_in_ms / 1000.0) * sr)
                    fade_out_frames = int((self.fade_out_ms / 1000.0) * sr)

                    voice = Voice(
                        samples=samples,
                        sample_rate=sr,
                        volume=final_gain,
                        fade_in_frames=fade_in_frames,
                        fade_out_frames=fade_out_frames,
                        loop_enabled=self.loop_enabled,
                        loop_duration=self.loop_duration,
                        start_time=time.time(),
                        randomness_amt=random
                    )
                    print(f"Signal playing with valence {valence}, arousal {arousal}, intensity {intensity}")
                    self.mixer.add_voice(voice)
            else:
                time.sleep(0.05)

    def find_top_samples_randomness(self, valence, arousal, n):
        """
        Sort by distance ascending. Let best_dist be the min distance.
        Let threshold = best_dist + randomness_amt.
        Then collect all samples with distance <= threshold.
        Shuffle them, pick up to n.
        """
        if not self.audio_data:
            return []
        dist_list = []
        for (fn, v, a) in self.audio_data:
            dist = math.sqrt((v - valence)**2 + (a - arousal)**2)
            dist_list.append((fn, dist))
        # sort ascending
        dist_list.sort(key=lambda x: x[1])
        if not dist_list:
            return []

        best_dist = dist_list[0][1]
        threshold = best_dist + self.randomness_amt
        # collect all up to threshold
        candidates = [item for item in dist_list if item[1] <= threshold]
        random.shuffle(candidates)
        # pick up to n
        return candidates[:n]

################################################################################
# Example usage
################################################################################

def test():
    mapper = AudioSentimentMapper(
        csv_path="mapping-society-sounds.csv",
        audio_root="Emo-Soundscapes/Emo-Soundscapes-Audio/600_Sounds/society",
        max_layers=15,
        distance_volume_range=10.0,
        fade_in_ms=500,
        fade_out_ms=500,
        loop_enabled=False,
        loop_duration=2.0,
        samplerate=44100,
        randomness_amt=0.05 #threshold distance for near-closest selection. incomign signals only 0.01-precision, whereas samples are scored at 0.000000001 precision
    )

    mapper.start()

    test_signals = []

    # Valence, Arousal in [-1, 1] stepping by 0.25
    vals = np.arange(-1.0, 1.0001, 0.25)
    # Intensity in [0, 1] stepping by 0.25
    ints = np.arange(0.0, 1.0001, 0.25)

    for v in vals:
        for a in vals:
            for i in ints:
                # Round for neatness
                test_signals.append( (round(v,2), round(a,2), round(i,2)) )

    for sig in test_signals:
        mapper.push_signal(*sig)
        time.sleep(2.0)

    print("Sleeping 10s...")
    time.sleep(10)

    print("Stopping...")
    mapper.stop()

if __name__ == "__main__":
    test()
