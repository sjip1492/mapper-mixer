import csv
import math
import glob
import os
from pydub import AudioSegment
from pydub.playback import play
import threading
import time
import random

class AudioSentimentMapper:
    def __init__(self, csv_path='mapping.csv', audio_root=''):
        """
        csv_path: path to CSV containing (filename, valence, arousal) mapping
        audio_root: optional folder path where audio files are stored
        """
        self.audio_data = []
        self.loaded_samples = {}
        self.load_csv(csv_path, audio_root)

        self.run_audio = True
        self.thread = threading.Thread(target=self.audio_thread_loop, daemon=True)
        
        # Shared data structure for incoming signals.
        self.incoming_signals = []

    def load_csv(self, csv_path, audio_root=''):
        """
        Loads CSV rows into self.audio_data = [(filename, valence, arousal), ...].
        Also preloads each audio file into an AudioSegment dictionary for fast access.
        """
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if len(row) < 3:
                    continue
                if i < 2:
                    continue
                filename, valence_str, arousal_str = row[0], row[1], row[2]
                valence = float(valence_str)
                arousal = float(arousal_str)
                self.audio_data.append((filename, valence, arousal))
                
                # Preload the audio file, searching recursively in audio_root
                if filename not in self.loaded_samples:
                    audio_path = self.find_audio_file(audio_root, filename)
                    if audio_path is None:
                        print(f"Warning: could not find file '{filename}' in '{audio_root}'")
                    else:
                        try:
                            self.loaded_samples[filename] = AudioSegment.from_file(audio_path)
                        except Exception as e:
                            print(f"Error loading file {audio_path}: {e}")
    
    def find_audio_file(self, audio_root, filename):
        """
        Recursively search for 'filename' within 'audio_root'.
        If the CSV already has a partial path (e.g. '600_Sounds/human/...'),
        this will also work as long as that partial path is rooted somewhere under 'audio_root'.
        Returns the first matching path found, or None if no match is found.
        """
        # Build a pattern that covers both exact filename and partial subpaths
        # Example pattern:
        #   /Users/.../Emo-Soundscapes-Audio/**/filename
        pattern = os.path.join(audio_root, '**', filename)
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return None
        # Return the first match
        return matches[0]
    

    def start(self):
        """Starts the background thread for audio playback."""
        self.thread.start()

    def stop(self):
        """Stops the audio thread gracefully."""
        self.run_audio = False
        self.thread.join()

    def push_signal(self, valence, arousal, intensity):
        """
        Receives an incoming [valence, arousal, intensity] triple
        We append it to a queue/list that the audio thread will read.
        """
        self.incoming_signals.append((valence, arousal, intensity))
   
    def play_segment_async(self, audio_segment):
        """
        Launch a new thread that calls pydub.play(...) so each new sound can overlap.
        """
        t = threading.Thread(target=play, args=(audio_segment,))
        t.start()

    def audio_thread_loop(self):
        """
        Continuously looks for new signals. For each signal, determine
        the best matching sample, scale volume by intensity, and play (mix).
        
        TODO: sophisticated crossfading or layering of multiple tracks
        """
        while self.run_audio:
            if self.incoming_signals:
                valence, arousal, intensity = self.incoming_signals.pop(0)
                
                # 1) Find the best sample from self.audio_data
                filename = self.find_closest_sample(valence, arousal, 0.05)
                if not filename:
                    continue
                
                base_segment = self.loaded_samples.get(filename)
                if base_segment is None:
                    continue

                # 2) Scale volume based on intensity
                # Map intensity in [0,1] to a decibel range, e.g. -30 dB to 0 dB
                dB_change = -30 + 30 * max(0, min(1, intensity))
                adjusted_segment = base_segment + dB_change

                # 3) Play the segment (blocking call with pydub)
                self.play_segment_async(adjusted_segment)
            else:
                # If no signals are in the queue, just sleep briefly
                time.sleep(0.05)

    def find_closest_sample(self, valence, arousal, randomness_amt=0.0):
        """
        Finds all samples' distances to (valence, arousal). Then picks
        the best sample OR a random sample within (best_distance + randomness_amt).
        
        If randomness_amt = 0.0, always picks the closest sample.
        Otherwise, picks randomly among those within that distance threshold.
        """
        # Gather (filename, distance) for each sample
        distances = []
        for (fn, v, a) in self.audio_data:
            dist = math.sqrt((v - valence) ** 2 + (a - arousal) ** 2)
            distances.append((fn, dist))

        # Find the minimum distance
        if not distances:
            return None
        min_distance = min(d[1] for d in distances)

        # Determine the threshold
        #   e.g. threshold = min_distance + 0.1 => picks all samples within 0.1 of the best distance
        threshold = min_distance + randomness_amt

        # Collect all filenames whose distance is <= threshold
        candidates = [fn for (fn, dist) in distances if dist <= threshold]

        if not candidates:
            return None

        # Randomly pick from the candidate set
        return random.choice(candidates)


if __name__ == "__main__":
    # Example usage:
    # Let's assume your CSV is "audio_data.csv"
    # and all WAVs are nested somewhere within /path/to/Emo-Soundscapes-Audio/
    mapper = AudioSentimentMapper(
        csv_path="mapping.csv",
        audio_root="Emo-Soundscapes/Emo-Soundscapes-Audio/"
    )
    mapper.start()

    # Push some test signals
    test_signals = [
        (-0.002, 0.92, 0.9),
        (0.902, 0.902, 0.9),
        (0.5, 0.0, 0.8),
        (0.9, 0.5, 1.0),
        (-0.5, 0.2, 0.3),
    ]

    # simulate the incoming stream of test signals
    for sig in test_signals:
        mapper.push_signal(*sig)
        time.sleep(0.1)

    # MAKE SURE IT IS CONTINUOUS
    mapper.stop()