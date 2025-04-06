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
    def __init__(self, 
                 csv_path='mapping.csv',
                audio_root='',
                max_layers=10,
                distance_volume_range=10.0,
                fade_in_ms=1000,
                fade_out_ms=1000,
                loop_enabled=False,
                loop_duration=10.0):
        """
        
        csv_path: path to CSV containing (filename, valence, arousal) mapping
        audio_root: optional folder path where audio files are stored
        layer_multiplier: the maximum number of layers for intensity=1.0.
            e.g. if intensity=1.0 and layer_multiplier=3, we might play 3 overlapped samples.
        fade_in_ms: fade-in duration in milliseconds for each segment.
        fade_out_ms: fade-out duration in milliseconds for each segment.
        loop_enabled: if True, each played segment is repeatedly looped for 'loop_duration' seconds.
        loop_duration: total time in seconds to keep re-playing each sample if loop_enabled is True.
        """
        self.audio_data = []
        self.loaded_samples = {}
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.max_layers = max_layers
        self.distance_volume_range = distance_volume_range
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.loop_enabled = loop_enabled
        self.loop_duration = loop_duration

        self.load_csv(csv_path, audio_root)

        self.run_audio = True
        self.thread = threading.Thread(target=self.audio_thread_loop, daemon=True)
        
        # Shared data structure for incoming signals.
        self.incoming_signals = []
        # track child threads for playback in this list so we can join them on stop.
        self.play_threads = []

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
                # skip first lines if they're headers, adjust if needed.
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

    def play_segment_sync(self, audio_segment):
        # Blocking play, the user can keep calling it in loops or separate threads.
        play(audio_segment)

    def audio_thread_loop(self):
        """
        Continuously looks for new signals. For each signal, determine
        the best matching sample, scale volume by intensity, and play (mix).
        """
        while self.run_audio:
            if self.incoming_signals:
                valence, arousal, intensity = self.incoming_signals.pop(0)
                
                # compute how many samples to play at once
                n_to_play = max(1, int(round(self.max_layers * max(0, min(1, intensity)))))
                top_samples = self.find_top_samples(valence, arousal, n_to_play)

                if not top_samples:
                    continue

                overall_db = -30 + 30 * max(0, min(1, intensity))

                # Among top_samples, find the min and max distances.
                dists = [d for (_, d) in top_samples]
                dist_min = min(dists)
                dist_max = max(dists)
                dist_range = dist_max - dist_min

                # For each sample in top_samples, compute a distance-based offset.
                for (filename, dist) in top_samples:
                    seg = self.loaded_samples.get(filename)
                    if seg is None:
                        continue

                    seg = seg.fade_in(self.fade_in_ms).fade_out(self.fade_out_ms)

                    # if all distances are the same, we'll do alpha=0.
                    if dist_range == 0:
                        alpha = 0.0
                    else:
                        alpha = (dist - dist_min) / dist_range

                    # alpha=0 => closest => 0 dB offset, alpha=1 => farthest => -distance_volume_range dB
                    distance_db = - self.distance_volume_range * alpha

                    # total offset = overall_db + distance_db
                    final_db = overall_db + distance_db

                    seg = seg + final_db


                    if self.loop_enabled:
                        t = threading.Thread(target=self._loop_segment, args=(seg, self.loop_duration))
                    else:
                        t = threading.Thread(target=self.play_segment_sync, args=(seg,))

                    # set daemon=False so it won't be killed abruptly
                    t.daemon = False
                    t.start()

                    # track it so we can join later
                    self.play_threads.append(t)
            else:
                time.sleep(0.05)

    def find_top_samples(self, valence, arousal, n):
        """
        Return the top n samples by ascending distance from (valence, arousal).
        Each return item is (filename, distance).
        If we have fewer than n samples in total, we return them all.
        """
        if not self.audio_data:
            return []

        # compute distances
        dist_list = []  # (filename, distance)
        for (fn, v, a) in self.audio_data:
            dist = math.sqrt((v - valence)**2 + (a - arousal)**2)
            dist_list.append((fn, dist))
        # sort ascending by distance
        dist_list.sort(key=lambda x: x[1])

        # take up to n.
        return dist_list[:n]

    def _loop_segment(self, audio_segment, loop_secs):
        """
        Re-play 'audio_segment' for up to loop_secs seconds.
        """
        start_time = time.time()
        while time.time() - start_time < loop_secs:
            self.play_segment_sync(audio_segment)

    def find_closest_sample(self, valence, arousal, randomness_amt=0.0):
        """
        Finds all samples' distances to (valence, arousal). Then picks
        the best sample OR a random sample within (best_distance + randomness_amt).
        
        If randomness_amt = 0.0, always picks the closest sample.
        Otherwise, picks randomly among those within that distance threshold.
        """
        # Gather (filename, distance) for each sample
        """
        Return a random candidate among the near-closest matches if randomness_amt>0.
        """
        if not self.audio_data:
            return None
        distances = []
        for (fn, v, a) in self.audio_data:
            dist = math.sqrt((v - valence)**2 + (a - arousal)**2)
            distances.append((fn, dist))
        min_distance = min(d[1] for d in distances)
        threshold = min_distance + randomness_amt
        candidates = [fn for (fn, dist) in distances if dist <= threshold]
        if not candidates:
            return None
        return random.choice(candidates)


if __name__ == "__main__":
    # Example usage:
    # Let's assume your CSV is "audio_data.csv"
    # and all WAVs are nested somewhere within /path/to/Emo-Soundscapes-Audio/
    mapper = AudioSentimentMapper(
        csv_path="mapping.csv",
        audio_root="Emo-Soundscapes/Emo-Soundscapes-Audio/",
        max_layers=10,                # up to 10 distinct samples for intensity=1
        distance_volume_range=10.0,   # 10 dB difference from closest to farthest in the top set
        fade_in_ms=1000,         # 1-second fade in
        fade_out_ms=1000,        # 1-second fade out
        loop_enabled=False,      # set True if you want looping
        loop_duration=2.0       # how long to keep re-playing if looping
    );

    mapper.start()

    # Push some test signals
    test_signals = [
        (-0.002, 0.92, 0.5),
        (0.902, 0.902, 0.2),
        # (0.5, 0.0, 0.8),
        # (0.9, 0.5, 1.0),
        # (-0.5, 0.2, 0.3),
        # (0.5, 0.0, 0.8),   # moderate valence/arousal, strong intensity
        # (0.9, 0.5, 1.0),   # high val/ar, full intensity => multiple layers, full volume
        # (-0.5, 0.2, 0.3),  # negative val, low intensity => fewer layers, lower volume
        # (-0.004, 0.9221, 0.39),
        # (0.91102, 0.902, 0.9),
        # (0.225, 0.10, 0.8),
        # (0.129, 0.15, 1.0),
        # (-0.425, 0.52, 0.3),
        # (0.325, 0.10, 0.8),   
        # (0.129, 0.15, 1.0),  
        # (-0.425, 0.32, 0.3),  
    ]

    # simulate the incoming stream of test signals with 0.1 gap in between signals sent
    for sig in test_signals:
        mapper.push_signal(*sig)
        time.sleep(0.1)

    mapper.stop()