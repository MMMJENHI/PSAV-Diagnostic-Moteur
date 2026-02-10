import numpy as np

class PSAVEngine:
    def __init__(self, samplerate, window_ms=100, factor=2.5, deadzone_ms=150):
        self.samplerate = samplerate
        self.window_size = int((window_ms / 1000) * samplerate)
        self.factor = factor
        self.deadzone_samples = int((deadzone_ms / 1000) * samplerate)
        self.offset_samples = int(0.010 * samplerate)

    def analyze(self, signal):
        abs_signal = np.abs(signal)
        thresholds = np.zeros(len(signal))
        detections = []
        last_det = -self.deadzone_samples
        
        cumsum = np.cumsum(np.insert(abs_signal, 0, 0))

        for i in range(len(signal)):
            end_win = max(0, i - self.offset_samples)
            start_win = max(0, end_win - self.window_size)
            
            if end_win > start_win:
                avg = (cumsum[end_win] - cumsum[start_win]) / (end_win - start_win)
            else:
                avg = np.mean(abs_signal[:500]) if len(abs_signal) > 500 else 0.01
            
            thresholds[i] = avg * self.factor
            
            if abs_signal[i] > thresholds[i]:
                if (i - last_det) > self.deadzone_samples:
                    detections.append(i)
                    last_det = i
                    
        return np.array(detections), thresholds
