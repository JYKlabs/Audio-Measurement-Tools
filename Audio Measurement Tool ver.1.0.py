import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sounddevice as sd
import csv
import os
from datetime import datetime
import json
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import welch
import queue
import threading
import time
import uuid
import webbrowser
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame

# Default settings
THD_frequencies = [997]
frequency_response_frequencies = [20]
num_trials = 3
crosstalk_default_duration = 5.0
frequency_response_default_duration = 3.0

def generate_sine_signal(freq, fs, duration, amplitude_db, dtype=np.float32):
    # Always generate signal in float32, scale based on dtype only at the end
    max_amplitude = 1.0  # Work with normalized amplitude initially
    amplitude = max_amplitude * 10**(amplitude_db / 20.0)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    
    # Scale based on dtype
    if dtype == np.int16:
        signal = signal * 32767
        signal = np.clip(signal, -32767, 32767).astype(np.int16)
    elif dtype == np.int32:
        signal = signal * 8388607
        signal = np.clip(signal, -8388607, 8388607).astype(np.int32)
    else:
        signal = signal.astype(np.float32)
    return signal

def generate_log_sweep(fs, duration, amplitude_db, f_start=None, f_end=None, dtype=np.float32):
    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
    amplitude = max_amplitude * 10**(amplitude_db / 20.0)
    n_fft = int(fs * duration)
    fft_resolution = fs / n_fft
    if f_start is None or f_start <= 0:
        f_start = fft_resolution
    if f_end is None:
        f_end = fs / 2
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    T = duration
    w1 = 2 * np.pi * f_start
    w2 = 2 * np.pi * f_end
    K = T * w1 / np.log(w2 / w1)
    L = T / np.log(w2 / w1)
    sweep = amplitude * np.sin(K * (np.exp(t / L) - 1))
    sweep[-1] = 0
    if dtype in [np.int16, np.int32]:
        sweep = np.clip(sweep, -max_amplitude, max_amplitude).astype(dtype)
    else:
        sweep = np.clip(sweep, -max_amplitude, max_amplitude).astype(dtype)
    return sweep, t

# Designing a low-pass filter according to AES17-2020 (5.2.5)
def design_low_pass_filter(fs, cutoff=20000, order=8):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    # Ensure cutoff is within valid range
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    return b, a

# Designing a notch filter according to AES17-2020 (5.2.8)
def design_notch_filter(freq, Q, fs):
    w0 = freq / (fs / 2)  # Normalized frequency
    b, a = signal.iirnotch(w0, Q)
    return b, a

def generate_output_signal(freq, fs, input_channel=0, num_channels=6, duration=5.0, dtype=np.float32):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    output = np.zeros((len(t), num_channels), dtype=dtype)
    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
    output[:, 0] = max_amplitude * np.sin(2 * np.pi * freq * t)
    crosstalk_level = 0.00001
    for ch in range(num_channels):
        if ch != 0:
            output[:, ch] = crosstalk_level * max_amplitude * np.sin(2 * np.pi * freq * t)
    if dtype in [np.int16, np.int32]:
        output = np.clip(output, -max_amplitude, max_amplitude).astype(dtype)
    else:
        output = output.astype(dtype)
    return output

def design_bandpass_filter(fs, lowcut=10, highcut=None):
    nyquist = fs / 2
    if highcut is None or highcut > nyquist:
        highcut = min(20000, nyquist * 0.99)  # 20 kHz
    if not (0 < lowcut < nyquist and 0 < highcut <= nyquist and lowcut < highcut):
        raise ValueError(f"Invalid filter cutoffs: lowcut={lowcut}, highcut={highcut}, fs={fs}")
    try:
        b, a = signal.butter(2, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
        return b, a
    except Exception as e:
        raise RuntimeError(f"Failed to design bandpass filter: {str(e)}")

def design_crosstalk_bandpass_filter(fs, center_freq=997, bandwidth_ratio=1/3):
    nyquist = fs / 2
    Q = 1 / bandwidth_ratio  # Q factor for third-octave filter (Q ≈ 3)
    lowcut = center_freq / np.sqrt(2 ** bandwidth_ratio)
    highcut = center_freq * np.sqrt(2 ** bandwidth_ratio)
    if lowcut <= 0 or highcut >= nyquist:
        raise ValueError(f"Invalid filter cutoffs: lowcut={lowcut}, highcut={highcut}, fs={fs}")
    b, a = signal.butter(2, [lowcut / nyquist, highcut / nyquist], btype='bandpass')
    return b, a

def compute_time_delay(input_signal, output_signal, fs, log_callback):
    input_signal_norm = input_signal / (np.max(np.abs(input_signal)) + 1e-10)
    output_signal_norm = output_signal / (np.max(np.abs(output_signal)) + 1e-10)
    max_output_amplitude = np.max(np.abs(output_signal))
    if max_output_amplitude < 0.00001:
        log_callback("Error: Recorded signal amplitude too low (< 0.00001) on primary channel.")
        log_callback("Check hardware routing in Audio Interface Panel.")
        return 0
    peak_idx = np.argmax(np.abs(output_signal_norm))
    log_callback(f"Peak detected at sample {peak_idx}")
    window_size = int(fs * 0.2)
    start_idx = max(0, peak_idx - window_size // 2)
    end_idx = min(len(output_signal), peak_idx + window_size // 2)
    output_segment = output_signal_norm[start_idx:end_idx]
    input_segment = input_signal_norm[:len(output_segment)]
    if len(input_segment) < len(output_segment):
        output_segment = output_segment[:len(input_segment)]
    elif len(output_segment) < len(input_segment):
        input_segment = input_segment[:len(output_segment)]
    correlation = signal.correlate(output_segment, input_segment, mode='full')
    delay_idx = np.argmax(correlation)
    max_correlation = np.max(correlation)
    center_idx = len(input_segment) - 1
    delay_samples = delay_idx - center_idx + start_idx
    log_callback(f"Max correlation: {max_correlation:.5f} (index {delay_idx})")
    log_callback(f"Computed delay samples: {delay_samples}")
    if max_correlation < 0.1:
        log_callback("Warning: Low correlation peak - time delay may be inaccurate.")
        return 0
    if delay_samples < 0:
        log_callback(f"Warning: Negative delay ({delay_samples}) detected. Setting to 0.")
        return 0
    return delay_samples

def record_output(input_signal, fs, input_device, output_device, input_channels, output_channel, dtype, log_callback, log_level_details=True):
    try:
        device_info = sd.query_devices()[output_device]
        supported_samplerates = [device_info['default_samplerate']] if isinstance(device_info['default_samplerate'], (int, float)) else device_info['default_samplerate']
        log_callback(f"Requested Sample Rate: {fs} Hz, Supported: {supported_samplerates}")
        log_callback(f"Note: Actual sample rate follows audio interface settings. Please ensure it matches {fs} Hz in your audio interface control panel.")

        sd.default.device = [input_device, output_device]
        sd.default.samplerate = fs
        sd.default.dtype = dtype
        sd.default.clip_off = True
        max_output_channels = device_info['max_output_channels']
        max_input_channels = sd.query_devices()[input_device]['max_input_channels']
        if output_channel >= max_output_channels:
            log_callback(f"Error: Output channel {output_channel + 1} exceeds max ({max_output_channels}). Using channel 0.")
            output_channel = 0
        valid_input_channels = [ch for ch in input_channels if 0 <= ch < max_input_channels]
        if len(valid_input_channels) != len(input_channels):
            log_callback(f"Warning: Invalid input channels. Using: {[ch + 1 for ch in valid_input_channels]}")
        if not valid_input_channels:
            log_callback("Error: No valid input channels provided.")
            return None
        sorted_channels = sorted(set(valid_input_channels))
        log_callback(f"Playing on output channel: {output_channel + 1}")
        log_callback(f"Recording on input channels: {[ch + 1 for ch in sorted_channels]}")
        log_callback("Ensure output channel is routed to input channels in Audio Interface Panel with full gain (0 dB). Verify digital loopback.")
        max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
        if dtype == np.int16:
            log_callback(f"Input signal max amplitude: {np.max(np.abs(input_signal))} (scaled for int16, max={max_amplitude})")
        elif dtype == np.int32:
            log_callback(f"Input signal max amplitude: {np.max(np.abs(input_signal))} (scaled for int32, max={max_amplitude})")
        else:
            log_callback(f"Input signal max amplitude: {np.max(np.abs(input_signal))} (float32, max={max_amplitude})")

        # Step 1: Record 0.1s of silence to estimate noise floor
        silence_duration = 0.1
        silence_samples = int(fs * silence_duration)
        silence_output = np.zeros((silence_samples, max_output_channels), dtype=dtype)
        input_queue_silence = queue.Queue()
        recorded_silence = []

        def input_callback_silence(indata, frames, time, status):
            if status:
                log_callback(f"Silence input stream status: {status}")
            if indata is not None and indata.size > 0:
                if dtype == np.int16:
                    indata = indata * 32767
                elif dtype == np.int32:
                    indata = indata * 8388607
                selected_data = indata[:, sorted_channels]
                input_queue_silence.put(selected_data.copy())

        log_callback("Recording 0.1s of silence for noise floor estimation...")
        input_stream_silence = sd.InputStream(
            samplerate=fs,
            channels=max_input_channels,
            dtype='float32',
            callback=input_callback_silence,
            blocksize=2048,
            latency='high'
        )
        input_stream_silence.start()
        time.sleep(silence_duration + 0.01)
        input_stream_silence.stop()
        input_stream_silence.close()

        while not input_queue_silence.empty():
            recorded_silence.append(input_queue_silence.get())
        if recorded_silence:
            recorded_silence = np.concatenate(recorded_silence, axis=0)[:silence_samples]
            if len(recorded_silence) < silence_samples:
                recorded_silence = np.pad(recorded_silence, ((0, silence_samples - len(recorded_silence)), (0, 0)), mode='constant')
        else:
            recorded_silence = np.zeros((silence_samples, len(sorted_channels)), dtype=np.float32)
            log_callback("Warning: No silence data recorded. Using zeros.")

        noise_floor = 1e-6
        if recorded_silence.size > 0:
            try:
                if dtype in [np.int16, np.int32]:
                    silence_for_welch = recorded_silence.astype(np.float32)
                    if dtype == np.int16:
                        silence_for_welch /= 32767
                    elif dtype == np.int32:
                        silence_for_welch /= 8388607
                else:
                    silence_for_welch = recorded_silence
                f, psd = welch(silence_for_welch[:, 0], fs, nperseg=4096, scaling='density')
                noise_floor = np.sqrt(np.mean(psd))
                log_callback(f"Estimated noise floor from silence (RMS): {noise_floor:.8f}")
                power_line_idx = np.argmin(np.abs(f - 60))
                power_line_psd = psd[power_line_idx]
                log_callback(f"Power line noise (60 Hz) PSD: {power_line_psd:.8f}")
            except Exception as e:
                log_callback(f"Error in silence noise floor estimation: {str(e)}. Setting noise floor to {noise_floor:.8f}")
        log_callback(f"Estimated noise floor: {noise_floor:.8f}")

        # Step 2: Record actual signal
        if input_signal.ndim == 1:
            output_data = input_signal
        else:
            output_data = input_signal[:, 0]
        output_data_full = np.zeros((len(output_data), max_output_channels), dtype=dtype)
        output_data_full[:, output_channel] = output_data
        output_data_full = np.clip(output_data_full, -max_amplitude, max_amplitude)
        if dtype == np.int16:
            log_callback(f"Output signal max amplitude: {np.max(np.abs(output_data_full))} (scaled for int16, max={max_amplitude})")
        elif dtype == np.int32:
            log_callback(f"Output signal max amplitude: {np.max(np.abs(output_data_full))} (scaled for int32, max={max_amplitude})")
        else:
            log_callback(f"Output signal max amplitude: {np.max(np.abs(output_data_full))} (float32, max={max_amplitude})")
        
        output_data_float = output_data_full.astype(np.float32)
        if dtype == np.int16:
            output_data_float = output_data_float / 32767
        elif dtype == np.int32:
            output_data_float = output_data_float / 8388607
        input_queue = queue.Queue()
        recorded_data = []
        playback_started = threading.Event()

        def input_callback(indata, frames, time, status):
            if status:
                log_callback(f"Input stream status: {status}")
            if indata is not None and indata.size > 0:
                if dtype == np.int16:
                    indata = indata * 32767
                elif dtype == np.int32:
                    indata = indata * 8388607
                selected_data = indata[:, sorted_channels]
                input_queue.put(selected_data.copy())

        def output_callback(outdata, frames, time, status):
            if status:
                log_callback(f"Output stream status: {status}")
            idx = output_callback.current_index
            remaining = len(output_data_float) - idx
            if remaining <= 0:
                outdata.fill(0)
                return
            chunk_size = min(frames, remaining)
            outdata[:chunk_size] = output_data_float[idx:idx + chunk_size]
            if chunk_size < frames:
                outdata[chunk_size:] = 0
            output_callback.current_index += chunk_size
            if idx == 0:
                playback_started.set()
        output_callback.current_index = 0
        log_callback("Starting playback and recording...")
        input_stream = sd.InputStream(
            samplerate=fs,
            channels=max_input_channels,
            dtype='float32',
            callback=input_callback,
            blocksize=2048,
            latency='high'
        )
        input_stream.start()
        output_latency = device_info.get('default_low_output_latency', 0.01)
        time.sleep(output_latency + 0.01)
        output_stream = sd.OutputStream(
            samplerate=fs,
            channels=max_output_channels,
            dtype='float32',
            callback=output_callback,
            blocksize=2048,
            latency='high'
        )
        output_stream.start()
        if not playback_started.wait(timeout=2.0):
            log_callback("Warning: Playback timeout. Recording may be misaligned.")
        total_samples = len(output_data)
        samples_recorded = 0
        while samples_recorded < total_samples:
            try:
                data = input_queue.get(timeout=10.0)
                recorded_data.append(data)
                samples_recorded += len(data)
            except queue.Empty:
                log_callback("Warning: Input queue timeout. Recording incomplete.")
                break
        output_stream.stop()
        output_stream.close()
        input_stream.stop()
        input_stream.close()
        log_callback("Playback and recording completed.")

        if not recorded_data:
            log_callback("Error: No recorded data.")
            return None
        recorded_data = np.concatenate(recorded_data, axis=0)[:total_samples]
        if len(recorded_data) < total_samples:
            log_callback(f"Warning: Recorded data length ({len(recorded_data)}) shorter than expected ({total_samples}). Padding zeros.")
            recorded_data = np.pad(recorded_data, ((0, total_samples - len(recorded_data)), (0, 0)), mode='constant')

        if dtype == np.int16:
            recorded_data = np.clip(np.round(recorded_data), -32767, 32767).astype(np.int16)
        elif dtype == np.int32:
            recorded_data = np.clip(np.round(recorded_data), -8388607, 8388607).astype(np.int32)
        else:
            recorded_data = recorded_data.astype(np.float32)

        # Log amplitude for each input channel
        for ch_idx, ch in enumerate(input_channels):
            channel_amplitude = np.max(np.abs(recorded_data[:, ch_idx]))
            log_callback(f"Input Channel {ch + 1} recorded signal max amplitude: {channel_amplitude}")
        log_callback(f"Overall recorded signal max amplitude: {np.max(np.abs(recorded_data))}")

        # Log Peak and Avg Input Level only if log_level_details is True
        if log_level_details:
            for ch_idx, ch in enumerate(input_channels):
                ch_signal = recorded_data[:, ch_idx].astype(np.float32)
                if dtype == np.int16:
                    ch_signal = ch_signal / 32767.0
                elif dtype == np.int32:
                    ch_signal = ch_signal / 8388607.0

                peak_amplitude = np.max(np.abs(ch_signal))
                dbfs_level = 20 * np.log10(peak_amplitude + 1e-10)
                rms_amplitude = np.sqrt(np.mean(np.square(ch_signal)))
                avg_dbfs_level = 20 * np.log10(rms_amplitude + 1e-10)
                log_callback(f"Input Channel {ch + 1} Peak Input Level: {dbfs_level:.5f} dBFS")
                log_callback(f"Input Channel {ch + 1} Avg Input Level: {avg_dbfs_level:.5f} dBFS")

        return recorded_data

    except Exception as e:
        log_callback(f"Recording error: {e}")
        return None

def compute_sweep_response(input_signal, output_signal, fs, duration, input_channel, input_level, log_callback=None, f_start=0, f_end=None, dtype=np.float32):
    n_fft = int(fs * duration)
    fft_resolution = fs / n_fft
    if f_start <= 0:
        f_start = fft_resolution
    if f_end is None:
        f_end = fs / 2
    if output_signal.ndim == 1:
        recorded_signal = output_signal
    elif output_signal.ndim == 2 and output_signal.shape[1] > input_channel:
        recorded_signal = output_signal[:, input_channel]
    else:
        log_callback(f"Error: Invalid output signal dimension: {output_signal.shape}, expected channel {input_channel}")
        return np.array([]), np.array([])

    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
    recorded_amplitude = np.max(np.abs(recorded_signal))
    if recorded_amplitude < 0.0001 * max_amplitude:
        log_callback(f"Error: Channel {input_channel + 1} recorded signal amplitude ({recorded_amplitude}) too low. Expected near {max_amplitude}. Check Audio Interface Control Panel routing.")
        return np.array([]), np.array([])

    # Normalize signals to [0, 1] range based on bit depth
    input_signal_float = input_signal.astype(np.float32)
    recorded_signal_float = recorded_signal.astype(np.float32)
    if dtype == np.int16:
        input_signal_float /= 32767
        recorded_signal_float /= 32767
    elif dtype == np.int32:
        input_signal_float /= 8388607
        recorded_signal_float /= 8388607

    delay_samples = compute_time_delay(input_signal, recorded_signal, fs, log_callback)
    expected_length = int(fs * duration)
    if delay_samples > 0:
        recorded_signal_float = recorded_signal_float[delay_samples:]
        input_signal_float = input_signal_float[:-delay_samples if delay_samples > 0 else None]
    elif delay_samples < 0:
        input_signal_float = input_signal_float[-delay_samples:]
        recorded_signal_float = recorded_signal_float[:len(input_signal_float)]
    else:
        recorded_signal_float = recorded_signal_float

    min_length = min(len(input_signal_float), len(recorded_signal_float))
    if min_length < expected_length:
        pad_length = expected_length - min_length
        input_signal_float = np.pad(input_signal_float[:min_length], (0, pad_length), mode='constant')
        recorded_signal_float = np.pad(recorded_signal_float[:min_length], (0, pad_length), mode='constant')
    else:
        input_signal_float = input_signal_float[:expected_length]
        recorded_signal_float = recorded_signal_float[:expected_length]

    window = signal.windows.hann(expected_length)
    input_signal_windowed = input_signal_float * window
    output_signal_windowed = recorded_signal_float * window
    N = expected_length
    input_fft = np.fft.rfft(input_signal_windowed, n=N)
    output_fft = np.fft.rfft(output_signal_windowed, n=N)
    freqs = np.fft.rfftfreq(N, 1/fs)
    transfer = output_fft / (input_fft + 1e-10)

    # Compute magnitude in dBFS, adjusted for input level
    magnitude = 20 * np.log10(np.abs(transfer) + 1e-10) + input_level
    freq_mask = (freqs >= f_start) & (freqs <= f_end)
    freqs = freqs[freq_mask]
    magnitude = magnitude[freq_mask]
    log_callback(f"Calculated magnitude (dBFS) for channel {input_channel + 1}.")
    return freqs, magnitude

def compute_crosstalk_freq_response(output_signals, fs, target_freq, input_level, crosstalk_input_channels, log_callback=None, dtype=np.float32, input_signal=None):
    if output_signals is None or output_signals.size == 0:
        log_callback("Error: Invalid output signals.")
        return np.array([]), []
    if input_signal is None:
        log_callback("Error: No input signal provided. Cannot compute crosstalk.")
        return np.array([]), []
    
    recorded_signals = output_signals
    num_channels = recorded_signals.shape[1] if recorded_signals.ndim == 2 else 1
    log_callback(f"Recorded signals shape: {recorded_signals.shape}, Channels: {num_channels}")
    N = len(recorded_signals)
    freqs = np.fft.rfftfreq(N, 1/fs)
    freq_mask = (freqs >= 20) & (freqs <= 20000)
    freqs = freqs[freq_mask]
    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
    magnitudes = []
    window = signal.windows.hann(N)

    if target_freq == 997:
        try:
            b_bp, a_bp = design_crosstalk_bandpass_filter(fs, center_freq=997, bandwidth_ratio=1/3)
            input_signal_filtered = signal.lfilter(b_bp, a_bp, input_signal).astype(np.float32)
            recorded_signals_filtered = np.zeros_like(recorded_signals, dtype=np.float32)
            for ch in range(num_channels):
                recorded_signals_filtered[:, ch] = signal.lfilter(b_bp, a_bp, recorded_signals[:, ch]).astype(np.float32)
            log_callback(f"Applied AES17-2020 bandpass filter centered at 997 Hz, Q={1/(1/3):.2f}")
        except Exception as e:
            log_callback(f"Error applying bandpass filter: {str(e)}. Proceeding without filter.")
            input_signal_filtered = input_signal
            recorded_signals_filtered = recorded_signals
    else:
        input_signal_filtered = input_signal
        recorded_signals_filtered = recorded_signals

    input_signal_windowed = input_signal_filtered * window
    input_fft = np.fft.rfft(input_signal_windowed)
    window_gain = np.mean(window)
    input_amplitude = np.max(np.abs(input_fft)) * 2 / (N * window_gain)

    for ch in range(num_channels):
        signal_ch = recorded_signals_filtered[:, ch] * window if recorded_signals_filtered.ndim == 2 else recorded_signals_filtered * window
        fft_ch = np.fft.rfft(signal_ch, n=N)
        fft_magnitude = np.abs(fft_ch[freq_mask]) * 2 / (N * window_gain)
        fft_magnitude = np.maximum(fft_magnitude, 1e-10)

        magnitude = 20 * np.log10(fft_magnitude / max_amplitude + 1e-10)
        magnitudes.append(magnitude)
        target_idx = np.argmin(np.abs(freqs - target_freq))
        log_callback(f"Channel {ch + 1} Magnitude at {target_freq} Hz: {magnitude[target_idx]:.5f} dBFS")
    return freqs, magnitudes

def compute_thd_and_thdn(input_signal, output_signal, freq, fs, include_noise=True, dtype=np.float32, log_callback=None, Q=2.0, input_level=0.0):
    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)

    # Input validation
    if input_signal is None or output_signal is None:
        if log_callback:
            log_callback("Error: input or output signal is None")
        return np.nan, (-float('inf'), 0, np.nan), 0.0, None, None
    if len(input_signal) != len(output_signal) or len(input_signal) == 0:
        if log_callback:
            log_callback("Error: Invalid signal length")
        return np.nan, (-float('inf'), 0, np.nan), 0.0, None, None
    
    # Apply bandpass filter for total signal (AES17-2020, 5.2.5)
    try:
        b_lp, a_lp = design_low_pass_filter(fs, cutoff=22000, order=12)
        output_signal_filtered = signal.lfilter(b_lp, a_lp, output_signal).astype(np.float32)

        # Normalize signal based on dtype
        if dtype == np.int16:
            output_signal_filtered = output_signal_filtered / 32767.0
            input_signal_norm = np.float32(input_signal) / 32767.0
        elif dtype == np.int32:
            output_signal_filtered = output_signal_filtered / 8388607.0
            input_signal_norm = np.float32(input_signal) / 8388607.0
        else:
            output_signal_filtered = output_signal_filtered  # float32, already normalized
            input_signal_norm = np.float32(input_signal)
        total_signal_rms = np.sqrt(np.mean(np.square(output_signal_filtered)))
        log_callback(f"Applied bandpass filter (20 Hz - 22 kHz), Total Signal RMS: {total_signal_rms:.8f}")
    except Exception as e:
        if log_callback:
            log_callback(f"Error in bandpass filter: {str(e)}")
        return np.nan, (-float('inf'), 0, np.nan), 0.0, None, None

    # Apply Notch filter for residual (AES17-2020, 5.2.8)
    try:
        b_notch, a_notch = design_notch_filter(freq, Q, fs)
        residual_signal = signal.lfilter(b_notch, a_notch, output_signal_filtered)

        # Apply Hann window to residual signal before RMS calculation
        N = len(residual_signal)
        window = signal.windows.hann(N)
        residual_signal_windowed = residual_signal * window
        residual_rms = np.sqrt(np.mean(np.square(residual_signal_windowed)))
        log_callback(f"Applied notch filter at {freq} Hz with Q={Q:.5f} on filtered signal, Residual RMS: {residual_rms:.8f}")
    except Exception as e:
        if log_callback:
            log_callback(f"Error in notch filter: {str(e)}")
        return np.nan, (-float('inf'), 0, np.nan), 0.0, None, None

    # Compute FFTs with Hann window (AES17-2020, 5.2.10)
    N = len(input_signal_norm)
    window = signal.windows.hann(N)
    input_signal_windowed = input_signal_norm * window
    output_signal_windowed = output_signal_filtered * window
    input_fft = np.fft.rfft(input_signal_windowed)
    output_fft = np.fft.rfft(output_signal_windowed)
    freqs = np.fft.rfftfreq(N, 1/fs)
    window_gain = np.mean(window)
    
    # Find fundamental frequency index
    fundamental_idx = np.argmin(np.abs(freqs - freq))
    input_amplitude = np.abs(input_fft[fundamental_idx]) * 2 / (N * window_gain)
    output_amplitude = np.abs(output_fft[fundamental_idx]) * 2 / (N * window_gain)
    if input_amplitude < 0.00001:  # Check against normalized threshold
        if log_callback:
            log_callback(f"Error: Input amplitude ({input_amplitude}) too low.")
        return np.nan, (-float('inf'), 0, np.nan), 0.0, None, None
    
    # Compute THD (AES17-2020, 6.3.1)
    harmonic_amplitudes = []
    n = 2
    while True:
        harmonic_freq = n * freq
        if harmonic_freq > 20000:
            break
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
        if harmonic_idx < len(output_fft):
            harmonic_amplitude = np.abs(output_fft[harmonic_idx]) * 2 / (N * window_gain)
            harmonic_amplitudes.append(harmonic_amplitude)
        n += 1
    harmonic_rms = np.sqrt(np.sum(np.square(harmonic_amplitudes))) if harmonic_amplitudes else 0
    thd = (harmonic_rms / input_amplitude * 100) if input_amplitude > 0 else 0
    
    # Compute THD+N
    thdn_db = -float('inf')
    thdn_percent = 0.0
    thdn_dbfs = -float('inf')
    if total_signal_rms > 0:
        thdn_ratio = residual_rms / total_signal_rms
        thdn_percent = thdn_ratio * 100
        thdn_db = 20 * np.log10(thdn_ratio + 1e-10)
        
        # Compute THD+N in dBFS with normalized signal
        output_signal_norm = output_signal.astype(np.float32)
        if dtype == np.int16:
            output_signal_norm /= 32767.0
        elif dtype == np.int32:
            output_signal_norm /= 8388607.0
        peak_amplitude = np.max(np.abs(output_signal_norm))
        signal_dbfs = 20 * np.log10(peak_amplitude + 1e-10)
        thdn_dbfs = signal_dbfs + thdn_db
        log_callback(f"Debug: peak_amplitude={peak_amplitude:.6f}, signal_dbfs={signal_dbfs:.6f}, thdn_db={thdn_db:.6f}, thdn_dbfs={thdn_dbfs:.6f}")
    
    # Compute frequency response from FFT
    magnitude = np.abs(output_fft) * 2 / (N * window_gain)
    magnitude = 20 * np.log10(magnitude + 1e-10)
    freq_mask = (freqs >= 20) & (freqs <= 20000)
    freqs = freqs[freq_mask]
    magnitude = magnitude[freq_mask]
    
    if log_callback:
        log_callback(f"THD: {thd:.5f}%")
        log_callback(f"THD+N: {thdn_db:.4f} dB, {thdn_percent:.5f}%, {thdn_dbfs:.4f} dBFS")
        log_callback(f"Computed frequency response: {len(freqs)} points from 20 Hz to 20 kHz")
    
    return thd, (thdn_db, thdn_percent, thdn_dbfs), output_amplitude, freqs, magnitude

class AudioMeasurementApp:
    def __init__(self, root):
        self.current_test_level = -1.0
        self.root = root
        self.root.title("Audio Measurement Tool")
        self.base_width = 1000
        self.base_height = 950
        self.root.geometry(f"{self.base_width}x{self.base_height}")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)
        self.results_thd = {
            'thd': {f: [] for f in THD_frequencies},
            'thdn': {f: [] for f in THD_frequencies},
            'thd_ratio_minus20db': {f: [] for f in [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]},
            'thdn_ratio_minus20db': {f: [] for f in [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]},
            'freq_response': [],
            'input_thd': [[]],  # Added for input level THD
            'input_thdn': [[]]  # Added for input level THD+N
        }
        self.results_crosstalk = {
            'crosstalk_response': []
        }
        self.results_frequency_response = {
            'frequency_response': [[]],  # For Measure Freq Response
            'input_thdn': [[] for _ in range(9)],  # For Measure Input THD+N, 9 levels (0 to -80 dBFS)
            'thdn_minus20db': [[]]  # For Measure THD+N (-20 dBFS)
        }
        self.devices = sd.query_devices()
        self.device_names = [d['name'] for d in self.devices]
        default_input, default_output = sd.default.device
        default_input_name = self.device_names[default_input] if default_input < len(self.device_names) else self.device_names[0]
        default_output_name = self.device_names[default_output] if default_output < len(self.device_names) else self.device_names[0]
        self.input_device = tk.StringVar(value=default_input_name)
        self.output_device = tk.StringVar(value=default_output_name)
        self.input_channel = tk.StringVar(value="1")
        self.output_channel = tk.StringVar(value="1")
        self.crosstalk_channel_vars = []
        self.crosstalk_channel_pairs = []
        self.sample_rates = ["44100", "48000", "96000", "192000"]
        self.bit_depths = ["16", "24", "32float"]
        self.sample_rate = tk.StringVar(value="192000")
        self.bit_depth = tk.StringVar(value="24")
        self.crosstalk_freq_var = tk.StringVar(value=str(THD_frequencies[0]))
        self.frequency_response_freq_var = tk.StringVar(value=str(frequency_response_frequencies[0]))
        self.include_thdn = tk.BooleanVar(value=False)
        self.input_level_db = tk.StringVar(value="-6")
        self.duration_var = tk.StringVar(value="5.0")
        self.num_trials_var = tk.StringVar(value="3")
        self.fig, self.ax1 = plt.subplots(figsize=(8, 4))
        self.thd_fig, self.thd_ax = plt.subplots(figsize=(8, 4))
        self.create_widgets()
        self.result_text.bind("<Control-c>", self.handle_copy)
        self.result_text.bind("<Command-c>", self.handle_copy)


    def create_widgets(self):
        # Main container frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Settings frame (no ScrolledFrame, just ttk.Frame)
        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Container for side-by-side Device and Measurement settings
        settings_container = ttk.Frame(settings_frame)
        settings_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Device Settings frame (left)
        device_frame = ttk.LabelFrame(settings_container, text="Device Settings", padding=10)
        device_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.OptionMenu(device_frame, self.input_device, self.input_device.get(), *self.device_names,
                       command=self.update_channel_options).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(device_frame, text="Output Device:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.OptionMenu(device_frame, self.output_device, self.output_device.get(), *self.device_names,
                       command=self.update_channel_options).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(device_frame, text="Main Input Channel:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.input_channel_menu = ttk.OptionMenu(device_frame, self.input_channel, self.input_channel.get(), "1")
        self.input_channel_menu.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(device_frame, text="Main Output Channel:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.output_channel_menu = ttk.OptionMenu(device_frame, self.output_channel, self.output_channel.get(), "1")
        self.output_channel_menu.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(device_frame, text="Crosstalk Input Channels:").grid(row=4, column=0, padx=5, pady=5, sticky="ne")
        self.crosstalk_pair_container = ttk.Frame(device_frame)
        self.crosstalk_pair_container.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Measurement settings frame (right)
        measurement_frame = ttk.LabelFrame(settings_container, text="Measurement Settings", padding=10)
        measurement_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ttk.Label(measurement_frame, text="Notch Filter Q:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.q_var = tk.DoubleVar(value=2.0)
        ttk.Entry(measurement_frame, textvariable=self.q_var, width=5).grid(row=6, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(measurement_frame, text="Bit Depth:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.OptionMenu(measurement_frame, self.bit_depth, self.bit_depth.get(), *self.bit_depths).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(measurement_frame, text="THD Frequency (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        ttk.OptionMenu(measurement_frame, self.crosstalk_freq_var, str(THD_frequencies[0]), *[str(f) for f in THD_frequencies]).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(measurement_frame, text="Freq Response Start (Hz):").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        ttk.OptionMenu(measurement_frame, self.frequency_response_freq_var, str(frequency_response_frequencies[0]), *[str(f) for f in frequency_response_frequencies]).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(measurement_frame, text="Input Level (dBFS):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        input_level_options = ["-10", "-6", "0", "+4"]
        ttk.OptionMenu(measurement_frame, self.input_level_db, self.input_level_db.get(), *input_level_options).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(measurement_frame, text="Sample Rate: Follows audio interface settings.").grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Configure grid weights for responsive resizing
        settings_container.columnconfigure(0, weight=1)
        settings_container.columnconfigure(1, weight=1)
        settings_container.rowconfigure(0, weight=1)

        # First button row: Measure buttons
        measure_button_frame = ttk.Frame(main_frame)
        measure_button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(measure_button_frame, text="Measure Input THD+N", command=self.measure_input_thdn, style="primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(measure_button_frame, text="Measure THD+N (-20 dBFS)", command=self.measure_thd_ratio_minus20db, style="primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(measure_button_frame, text="Measure Crosstalk", command=self.measure_crosstalk, style="primary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(measure_button_frame, text="Measure Freq Response", command=self.measure_frequency_response, style="primary.TButton").pack(side=tk.LEFT, padx=5)

        # Second button row: Save and Copy buttons
        save_button_frame = ttk.Frame(main_frame)
        save_button_frame.pack(fill=tk.X, pady=2)
        ttk.Button(save_button_frame, text="Save THD CSV", command=self.save_results, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Save Crosstalk CSV", command=self.save_crosstalk_csv, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Save Freq Response", command=self.save_sweep_txt, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Save Graph", command=self.save_graph, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Save THD Graph", command=self.save_thd_graph, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Save Preset", command=self.save_preset, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Load Preset", command=self.save_load_preset, style="secondary.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(save_button_frame, text="Copy Log", command=self.copy_log, style="secondary.TButton").pack(side=tk.LEFT, padx=5)

        # Third button row: Donation buttons
        donation_frame = ttk.LabelFrame(main_frame, text="Support the Project", padding=10)
        donation_frame.pack(fill=tk.X, pady=2)
        ttk.Label(donation_frame, text="Like this tool? Consider supporting Jooyoung Kim with a donation!:").pack(side=tk.LEFT, padx=5)
        amounts = [1.99, 4.99, 9.99, 19.99, 49.99]
        styles = ["success.TButton", "info.TButton", "primary.TButton", "warning.TButton", "danger.TButton"]
        for amount, style in zip(amounts, styles):
            ttk.Button(
                donation_frame,
                text=f"${amount}",
                command=lambda a=amount: webbrowser.open(f"https://paypal.me/JooyoungMusic/{a}"),
                style=style,
                width=8
            ).pack(side=tk.LEFT, padx=5)

        # Results frame (contains notebook)
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results notebook
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_frame = ttk.Frame(self.results_notebook, padding=5)
        self.results_notebook.add(log_frame, text="Log")
        self.result_text = tk.Text(log_frame, height=10, width=80, font=("Arial", 14), fg="white", bg="#2b2b2b")
        self.result_text.config(state='normal')
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.bind("<Command-c>", lambda event: self.copy_selected_text())
        self.result_text.bind("<Button-1>", lambda event: self.result_text.focus_set())

        # Graph tab
        graph_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(graph_frame, text="Graph")
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add THD Graph tab
        thd_graph_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(thd_graph_frame, text="THD Graph")
        self.thd_canvas = FigureCanvasTkAgg(self.thd_fig, master=thd_graph_frame)
        self.thd_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Context menu for log
        self.context_menu = tk.Menu(self.result_text, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_selected_text)
        self.result_text.bind("<Button-3>", self.show_context_menu)

        # Update channel options
        self.update_channel_options()

    def add_donation_buttons(self, parent):
        ttk.Label(parent, text="Like this tool? Consider supporting Jooyoung Kim with a donation!:").pack(side=tk.LEFT, padx=5)
        amounts = [1.99, 4.99, 9.99, 19.99, 49.99]
        styles = ["success.TButton", "info.TButton", "primary.TButton", "warning.TButton", "danger.TButton"]
        for amount, style in zip(amounts, styles):
            ttk.Button(
                parent,
                text=f"${amount}",
                command=lambda a=amount: webbrowser.open(f"https://paypal.me/JooyoungMusic/{a}"),
                style=style,
                width=8
            ).pack(side=tk.LEFT, padx=5)

    def handle_copy(self, event):
        try:
            selected_text = self.result_text.get("sel.first", "sel.last")
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.log_to_text("Copied selected text to clipboard")
        except tk.TclError:
            self.log_to_text("Error: No text selected for copying")
        return "break"

    def block_key_input(self, event):
        # Allow Cmd+C (Mac) or Ctrl+C (Windows/Linux)
        if event.keysym == "c" and (event.state & 0x4):  # Command or Control key
            return None
        return "break"

    def show_context_menu(self, event):
        self.result_text.focus_set()
        self.context_menu.post(event.x_root, event.y_root)

    def copy_selected_text(self):
        try:
            selected_text = self.result_text.get("sel.first", "sel.last")
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.log_to_text("Selected text copied to clipboard")
        except tk.TclError:
            self.log_to_text("Error: No text selected for copying")

    def update_crosstalk_channels(self):
        self.crosstalk_channel_pairs = [(0, i) for i, var in enumerate(self.crosstalk_channel_vars) if var.get()]
        self.log_to_text(f"Updated crosstalk channels: {[ch + 1 for _, ch in self.crosstalk_channel_pairs]}")

    def update_channel_options(self, *args):
        for widget in self.crosstalk_pair_container.winfo_children():
            widget.destroy()
        self.crosstalk_channel_vars = []
        self.crosstalk_channel_pairs = []
        self.crosstalk_checkboxes = []
        input_device_idx = self.device_names.index(self.input_device.get()) if self.device_names else 0
        output_device_idx = self.device_names.index(self.output_device.get()) if self.device_names else 0
        input_channels = self.devices[input_device_idx]['max_input_channels'] if self.device_names else 1
        output_channels = self.devices[output_device_idx]['max_output_channels'] if self.device_names else 1
        input_channel_options = [str(i + 1) for i in range(input_channels)]
        output_channel_options = [str(i + 1) for i in range(output_channels)]
        self.input_channel_menu['menu'].delete(0, 'end')
        for ch in input_channel_options:
            self.input_channel_menu['menu'].add_command(label=ch, command=lambda v=ch: self.input_channel.set(v))
        self.input_channel.set(input_channel_options[0] if input_channel_options else "1")
        self.output_channel_menu['menu'].delete(0, 'end')
        for ch in output_channel_options:
            self.output_channel_menu['menu'].add_command(label=ch, command=lambda v=ch: self.output_channel.set(v))
        self.output_channel.set(output_channel_options[0] if output_channel_options else "1")
        columns_per_row = 8
        for i in range(input_channels):
            var = tk.BooleanVar(value=False)
            self.crosstalk_channel_vars.append(var)
            row = i // columns_per_row
            col = i % columns_per_row
            chk = ttk.Checkbutton(
                self.crosstalk_pair_container,
                text=str(i + 1),
                variable=var,
                command=self.update_crosstalk_channels
            )
            chk.grid(row=row, column=col, sticky="w", padx=5, pady=2)
            self.crosstalk_checkboxes.append(chk)

    def log_to_text(self, message):
            self.result_text.insert(tk.END, message + "\n")
            self.result_text.see(tk.END)
            self.root.update()

    def measure_input_thdn(self):
        self.results_frequency_response = {
                'frequency_response': [[]],  # For Measure Freq Response
                'input_thdn': [[] for _ in range(9)],  # For Measure Input THD+N
                'thdn_minus20db': [[]]  # For Measure THD+N (-20 dBFS)
            }

        self.results_thd = {
            'thd': {f: [] for f in THD_frequencies},
            'thdn': {f: [] for f in THD_frequencies},
            'thd_ratio_minus20db': {f: [] for f in [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]},
            'thdn_ratio_minus20db': {f: [] for f in [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]},
            'freq_response': [],
            'input_thd': [[] for _ in range(9)],
            'input_thdn': [[] for _ in range(9)]
        }

        self.thd_results = []
        for _ in range(9):
            self.thd_results.append({
                'frequencies': THD_frequencies,
                'thd': [],
                'thd_std': [],
                'thdn_db': [],
                'thdn_db_std': [],
                'thdn_percent': [],
                'thdn_percent_std': [],
                'thdn_dbfs': [],
                'thdn_dbfs_std': [],
                'dbfs_input': []
            })

        self.log_to_text("Starting Input THD+N Measurement")
        output_device_idx = self.device_names.index(self.output_device.get())
        device_info = sd.query_devices()[output_device_idx]
        fs = int(device_info['default_samplerate']) if isinstance(device_info['default_samplerate'], (int, float)) else int(device_info['default_samplerate'][0])
        self.log_to_text(f"Using sample rate from audio interface: {fs} Hz")
        duration = float(self.duration_var.get())
        num_trials = int(self.num_trials_var.get())
        input_device_idx = self.device_names.index(self.input_device.get())
        output_device_idx = self.device_names.index(self.output_device.get())
        input_channel = int(self.input_channel.get()) - 1
        output_channel = int(self.output_channel.get()) - 1
        dtype = np.int16 if self.bit_depth.get() == "16" else np.int32 if self.bit_depth.get() == "24" else np.float32
        all_input_channels = [input_channel]
        if not all_input_channels:
            messagebox.showerror("Error", "Select at least one input channel.")
            return
        device_info = sd.query_devices()[output_device_idx]
        supported_samplerates = [device_info['default_samplerate']] if isinstance(device_info['default_samplerate'], (int, float)) else device_info['default_samplerate']
        if fs not in supported_samplerates:
            self.log_to_text(f"Warning: Sample rate {fs} Hz not supported. Using {supported_samplerates[0]} Hz.")
            fs = supported_samplerates[0]
        self.result_text.delete(1.0, tk.END)

        input_levels = [0, -10, -20, -30, -40, -50, -60, -70, -80]
        try:
            for level_idx, input_level in enumerate(input_levels):
                self.log_to_text(f"\nInput Level: {input_level} dBFS")
                thd_values = []
                thdn_db_values = []
                thdn_percent_values = []
                thdn_dbfs_values = []
                dbfs_inputs = []

                for trial in range(num_trials):
                    self.log_to_text(f"Trial {trial + 1}/{num_trials}")
                    input_signal = generate_sine_signal(997, fs, duration, input_level, dtype)
                    recorded_data = record_output(
                        input_signal, fs, input_device_idx, output_device_idx,
                        all_input_channels, output_channel, dtype, self.log_to_text
                    )
                    if recorded_data is None:
                        self.log_to_text("Error: No recorded data")
                        continue

                    peak_amplitude = np.max(np.abs(recorded_data[:, 0]))
                    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
                    normalized_amplitude = peak_amplitude / max_amplitude
                    dbfs_input = 20 * np.log10(normalized_amplitude + 1e-10)
                    self.log_to_text(f"Current Channel dBFS Input: {dbfs_input:.6f} dBFS")
                    thd, thdn, output_amplitude, freqs, magnitude = compute_thd_and_thdn(
                        input_signal, recorded_data[:, 0], 997, fs,
                        log_callback=self.log_to_text, Q=float(self.q_var.get()), input_level=input_level, dtype=dtype
                    )
                    dbfs_inputs.append(dbfs_input)
                    thd_values.append(thd)
                    thdn_db_values.append(thdn[0])
                    thdn_percent_values.append(thdn[1])
                    thdn_dbfs_values.append(thdn[2])
                    self.log_to_text(f"Current Channel Total dBFS Input: {dbfs_input:.6f} dBFS")
                    self.log_to_text(f"Max output amplitude: {output_amplitude:.6f}")

                    if input_level == -10 and freqs is not None and magnitude is not None:
                        if len(freqs) == len(magnitude) and len(freqs) > 0:
                            self.results_frequency_response['input_thdn'][level_idx].append((freqs, magnitude))
                        else:
                            self.log_to_text(f"Error: Invalid frequency response data for trial {trial + 1}")

                if thd_values:
                    thd_mean = np.mean(thd_values)
                    thd_std = np.std(thd_values) if len(thd_values) > 1 else 0
                    thdn_db_mean = np.mean(thdn_db_values)
                    thdn_db_std = np.std(thdn_db_values) if len(thdn_db_values) > 1 else 0
                    thdn_percent_mean = np.mean(thdn_percent_values)
                    thdn_percent_std = np.std(thdn_percent_values) if len(thdn_percent_values) > 1 else 0
                    thdn_dbfs_mean = np.mean(thdn_dbfs_values)
                    thdn_dbfs_std = np.std(thdn_dbfs_values) if len(thdn_dbfs_values) > 1 else 0
                    dbfs_input_mean = np.mean(dbfs_inputs)
                    self.results_thd['input_thd'][level_idx].append(thd_mean)
                    self.results_thd['input_thdn'][level_idx].append((thdn_db_mean, thdn_percent_mean, thdn_dbfs_mean))
                    self.thd_results[level_idx]['thd'].append(thd_mean)
                    self.thd_results[level_idx]['thd_std'].append(thd_std)
                    self.thd_results[level_idx]['thdn_db'].append(thdn_db_mean)
                    self.thd_results[level_idx]['thdn_db_std'].append(thdn_db_std)
                    self.thd_results[level_idx]['thdn_percent'].append(thdn_percent_mean)
                    self.thd_results[level_idx]['thdn_percent_std'].append(thdn_percent_std)
                    self.thd_results[level_idx]['thdn_dbfs'].append(thdn_dbfs_mean)
                    self.thd_results[level_idx]['thdn_dbfs_std'].append(thdn_dbfs_std)
                    self.thd_results[level_idx]['dbfs_input'].append(dbfs_input_mean)
                else:
                    self.log_to_text(f"No valid data for 997 Hz at {input_level} dBFS")

            self.log_to_text("\n=== Total THD+N Summary ===")
            self.log_to_text("dBFS Level | Freq (Hz) | THD (%) | THD+N (dB) | THD+N (%) | THD+N (dBFS) | dBFS Input")
            self.log_to_text("-" * 100)

            for level_idx, input_level in enumerate(input_levels):
                if not self.thd_results[level_idx]['thd']:
                    self.log_to_text(f"No valid THD+N data for {input_level} dBFS.")
                    continue

                thd = self.thd_results[level_idx]['thd'][0]
                thd_std = self.thd_results[level_idx]['thd_std'][0]
                thdn_db = self.thd_results[level_idx]['thdn_db'][0]
                thdn_db_std = self.thd_results[level_idx]['thdn_db_std'][0]
                thdn_percent = self.thd_results[level_idx]['thdn_percent'][0]
                thdn_percent_std = self.thd_results[level_idx]['thdn_percent_std'][0]
                thdn_dbfs = self.thd_results[level_idx]['thdn_dbfs'][0]
                thdn_dbfs_std = self.thd_results[level_idx]['thdn_dbfs_std'][0]
                dbfs_input = self.thd_results[level_idx]['dbfs_input'][0]

                self.log_to_text(
                    f"{input_level:10d} | {997:9d} | {thd:7.5f} ± {thd_std:.5f} | "
                    f"{thdn_db:10.5f} ± {thdn_db_std:.5f} | "
                    f"{thdn_percent:9.5f} ± {thdn_percent_std:.5f} | "
                    f"{thdn_dbfs:12.5f} ± {thdn_dbfs_std:.5f} | "
                    f"{dbfs_input:20.5f}"
                )

            self.plot_thd_results()
            if self.results_frequency_response['input_thdn'][1]:  # -10 dBFS
                self.plot_sweep_results(all_input_channels, key='input_thdn')
                self.results_notebook.select(1)
        except Exception as e:
            self.log_to_text(f"Error in measure_input_thdn: {str(e)}")
            messagebox.showerror("Error", str(e))

    # In AudioMeasurementApp class, update measure_thd_ratio_minus20db
    def measure_thd_ratio_minus20db(self):
        self.current_test_level = -20.0
        self.results_thd = {
            'thd': {f: [] for f in THD_frequencies},
            'thdn': {f: [] for f in THD_frequencies},
            'thd_ratio_minus20db': {f: [] for f in [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]},
            'thdn_ratio_minus20db': {f: [] for f in [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]},
            'freq_response': [],
            'input_thd': [[] for _ in range(9)],
            'input_thdn': [[] for _ in range(9)]
        }
        self.results_frequency_response = {
            'frequency_response': [[]],
            'input_thdn': [[] for _ in range(9)],
            'thdn_minus20db': [[]]
        }
        self.thd_results = {
            'frequencies': [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000],
            'thd': [],
            'thd_std': [],
            'thdn_db': [],
            'thdn_db_std': [],
            'thdn_percent': [],
            'thdn_percent_std': [],
            'thdn_dbfs': [],
            'thdn_dbfs_std': [],
            'dbfs_input': []
        }
        self.log_to_text("Starting THD+N Measurement at -20 dBFS")
        output_device_idx = self.device_names.index(self.output_device.get())
        device_info = sd.query_devices()[output_device_idx]
        fs = int(device_info['default_samplerate']) if isinstance(device_info['default_samplerate'], (int, float)) else int(device_info['default_samplerate'][0])
        self.log_to_text(f"Using sample rate from audio interface: {fs} Hz")
        duration = float(self.duration_var.get())
        num_trials = int(self.num_trials_var.get())
        input_device_idx = self.device_names.index(self.input_device.get())
        output_device_idx = self.device_names.index(self.output_device.get())
        input_channel = int(self.input_channel.get()) - 1
        output_channel = int(self.output_channel.get()) - 1
        dtype = np.int16 if self.bit_depth.get() == "16" else np.int32 if self.bit_depth.get() == "24" else np.float32
        all_input_channels = [input_channel]
        if not all_input_channels:
            messagebox.showerror("Error", "Select at least one input channel.")
            return
        device_info = sd.query_devices()[output_device_idx]
        supported_samplerates = [device_info['default_samplerate']] if isinstance(device_info['default_samplerate'], (int, float)) else device_info['default_samplerate']
        if fs not in supported_samplerates:
            self.log_to_text(f"Warning: Sample rate {fs} Hz not supported. Using {supported_samplerates[0]} Hz.")
            fs = supported_samplerates[0]
        self.result_text.delete(1.0, tk.END)
        input_level = -20.0
        test_frequencies = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]
        try:
            for freq in test_frequencies:
                self.log_to_text(f"\nFrequency: {freq} Hz")
                thd_values = []
                thdn_db_values = []
                thdn_percent_values = []
                thdn_dbfs_values = []
                dbfs_inputs = []
                for trial in range(num_trials):
                    self.log_to_text(f"Trial {trial + 1}/{num_trials}")
                    input_signal = generate_sine_signal(freq, fs, duration, input_level, dtype)
                    recorded_data = record_output(
                        input_signal, fs, input_device_idx, output_device_idx,
                        all_input_channels, output_channel, dtype, self.log_to_text
                    )
                    if recorded_data is None:
                        self.log_to_text("Error: No recorded data")
                        continue
                    peak_amplitude = np.max(np.abs(recorded_data[:, 0]))
                    max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
                    normalized_amplitude = peak_amplitude / max_amplitude
                    dbfs_input = 20 * np.log10(normalized_amplitude + 1e-10)
                    self.log_to_text(f"Current Channel dBFS Input: {dbfs_input:.6f} dBFS")
                    thd, thdn, output_amplitude, freqs, magnitude = compute_thd_and_thdn(
                        input_signal, recorded_data[:, 0], freq, fs,
                        log_callback=self.log_to_text, Q=float(self.q_var.get()), input_level=input_level, dtype=dtype
                    )

                    if np.isnan(thd):
                        self.log_to_text(f"Error: Invalid THD for trial {trial + 1}")
                        continue
                    thd_values.append(thd)
                    thdn_db_values.append(thdn[0])
                    thdn_percent_values.append(thdn[1])
                    thdn_dbfs_values.append(thdn[2])
                    peak_amplitude = np.max(np.abs(recorded_data[:, 0]))
                    normalized_amplitude = peak_amplitude / max_amplitude
                    dbfs_input = 20 * np.log10(normalized_amplitude + 1e-10)
                    dbfs_inputs.append(dbfs_input)
                    self.results_thd['thd_ratio_minus20db'][freq].append(thd)
                    self.results_thd['thdn_ratio_minus20db'][freq].append(thdn)

                    # Save frequency response only for 1250 Hz
                    if freq == 1250 and freqs is not None and magnitude is not None:
                        if len(freqs) == len(magnitude) and len(freqs) > 0:
                            self.results_frequency_response['thdn_minus20db'][0].append((freqs, magnitude))
                            self.log_to_text(f"Stored frequency response for 1250 Hz, trial {trial + 1}")
                        else:
                            self.log_to_text(f"Error: Invalid frequency response data for trial {trial + 1}")

                    thdn_db, thdn_percent, thdn_dbfs = thdn
                    dbfs_inputs.append(dbfs_input)
                    thd_values.append(thd)
                    thdn_db_values.append(thdn_db)
                    thdn_percent_values.append(thdn_percent)
                    thdn_dbfs_values.append(thdn_dbfs)
                    self.results_thd['thd_ratio_minus20db'][freq].append(thd_values)
                    self.results_thd['thdn_ratio_minus20db'][freq].append([thdn_db, thdn_percent, thdn_dbfs])

                if thd_values:
                    self.thd_results['thd'].append(np.mean(thd_values))
                    self.thd_results['thd_std'].append(np.std(thd_values) if len(thd_values) > 1 else 0)
                    self.thd_results['thdn_db'].append(np.mean(thdn_db_values))
                    self.thd_results['thdn_db_std'].append(np.std(thdn_db_values) if len(thdn_db_values) > 1 else 0)
                    self.thd_results['thdn_percent'].append(np.mean(thdn_percent_values))
                    self.thd_results['thdn_percent_std'].append(np.std(thdn_percent_values) if len(thdn_percent_values) > 1 else 0)
                    self.thd_results['thdn_dbfs'].append(np.mean(thdn_dbfs_values))
                    self.thd_results['thdn_dbfs_std'].append(np.std(thdn_dbfs_values) if len(thdn_dbfs_values) > 1 else 0)
                    self.thd_results['dbfs_input'].append(np.mean(dbfs_inputs))
                    self.display_thd_results(freq, all_input_channels, recorded_data, dtype, key='thd_ratio_minus20db')
                else:
                    self.thd_results['thd'].append(0)
                    self.thd_results['thd_std'].append(0)
                    self.thd_results['thdn_db'].append(0)
                    self.thd_results['thdn_db_std'].append(0)
                    self.thd_results['thdn_percent'].append(0)
                    self.thd_results['thdn_percent_std'].append(0)
                    self.thd_results['thdn_dbfs'].append(0)
                    self.thd_results['thdn_dbfs_std'].append(0)
                    self.thd_results['dbfs_input'].append(input_level)
                    self.log_to_text(f"No valid data for frequency {freq} Hz")
            self.log_to_text(f"\n=== Total THD+N Summary at -20 dBFS ===")
            self.log_to_text(f"{'Freq (Hz)':<10} | {'THD (%)':<15} | {'THD+N (dB)':<15} | {'THD+N (%)':<15} | {'THD+N (dBFS)':<15} | {'dBFS Input':<15}")
            self.log_to_text("-" * 100)
            for freq, thd, thd_std, thdn_db, thdn_db_std, thdn_percent, thdn_percent_std, thdn_dbfs, thdn_dbfs_std, dbfs_input in zip(
                self.thd_results['frequencies'],
                self.thd_results['thd'],
                self.thd_results['thd_std'],
                self.thd_results['thdn_db'],
                self.thd_results['thdn_db_std'],
                self.thd_results['thdn_percent'],
                self.thd_results['thdn_percent_std'],
                self.thd_results['thdn_dbfs'],
                self.thd_results['thdn_dbfs_std'],
                self.thd_results['dbfs_input']
            ):
                self.log_to_text(
                    f"{freq:<10} | {thd:.5f} ± {thd_std:.5f} | {thdn_db:.5f} ± {thdn_db_std:.5f} | {thdn_percent:.5f} ± {thdn_percent_std:.5f} | {thdn_dbfs:.5f} ± {thdn_dbfs_std:.5f} | {dbfs_input:.5f}"
                )
            self.plot_thd_results()
            self.plot_sweep_results(all_input_channels, key='thdn_minus20db')
        except Exception as e:
            self.log_to_text(f"Error: {e}")
            messagebox.showerror("Error", str(e))

    def log_thd_summary(self, input_level, level_idx=None, is_input_level=True):
        self.log_to_text(f"\n=== THD+N Summary at {input_level} dBFS ===")
        try:
            if is_input_level and isinstance(self.thd_results, list):

                # Handle Measure Input THD+N
                if level_idx is None or level_idx >= len(self.thd_results):
                    self.log_to_text(f"Error: Invalid level index {level_idx}")
                    return
                results = self.thd_results[level_idx]
                if not results['thd'] or len(results['frequencies']) != len(results['thd']):
                    self.log_to_text("No valid THD data to summarize")
                    return
                for freq, thd, thd_std, thdn_db, thdn_db_std, thdn_percent, thdn_percent_std, thdn_dbfs, thdn_dbfs_std, dbfs_input in zip(
                    results['frequencies'],
                    results['thd'],
                    results['thd_std'],
                    results['thdn_db'],
                    results['thdn_db_std'],
                    results['thdn_percent'],
                    results['thdn_percent_std'],
                    results['thdn_dbfs'],
                    results['thdn_dbfs_std'],
                    results['dbfs_input']
                ):
                    self.log_to_text(f"Frequency {freq} Hz:")
                    self.log_to_text(f"  THD: {thd:.5f}% ± {thd_std:.5f}")
                    self.log_to_text(f"  THD+N: {thdn_db:.5f} dB ± {thdn_db_std:.5f}, {thdn_percent:.5f}% ± {thdn_percent_std:.5f}, {thdn_dbfs:.5f} dBFS ± {thdn_dbfs_std:.5f}")
                    self.log_to_text(f"  dBFS Input: {dbfs_input:.5f}")
            elif not is_input_level and isinstance(self.thd_results, dict):

                # Handle Measure THD+N (-20 dBFS)
                if not self.thd_results['thd'] or len(self.thd_results['frequencies']) != len(self.thd_results['thd']):
                    self.log_to_text("No valid THD data to summarize")
                    return
                dbfs_key = 2  # Only handle -20 dBFS
                for freq, thd, thd_std, thdn_db, thdn_db_std, thdn_percent, thdn_percent_std, thdn_dbfs, thdn_dbfs_std, dbfs_input in zip(
                    self.thd_results['frequencies'],
                    self.thd_results['thd'],
                    self.thd_results['thd_std'],
                    self.thd_results['thdn_db'],
                    self.thd_results['thdn_db_std'],
                    self.thd_results['thdn_percent'],
                    self.thd_results['thdn_percent_std'],
                    self.thd_results['thdn_dbfs'],
                    self.thd_results['thdn_dbfs_std'],
                    self.thd_results['dbfs_input'][dbfs_key]
                ):
                    self.log_to_text(f"Frequency {freq} Hz:")
                    self.log_to_text(f"  THD: {thd:.5f}% ± {thd_std:.5f}")
                    self.log_to_text(f"  THD+N: {thdn_db:.5f} dB ± {thdn_db_std:.5f}, {thdn_percent:.5f}% ± {thdn_percent_std:.5f}, {thdn_dbfs:.5f} dBFS ± {thdn_dbfs_std:.5f}")
                    self.log_to_text(f"  dBFS Input: {dbfs_input:.5f}")
            else:
                self.log_to_text("Error: Invalid thd_results structure")
        except Exception as e:
            self.log_to_text(f"Error in THD summary: {e}")

    def measure_crosstalk(self):
        output_device_idx = self.device_names.index(self.output_device.get())
        device_info = sd.query_devices()[output_device_idx]
        fs = int(device_info['default_samplerate']) if isinstance(device_info['default_samplerate'], (int, float)) else int(device_info['default_samplerate'][0])
        self.log_to_text(f"Using sample rate from audio interface: {fs} Hz")
        bit_depth = self.bit_depth.get()
        input_device_idx = self.device_names.index(self.input_device.get())
        output_device_idx = self.device_names.index(self.output_device.get())
        main_input_channel = int(self.input_channel.get()) - 1
        main_output_channel = int(self.output_channel.get()) - 1
        max_output_channels = self.devices[output_device_idx]['max_output_channels']
        max_input_channels = self.devices[input_device_idx]['max_input_channels']
        input_level = -20.0
        if bit_depth == "16":
            dtype = np.int16
        elif bit_depth == "24":
            dtype = np.int32
        else:
            dtype = np.float32
        crosstalk_pairs = self.crosstalk_channel_pairs
        all_output_channels = [main_output_channel]
        valid_crosstalk_channels = [in_ch for _, in_ch in crosstalk_pairs if in_ch is not None and 0 <= in_ch < max_input_channels]
        all_input_channels = []
        if main_input_channel not in all_input_channels:
            all_input_channels.append(main_input_channel)
        for ch in valid_crosstalk_channels:
            if ch not in all_input_channels:
                all_input_channels.append(ch)
        all_input_channels = sorted(all_input_channels)
        self.log_to_text(f"Crosstalk pairs: {[(out_ch + 1, in_ch + 1) for out_ch, in_ch in crosstalk_pairs]}")
        self.log_to_text(f"All input channels: {[ch + 1 for ch in all_input_channels]}")
        self.log_to_text(f"Valid crosstalk channels: {[ch + 1 for ch in valid_crosstalk_channels]}")
        if not all_input_channels or not all_output_channels:
            messagebox.showerror("Error", "Select at least one valid input and output channel.")
            return

        # Validate output channel
        if main_output_channel >= max_output_channels:
            self.log_to_text(f"Error: Output channel {main_output_channel + 1} exceeds max ({max_output_channels}). Using channel 0.")
            main_output_channel = 0
            all_output_channels = [0]
        device_info = sd.query_devices()[output_device_idx]
        supported_samplerates = [device_info['default_samplerate']] if isinstance(device_info['default_samplerate'], (int, float)) else device_info['default_samplerate']
        if fs not in supported_samplerates:
            self.log_to_text(f"Warning: Sample rate {fs} Hz not supported. Using closest supported rate: {supported_samplerates[0]} Hz.")
            fs = supported_samplerates[0]
        self.result_text.delete(1.0, tk.END)
        self.results_crosstalk['crosstalk_response'] = []
        try:
            test_frequencies = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]
            num_trials = 3
            for freq in test_frequencies:
                self.log_to_text(f"\n=== Inter-channel Crosstalk Measurement at {freq} Hz ===")
                for output_channel in set(all_output_channels):
                    freq_results = []
                    input_channels = all_input_channels
                    input_channel_map = {i: ch for i, ch in enumerate(input_channels)}
                    self.log_to_text(f"Input channels for measurement: {[ch + 1 for ch in input_channels]}")
                    for trial in range(num_trials):
                        self.log_to_text(f"Trial {trial + 1}/{num_trials} for {freq} Hz, Output Channel {output_channel + 1}")
                        input_signal = generate_sine_signal(freq, fs, crosstalk_default_duration, input_level, dtype)
                        self.log_to_text(f"Generated input signal max amplitude: {np.max(np.abs(input_signal))}")
                        output_signals = record_output(
                            input_signal, fs, input_device_idx, output_device_idx,
                            input_channels, output_channel, dtype, self.log_to_text
                        )
                        if output_signals is None:
                            self.log_to_text("Error: Output signals are None. Using simulation data.")
                            output_signals = generate_output_signal(freq, fs, 0, len(input_channels), crosstalk_default_duration, dtype)
                        self.log_to_text(f"Output signals shape: {output_signals.shape}")
                        if output_signals.ndim != 2 or output_signals.shape[1] != len(input_channels):
                            self.log_to_text(f"Error: Invalid output signals shape: {output_signals.shape}. Expected {len(input_channels)} channels.")
                            continue
                        max_amplitude = 8388607 if dtype == np.int32 else (32767 if dtype == np.int16 else 1.0)
                        recorded_max = np.max(np.abs(output_signals))
                        if recorded_max > 0:
                            target_amplitude = max_amplitude * 10 ** (input_level / 20.0)
                            scale_factor = target_amplitude / recorded_max
                            output_signals = output_signals * scale_factor
                            self.log_to_text(f"Applied scale factor {scale_factor:.4f} to match input level {input_level} dBFS")
                        for ch_idx, ch in enumerate(input_channels):
                            channel_amplitude = np.max(np.abs(output_signals[:, ch_idx]))
                            self.log_to_text(f"Input channel amplitude (channel {ch + 1}): {channel_amplitude}")
                        freqs, magnitudes = compute_crosstalk_freq_response(
                            output_signals, fs, freq, input_level, valid_crosstalk_channels, self.log_to_text, dtype, input_signal=input_signal
                        )
                        if freqs.size == 0 or not magnitudes:
                            self.log_to_text(f"Error: Failed to compute frequency response for trial {trial + 1}.")
                            continue
                        freq_results.append((freqs, magnitudes))
                    if freq_results:
                        all_magnitudes = [mags for _, mags in freq_results]
                        mean_magnitudes = np.mean(all_magnitudes, axis=0)
                        self.log_to_text(f"Averaged magnitudes for {freq} Hz: {len(mean_magnitudes)} channels")
                        self.results_crosstalk['crosstalk_response'].append((freq, output_channel, [(freqs, mean_magnitudes)]))
                    else:
                        self.log_to_text(f"No valid results for {freq} Hz, Output Channel {output_channel + 1}")
            self.log_to_text(f"Crosstalk response: {len(self.results_crosstalk['crosstalk_response'])} frequencies")
            if not self.results_crosstalk['crosstalk_response']:
                self.log_to_text("Warning: No crosstalk data collected. Check channel settings and hardware connections.")
            self.display_crosstalk_results(test_frequencies, all_input_channels, main_input_channel, main_output_channel)
            self.plot_crosstalk_results(all_input_channels, main_input_channel, main_output_channel)
        except Exception as e:
            self.log_to_text(f"Error in crosstalk measurement: {e}")
            messagebox.showerror("Error", str(e))

    def measure_frequency_response(self):
            self.results_frequency_response = {
                'frequency_response': [[]],
                'input_thdn': [[] for _ in range(9)],
                'thdn_minus20db': [[]]
            }
            output_device_idx = self.device_names.index(self.output_device.get())
            device_info = sd.query_devices()[output_device_idx]
            fs = int(device_info['default_samplerate']) if isinstance(device_info['default_samplerate'], (int, float)) else int(device_info['default_samplerate'][0])
            self.log_to_text(f"Using sample rate from audio interface: {fs} Hz")
            bit_depth = self.bit_depth.get()
            input_device_idx = self.device_names.index(self.input_device.get())
            output_device_idx = self.device_names.index(self.output_device.get())
            input_channel = int(self.input_channel.get()) - 1
            output_channel = int(self.output_channel.get()) - 1
            input_level = float(self.input_level_db.get())
            if bit_depth == "16":
                dtype = np.int16
                max_amplitude = 32767
            elif bit_depth == "24":
                dtype = np.int32
                max_amplitude = 8388607
            else:
                dtype = np.float32
                max_amplitude = 1.0
            all_input_channels = [input_channel]
            if not all_input_channels or input_channel < 0:
                messagebox.showerror("Error", f"Invalid input channel: {input_channel + 1}.")
                self.log_to_text(f"Error: Invalid input channel {input_channel + 1}.")
                return
            device_info = sd.query_devices()[output_device_idx]
            supported_samplerates = [device_info['default_samplerate']] if isinstance(device_info['default_samplerate'], (int, float)) else device_info['default_samplerate']
            if fs not in supported_samplerates:
                self.log_to_text(f"Warning: Sample rate {fs} Hz not supported. Using closest supported rate: {supported_samplerates[0]} Hz.")
                fs = supported_samplerates[0]
            self.result_text.delete(1.0, tk.END)
            self.results_frequency_response['frequency_response'] = [[]]
            try:
                for trial in range(num_trials):
                    self.log_to_text(f"Sweep trial {trial + 1}/{num_trials}")
                    f_start = int(self.frequency_response_freq_var.get())
                    input_signal, t = generate_log_sweep(fs, frequency_response_default_duration, input_level, f_start=f_start, dtype=dtype)
                    self.log_to_text(f"Generated sweep signal max amplitude: {np.max(np.abs(input_signal))}")
                    output_signals = record_output(input_signal, fs, input_device_idx, output_device_idx, all_input_channels, output_channel, dtype, self.log_to_text, log_level_details=True)
                    if output_signals is None or output_signals.size == 0:
                        self.log_to_text("Error: No recorded data. Check audio interface routing and gain settings.")
                        continue

                    # Ensure output_signals is 2D
                    if output_signals.ndim == 1:
                        output_signals = output_signals[:, np.newaxis]
                    self.log_to_text(f"Output signals shape: {output_signals.shape}, Expected channels: {len(all_input_channels)}")
                    if output_signals.shape[1] < len(all_input_channels):
                        self.log_to_text(f"Error: Recorded channels {output_signals.shape[1]} less than expected {len(all_input_channels)}.")
                        continue

                    # Map input_channel to index in output_signals
                    try:
                        channel_idx = all_input_channels.index(input_channel)
                    except ValueError:
                        self.log_to_text(f"Error: Input channel {input_channel + 1} not in recorded channels.")
                        continue
                    freqs, magnitude = compute_sweep_response(input_signal, output_signals, fs, frequency_response_default_duration, channel_idx, input_level, self.log_to_text, f_start, dtype=dtype)
                    if freqs.size > 0 and len(freqs) == len(magnitude):
                        self.results_frequency_response['frequency_response'][0].append((freqs, magnitude))
                        self.log_to_text(f"Trial {trial + 1} added: {len(freqs)} frequency points")
                    else:
                        self.log_to_text(f"Error: Invalid sweep response for trial {trial + 1}. Freqs: {len(freqs)}, Magnitude: {len(magnitude)}")
                if not self.results_frequency_response['frequency_response'][0]:
                    self.log_to_text("Error: No valid frequency response data collected.")
                    return
                self.display_sweep_results(all_input_channels)
                self.plot_sweep_results(all_input_channels, key='frequency_response')
            except Exception as e:
                self.log_to_text(f"Error: {e}")
                messagebox.showerror("Error", str(e))

    # In AudioMeasurementApp class, update display_thd_results
    def display_thd_results(self, freq, all_input_channels, output_signals, dtype, key='thd'):
        max_amplitude = 1.0 if dtype == np.float32 else (32767 if dtype == np.int16 else 8388607)
        self.log_to_text(f"\n=== THD Results at {freq} Hz ===")
        input_channel = int(self.input_channel.get()) - 1
        ch_idx = 0
        ch = input_channel

        try:
            if key == 'input_thd':
                input_levels = [0, -10, -20, -30, -40, -50, -60, -70, -80]
                if ch_idx >= len(self.results_thd['input_thd']):
                    self.log_to_text(f"Error: Channel index {ch_idx} out of range for input_thd.")
                    return
                num_levels = min(len(self.results_thd['input_thd'][ch_idx]), len(input_levels))
                for level_idx in range(num_levels):
                    input_level = input_levels[level_idx]
                    if self.results_thd['input_thd'][ch_idx][level_idx]:
                        thd_results = self.results_thd['input_thd'][ch_idx][level_idx]
                        thd_results = thd_results if isinstance(thd_results, (list, np.ndarray)) else [thd_results]
                        mean_thd = np.mean(thd_results)
                        std_thd = np.std(thd_results) if len(thd_results) > 1 else 0
                        thdn_results = self.results_thd['input_thdn'][ch_idx][level_idx]
                        thdn_results = thdn_results if isinstance(thdn_results, list) else []
                        mean_thdn_db = np.mean([db for db, _, _ in thdn_results]) if thdn_results else 0
                        std_thdn_db = np.std([db for db, _, _ in thdn_results]) if len(thdn_results) > 1 else 0
                        mean_thdn_percent = np.mean([pct for _, pct, _ in thdn_results]) if thdn_results else 0
                        std_thdn_percent = np.std([pct for _, pct, _ in thdn_results]) if len(thdn_results) > 1 else 0
                        mean_thdn_dbfs = np.mean([dbfs for _, _, dbfs in thdn_results]) if thdn_results else 0
                        std_thdn_dbfs = np.std([dbfs for _, _, dbfs in thdn_results]) if len(thdn_results) > 1 else 0
                        dbfs_input = input_level
                        self.log_to_text(
                            f"Frequency: {freq} Hz, Input Level: {input_level} dBFS, Channel: {self.input_channel.get()}, "
                            f"THD: {mean_thd:.5f}%, THD Std Dev: {std_thd:.5f}%, "
                            f"THD+N: {mean_thdn_db:.5f} dB, THD+N Std Dev: {std_thdn_db:.5f} dB, "
                            f"THD+N: {mean_thdn_percent:.5f}%, THD+N Std Dev: {std_thdn_percent:.5f}%, "
                            f"THD+N: {mean_thdn_dbfs:.5f} dBFS, THD+N Std Dev: {std_thdn_dbfs:.5f} dBFS, "
                            f"dBFS Input: {dbfs_input:.5f}"
                        )
            else:
                thd_key = key
                thdn_key = f'thdn_{key.split("_")[1]}_{key.split("_")[2]}'
                if hasattr(self, 'thd_results') and isinstance(self.thd_results, dict):
                    freq_idx = self.thd_results['frequencies'].index(freq)
                    mean_thd = self.thd_results['thd'][freq_idx]
                    std_thd = self.thd_results['thd_std'][freq_idx]
                    mean_thdn_db = self.thd_results['thdn_db'][freq_idx]
                    std_thdn_db = self.thd_results['thdn_db_std'][freq_idx]
                    mean_thdn_percent = self.thd_results['thdn_percent'][freq_idx]
                    std_thdn_percent = self.thd_results['thdn_percent_std'][freq_idx]
                    mean_thdn_dbfs = self.thd_results['thdn_dbfs'][freq_idx]
                    std_thdn_dbfs = self.thd_results['thdn_dbfs_std'][freq_idx]
                    input_level = -20.0
                    dbfs_input = self.thd_results['dbfs_input'][freq_idx]
                    self.log_to_text(
                        f"Frequency: {freq} Hz, Input Level: {input_level} dBFS, Channel: {self.input_channel.get()}, "
                        f"THD: {mean_thd:.5f}%, THD Std Dev: {std_thd:.5f}%, "
                        f"THD+N: {mean_thdn_db:.5f} dB, THD+N Std Dev: {std_thdn_db:.5f} dB, "
                        f"THD+N: {mean_thdn_percent:.5f}%, THD+N Std Dev: {std_thdn_percent:.5f}%, "
                        f"THD+N: {mean_thdn_dbfs:.5f} dBFS, THD+N Std Dev: {std_thdn_dbfs:.5f} dBFS, "
                        f"dBFS Input: {dbfs_input:.5f}"
                    )
                else:
                    self.log_to_text(f"No THD data available for channel {ch + 1} at {freq} Hz")
        except Exception as e:
            self.log_to_text(f"Error in display_thd_results: {str(e)}")


    def display_crosstalk_results(self, test_frequencies, all_input_channels, main_input_channel, output_channel):
        self.result_text.delete(1.0, tk.END)
        try:
            if not self.results_crosstalk['crosstalk_response']:
                self.result_text.insert(tk.END, "No crosstalk response data available.\n")
                return
            self.result_text.insert(tk.END, "\n=== Inter-channel Crosstalk Results ===\n\n")
            ref_channel = main_input_channel
            if ref_channel not in all_input_channels:
                self.log_to_text(f"Warning: Reference channel {ref_channel + 1} not in input channels. Using channel {all_input_channels[0] + 1}.")
                ref_channel = all_input_channels[0]
            input_channel_map = {i: ch for i, ch in enumerate(all_input_channels)}
            self.log_to_text(f"Input channel map for display: {input_channel_map}")
            self.log_to_text(f"Reference channel: Input Channel {ref_channel + 1}")
            for freq in test_frequencies:
                self.result_text.insert(tk.END, f"At {freq} Hz:\n")
                freq_result = next((r for r in self.results_crosstalk['crosstalk_response'] if r[0] == freq), None)
                if not freq_result:
                    self.result_text.insert(tk.END, "No data available.\n\n")
                    continue
                freq, out_ch, freq_results = freq_result
                mean_magnitudes = freq_results[0][1]
                target_idx = np.argmin(np.abs(freq_results[0][0] - freq))
                self.log_to_text(f"Processing {freq} Hz, Output Channel {out_ch + 1}, Channels: {len(mean_magnitudes)}")
                ref_idx = all_input_channels.index(ref_channel)
                if ref_idx >= len(mean_magnitudes):
                    self.log_to_text(f"Error: Reference channel index {ref_idx} out of range for mean_magnitudes {len(mean_magnitudes)}")
                    continue
                ref_level = mean_magnitudes[ref_idx][target_idx]
                self.result_text.insert(tk.END, f"Output Channel {out_ch + 1} Input Channel {ref_channel + 1} Magnitude: {ref_level:.5f} dBFS\n")
                for ch_idx in range(len(mean_magnitudes)):
                    in_ch = input_channel_map.get(ch_idx)
                    if in_ch is None or in_ch == ref_channel:
                        continue
                    rms_db = mean_magnitudes[ch_idx][target_idx]
                    crosstalk_db = rms_db - ref_level
                    if crosstalk_db > 0:
                        self.log_to_text(f"Warning: Positive crosstalk ({crosstalk_db:.5f} dB) for Input Channel {in_ch + 1}. Check signal routing.")
                    self.result_text.insert(tk.END, f"Output Channel {out_ch + 1} Input Channel {in_ch + 1} Crosstalk: {crosstalk_db:.5f} dB\n")
                self.result_text.insert(tk.END, "\n")
        except Exception as e:
            self.log_to_text(f"Display error: {e}")
            self.result_text.insert(tk.END, f"Error: {e}\n")

    def display_sweep_results(self, all_input_channels):
        self.log_to_text("\n=== Frequency Response Results ===")
        self.log_to_text(f"Measured with log sweep from {self.frequency_response_freq_var.get()} Hz to Nyquist, averaged over {num_trials} trials, channel: {all_input_channels[0] + 1}")

    def plot_thd_results(self):
        self.thd_ax.clear()
        try:

            # Check if thd_results is a list (from measure_input_thdn) or dict (from measure_thd_ratio_minusXdb)
            if isinstance(self.thd_results, list):

                # Handle Measure Input THD+N data
                input_levels = [0, -10, -20, -30, -40, -50, -60, -70, -80]
                thd_values = []
                thdn_percent_values = []

                for level_idx, level in enumerate(input_levels):
                    if not self.thd_results[level_idx]['thd']:
                        self.log_to_text(f"No THD data to plot for level {level} dBFS.")
                        continue
                    thd = self.thd_results[level_idx]['thd'][0] if self.thd_results[level_idx]['thd'] else float('nan')
                    thdn_percent = self.thd_results[level_idx]['thdn_percent'][0] if self.thd_results[level_idx]['thdn_percent'] else float('nan')
                    thd_values.append(thd)
                    thdn_percent_values.append(thdn_percent)

                # Plot THD (%) with line connection
                self.thd_ax.plot(input_levels, thd_values, label="THD (%)", marker='o', color='red', linestyle='-', linewidth=1)

                # Plot THD+N (%) with line connection
                self.thd_ax.plot(input_levels, thdn_percent_values, label="THD+N (%)", marker='^', color='blue', linestyle='--', linewidth=1)

                title = 'THD and THD+N vs Input Level'
                self.thd_ax.set_xlabel('Input Level (dBFS)')
                self.thd_ax.set_xticks(input_levels)
            else:
                # Handle Measure THD+N (-20 dBFS) data
                frequencies = self.thd_results['frequencies']
                thd = self.thd_results['thd']
                thdn_percent = self.thd_results['thdn_percent']
                if not thd:
                    self.log_to_text("No THD data to plot.")
                    return

                # Plot THD (%) with line connection
                self.thd_ax.semilogx(frequencies, thd, label="THD (%)", marker='o', color='red', linestyle='-', linewidth=1)

                # Plot THD+N (%) with line connection
                self.thd_ax.semilogx(frequencies, thdn_percent, label="THD+N (%)", marker='^', color='blue', linestyle='--', linewidth=1)
                title = 'THD and THD+N vs Frequency'
                self.thd_ax.set_xlabel('Frequency (Hz)')

                # Set x-axis ticks and labels to show frequencies explicitly
                self.thd_ax.set_xticks(frequencies)
                self.thd_ax.set_xticklabels([str(int(f)) for f in frequencies], rotation=45)

            self.thd_ax.set_ylabel('Percentage (%)')
            self.thd_ax.set_title(title)
            self.thd_ax.grid(True)
            self.thd_ax.legend()
            self.thd_fig.tight_layout()
            self.thd_canvas.draw()
            self.thd_canvas.flush_events()
            self.log_to_text("THD graph updated")
        except Exception as e:
            self.log_to_text(f"Plotting error: {e}")

    def plot_sweep_results(self, all_input_channels, key='frequency_response'):
        self.ax1.clear()
        try:
            ch_idx = 0
            ch = all_input_channels[ch_idx]
            colors = ['blue']
            labels = {
                'input_thdn': 'Input THD+N',
                'thdn_minus20db': 'THD+N (-20 dBFS)',
                'frequency_response': 'Frequency Response'
            }
            title = ""
            if key in ['input_thdn', 'thdn_minus20db']:
                level_idx = 1 if key == 'input_thdn' else 0
                if key in self.results_frequency_response and self.results_frequency_response[key][level_idx]:
                    all_data = self.results_frequency_response[key][level_idx]
                    if not all_data:
                        self.log_to_text(f"No data to plot for {key} at level_idx {level_idx}")
                        return
                    all_freqs = all_data[0][0]
                    all_magnitudes = [data[1] for data in all_data]
                    mean_magnitudes = np.mean(all_magnitudes, axis=0)
                    std_magnitudes = np.std(all_magnitudes, axis=0) if len(all_magnitudes) > 1 else np.zeros_like(mean_magnitudes)
                    self.ax1.semilogx(all_freqs, mean_magnitudes, label=f"{labels[key]} Ch. {ch + 1}", color=colors[0])
                    self.ax1.fill_between(all_freqs, mean_magnitudes - std_magnitudes, mean_magnitudes + std_magnitudes, 
                                         alpha=0.2, color=colors[0], label=f"{labels[key]} Std Dev")
                    title = f"Frequency Response from {labels[key]}"
                    self.ax1.set_xlim(20, 20000)
                    self.ax1.set_ylim(-180, 10)
            elif key == 'frequency_response':
                level_idx = 0
                if key in self.results_frequency_response and self.results_frequency_response[key][level_idx]:
                    all_data = self.results_frequency_response[key][level_idx]
                    if not all_data:
                        self.log_to_text(f"No data to plot for {key} at level_idx {level_idx}")
                        return
                    all_freqs = all_data[0][0]
                    all_magnitudes = [data[1] for data in all_data]
                    mean_magnitudes = np.mean(all_magnitudes, axis=0)
                    std_magnitudes = np.std(all_magnitudes, axis=0) if len(all_magnitudes) > 1 else np.zeros_like(mean_magnitudes)
                    self.ax1.semilogx(all_freqs, mean_magnitudes, label=f"{labels[key]} Ch. {ch + 1}", color=colors[0])
                    self.ax1.fill_between(all_freqs, mean_magnitudes - std_magnitudes, mean_magnitudes + std_magnitudes, 
                                         alpha=0.2, color=colors[0], label=f"{labels[key]} Std Dev")
                    title = f"Frequency Response at {self.input_level_db.get()} dBFS"
                    output_device_idx = self.device_names.index(self.output_device.get())
                    device_info = sd.query_devices()[output_device_idx]
                    fs = int(device_info['default_samplerate']) if isinstance(device_info['default_samplerate'], (int, float)) else int(device_info['default_samplerate'][0])
                    self.log_to_text(f"Using sample rate from audio interface: {fs} Hz")
                    nyquist_freq = fs / 2
                    self.ax1.set_xlim(20, nyquist_freq)
                    self.ax1.set_ylim(-120, 10)
            self.ax1.set_xlabel("Frequency (Hz)")
            self.ax1.set_ylabel("Magnitude (dBFS)")
            self.ax1.set_title(title)
            self.ax1.grid(True, which="both")
            self.ax1.legend()
            self.fig.tight_layout()
            self.canvas.draw()
            self.log_to_text(f"Frequency response graph updated for channel {ch + 1} ({key})")
        except Exception as e:
            self.log_to_text(f"Plotting error: {e}")

    def plot_crosstalk_results(self, all_input_channels, main_input_channel, output_channel):
        self.ax1.clear()
        try:
            if not self.results_crosstalk['crosstalk_response']:
                self.log_to_text("No crosstalk response data available.")
                return
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
            nyquist_freq = int(self.sample_rate.get()) / 2
            test_frequencies = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]
            ref_channel = main_input_channel
            input_channel_map = {i: ch for i, ch in enumerate(all_input_channels)}
            for ch_idx in range(len(all_input_channels)):
                in_ch = input_channel_map.get(ch_idx)
                if in_ch is None:
                    continue
                crosstalk_values = []
                for freq in test_frequencies:
                    freq_result = next((r for r in self.results_crosstalk['crosstalk_response'] if r[0] == freq), None)
                    if not freq_result:
                        crosstalk_values.append(-150)
                        continue
                    freq, _, freq_results = freq_result
                    mean_magnitudes = freq_results[0][1]  # Use averaged magnitudes
                    target_idx = np.argmin(np.abs(freq_results[0][0] - freq))
                    rms_db = mean_magnitudes[ch_idx][target_idx]
                    if in_ch == ref_channel:
                        crosstalk_values.append(rms_db)
                    else:
                        ref_idx = all_input_channels.index(ref_channel)
                        ref_level = mean_magnitudes[ref_idx][target_idx]
                        crosstalk_db = rms_db - ref_level
                        if crosstalk_db > 0:
                            self.log_to_text(f"Warning: Positive crosstalk ({crosstalk_db:.5f} dB) at {freq} Hz for Input Channel {in_ch + 1}")
                        crosstalk_values.append(crosstalk_db)
                if any(v != -150 for v in crosstalk_values):
                    label = f"In Ch. {in_ch + 1} (Reference)" if in_ch == ref_channel else f"In Ch. {in_ch + 1}"
                    self.ax1.semilogx(test_frequencies, crosstalk_values, label=label, color=colors[ch_idx % len(colors)])
                    for f, crosstalk in zip(test_frequencies, crosstalk_values):
                        if crosstalk != -150:
                            self.ax1.plot(f, crosstalk, 'o', color=colors[ch_idx % len(colors)])
            self.ax1.set_xlabel("Frequency (Hz)")
            self.ax1.set_ylabel("Magnitude (dBFS for Reference, dB relative to Reference for Crosstalk Input)")
            self.ax1.set_title(f"Crosstalk Response for Output Channel {output_channel + 1}")
            self.ax1.grid(True, which="both")
            self.ax1.legend()
            self.ax1.set_ylim(-180, 0)
            self.ax1.set_xlim(0, 30000)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.log_to_text(f"Plotting error: {e}")

    def save_graph(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            self.log_to_text("Graph save canceled by user.")
            return
        try:
            # Ensure the directory exists and is writable
            directory = os.path.dirname(file_path) or '.'
            os.makedirs(directory, exist_ok=True)
            if not os.access(directory, os.W_OK):
                raise PermissionError(f"Directory {directory} is not writable.")
            self.log_to_text(f"Attempting to save graph to: {file_path}")
            # Save the active tab's graph
            active_tab = self.results_notebook.index(self.results_notebook.select())
            if active_tab == 1:  # Graph tab
                self.canvas.draw()  # Ensure canvas is updated
                self.log_to_text("Drawing Graph canvas")
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.log_to_text("Graph canvas saved")
            elif active_tab == 2:  # THD Graph tab
                self.thd_canvas.draw()  # Ensure canvas is updated
                self.log_to_text("Drawing THD Graph canvas")
                self.thd_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.log_to_text("THD Graph canvas saved")
            # Verify file existence with additional delay and retry
            for _ in range(3):
                time.sleep(0.2)  # Increased delay to handle sync issues
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        self.log_to_text(f"Graph saved to {file_path} (Size: {file_size} bytes)")
                        messagebox.showinfo("Success", f"Graph saved to {file_path}")
                        return
            raise FileNotFoundError(f"File not found or empty at {file_path} after save attempt.")
        except PermissionError as pe:
            self.log_to_text(f"Permission error saving graph: {str(pe)}. Check directory permissions or try a local path.")
            messagebox.showerror("Error", f"Permission denied: {str(pe)}. Try saving to a local directory.")
        except FileNotFoundError as fnf:
            self.log_to_text(f"File not found error: {str(fnf)}. Verify storage device is connected and writable.")
            messagebox.showerror("Error", f"File not saved: {str(fnf)}. Check storage device.")
        except Exception as e:
            self.log_to_text(f"Error saving graph: {str(e)}. Check file path and permissions.")
            messagebox.showerror("Error", f"Failed to save graph: {str(e)}. Check file path and permissions.")

    def save_sweep_txt(self):
        if (not self.results_frequency_response['frequency_response'] and 
            not self.results_frequency_response['input_thdn'] and 
            not self.results_frequency_response['thdn_minus20db']):
            messagebox.showerror("Error", "No frequency response data available.")
            self.log_to_text("Error: No data to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            source_names = {
                'frequency_response': 'Frequency Response',
                'input_thdn': 'Input THD+N (-10 dBFS)',
                'thdn_minus20db': 'THD+N (-20 dBFS)'
            }
            with open(file_path, 'w') as f:
                f.write("Source, Channel, Frequency (Hz), Magnitude (dBFS)\n")
                for plot_key in ['frequency_response', 'input_thdn', 'thdn_minus20db']:
                    for ch_idx, ch in enumerate(range(len(self.results_frequency_response[plot_key]))):
                        if not self.results_frequency_response[plot_key][ch_idx]:
                            continue
                        all_data = self.results_frequency_response[plot_key][ch_idx]
                        freqs = all_data[0][0]
                        all_magnitudes = [data[1] for data in all_data]
                        mean_magnitudes = np.mean(all_magnitudes, axis=0)
                        for freq, mean_mag in zip(freqs, mean_magnitudes):
                            f.write(f"{source_names[plot_key]}, {ch + 1}, {freq:.5f}, {mean_mag:.5f}\n")
            self.log_to_text(f"Data saved to {file_path}")
            messagebox.showinfo("Success", f"Data saved to {file_path}")

    def save_crosstalk_csv(self):
        if not self.results_crosstalk['crosstalk_response']:
            messagebox.showerror("Error", "No crosstalk response data available.")
            self.log_to_text("Error: No crosstalk data to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)

                    # Write title
                    writer.writerow(["=== Inter-channel Crosstalk Results ==="])

                    # Define header
                    ref_channel = int(self.input_channel.get()) - 1
                    all_input_channels = sorted(set(
                        ch for freq, out_ch, freq_results in self.results_crosstalk['crosstalk_response']
                        for _, mags in freq_results for ch in range(len(mags))
                    ))
                    if not all_input_channels:
                        self.log_to_text("Error: No input channels found in crosstalk data.")
                        return

                    # Validate ref_channel
                    if ref_channel not in all_input_channels:
                        self.log_to_text(f"Warning: Reference channel {ref_channel + 1} not in input channels. Using channel {all_input_channels[0] + 1}.")
                        ref_channel = all_input_channels[0]
                    input_channel_map = {i: ch for i, ch in enumerate(all_input_channels)}
                    header = ["Frequency (Hz)"] + [
                        f"Channel {ch + 1} ({'dBFS' if ch == ref_channel else 'dB'})"
                        for ch in all_input_channels
                    ]
                    writer.writerow(header)

                    # Write data for each frequency
                    test_frequencies = [20, 40, 80, 160, 315, 630, 1250, 2500, 5000, 10000, 20000]
                    for freq in test_frequencies:
                        freq_result = next((r for r in self.results_crosstalk['crosstalk_response'] if r[0] == freq), None)
                        if not freq_result:
                            self.log_to_text(f"Warning: No data for frequency {freq} Hz.")
                            continue
                        freq, out_ch, freq_results = freq_result
                        all_magnitudes = [mags for _, mags in freq_results]
                        mean_magnitudes = np.mean(all_magnitudes, axis=0)
                        target_idx = np.argmin(np.abs(freq_results[0][0] - freq))
                        ref_idx = all_input_channels.index(ref_channel)
                        ref_level = mean_magnitudes[ref_idx][target_idx] if ref_idx < len(mean_magnitudes) else 0
                        row = [f"{freq:.5f}"]
                        for ch_idx in range(len(all_input_channels)):
                            in_ch = input_channel_map.get(ch_idx, ch_idx)
                            if ch_idx >= len(mean_magnitudes):
                                row.append("")
                                continue
                            rms_db = mean_magnitudes[ch_idx][target_idx]
                            if in_ch == ref_channel:
                                level = f"{rms_db:.5f}"  # dBFS for reference channel
                            else:
                                level = f"{rms_db - ref_level:.5f}" if ref_level != 0 else f"{rms_db:.5f}"  # dB for crosstalk
                            row.append(level)
                        writer.writerow(row)
                self.log_to_text(f"Crosstalk results saved to {file_path}")
                messagebox.showinfo("Success", f"Crosstalk results saved to {file_path}")
            except Exception as e:
                self.log_to_text(f"Error saving crosstalk results: {e}")
                messagebox.showerror("Error", f"Failed to save crosstalk results: {e}")

    def save_results(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["=== THD and THD+N Results ==="])
                writer.writerow([
                    "Frequency (Hz)", "Input Level (dBFS)", "Channel",
                    "THD (%)", "THD Std Dev (%)",
                    "THD+N (dB)", "THD+N Std Dev (dB)",
                    "THD+N (%)", "THD+N Std Dev (%)",
                    "THD+N (dBFS)", "THD+N Std Dev (dBFS)",
                    "dBFS Input"
                ])

                # Check if thd_results exists and has data
                if not hasattr(self, 'thd_results') or not self.thd_results:
                    self.log_to_text("Error: No THD results available to save.")
                    messagebox.showerror("Error", "No THD results available to save.")
                    return

                # Handle Measure Input THD+N data (self.thd_results is a list)
                if isinstance(self.thd_results, list):
                    input_levels = [0, -10, -20, -30, -40, -50, -60, -70, -80]
                    for level_idx, input_level in enumerate(input_levels):
                        if level_idx >= len(self.thd_results) or not self.thd_results[level_idx]['thd']:
                            continue

                        # Write only the latest valid data for this input level
                        thd = self.thd_results[level_idx]['thd'][0] if self.thd_results[level_idx]['thd'] else 0
                        thd_std = self.thd_results[level_idx]['thd_std'][0] if self.thd_results[level_idx]['thd_std'] else 0
                        thdn_db = self.thd_results[level_idx]['thdn_db'][0] if self.thd_results[level_idx]['thdn_db'] else 0
                        thdn_db_std = self.thd_results[level_idx]['thdn_db_std'][0] if self.thd_results[level_idx]['thdn_db_std'] else 0
                        thdn_percent = self.thd_results[level_idx]['thdn_percent'][0] if self.thd_results[level_idx]['thdn_percent'] else 0
                        thdn_percent_std = self.thd_results[level_idx]['thdn_percent_std'][0] if self.thd_results[level_idx]['thdn_percent_std'] else 0
                        thdn_dbfs = self.thd_results[level_idx]['thdn_dbfs'][0] if self.thd_results[level_idx]['thdn_dbfs'] else 0
                        thdn_dbfs_std = self.thd_results[level_idx]['thdn_dbfs_std'][0] if self.thd_results[level_idx]['thdn_dbfs_std'] else 0
                        dbfs_input = self.thd_results[level_idx]['dbfs_input'][0] if self.thd_results[level_idx]['dbfs_input'] else input_level
                        writer.writerow([
                            997, input_level, f"Channel {self.input_channel.get()}",
                            f"{thd:.5f}", f"{thd_std:.5f}",
                            f"{thdn_db:.5f}", f"{thdn_db_std:.5f}",
                            f"{thdn_percent:.5f}", f"{thdn_percent_std:.5f}",
                            f"{thdn_dbfs:.5f}", f"{thdn_dbfs_std:.5f}",
                            f"{dbfs_input:.5f}"
                        ])

                # Handle Measure THD+N (-20 dBFS) data (self.thd_results is a dict)
                elif isinstance(self.thd_results, dict):
                    test_frequencies = self.thd_results['frequencies']
                    input_level = -20.0
                    for freq_idx, freq in enumerate(test_frequencies):
                        if freq_idx >= len(self.thd_results['thd']):
                            continue

                        # Write only the latest valid data for this frequency
                        thd = self.thd_results['thd'][freq_idx] if self.thd_results['thd'] else 0
                        thd_std = self.thd_results['thd_std'][freq_idx] if self.thd_results['thd_std'] else 0
                        thdn_db = self.thd_results['thdn_db'][freq_idx] if self.thd_results['thdn_db'] else 0
                        thdn_db_std = self.thd_results['thdn_db_std'][freq_idx] if self.thd_results['thdn_db_std'] else 0
                        thdn_percent = self.thd_results['thdn_percent'][freq_idx] if self.thd_results['thdn_percent'] else 0
                        thdn_percent_std = self.thd_results['thdn_percent_std'][freq_idx] if self.thd_results['thdn_percent_std'] else 0
                        thdn_dbfs = self.thd_results['thdn_dbfs'][freq_idx] if self.thd_results['thdn_dbfs'] else 0
                        thdn_dbfs_std = self.thd_results['thdn_dbfs_std'][freq_idx] if self.thd_results['thdn_dbfs_std'] else 0
                        dbfs_input = self.thd_results['dbfs_input'][freq_idx] if self.thd_results['dbfs_input'] else input_level
                        writer.writerow([
                            freq, input_level, f"Channel {self.input_channel.get()}",
                            f"{thd:.5f}", f"{thd_std:.5f}",
                            f"{thdn_db:.5f}", f"{thdn_db_std:.5f}",
                            f"{thdn_percent:.5f}", f"{thdn_percent_std:.5f}",
                            f"{thdn_dbfs:.5f}", f"{thdn_dbfs_std:.5f}",
                            f"{dbfs_input:.5f}"
                        ])

                self.log_to_text(f"THD results saved to {file_path} as CSV")
                messagebox.showinfo("Success", f"THD results saved to {file_path}")
        except Exception as e:
            self.log_to_text(f"Error saving THD Results: {str(e)}")
            messagebox.showerror("Error saving THD results", str(e))

    def save_thd_graph(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if not file_path:
            self.log_to_text("THD Graph save canceled by user.")
            return
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            # Save the THD Graph
            self.thd_canvas.draw()  # Ensure canvas is updated
            self.thd_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.log_to_text(f"THD Graph saved to {file_path}")
            messagebox.showinfo("Success", f"THD Graph saved to {file_path}")
        except Exception as e:
            self.log_to_text(f"Error saving THD Graph: {str(e)}. Check file path and permissions.")
            messagebox.showerror("Error", f"Failed to save THD Graph: {str(e)}. Check file path and permissions.")

    def save_load_preset(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                preset = json.load(f)
            self.input_device.set(preset["input_device"] if preset["input_device"] in self.device_names else self.device_names[0])
            self.output_device.set(preset["output_device"] if preset["output_device"] in self.device_names else self.device_names[0])
            self.update_channel_options()
            self.input_channel.set(preset["input_channel"])
            self.output_channel.set(preset["output_channel"])
            crosstalk_inputs = preset.get("crosstalk_inputs", [])
            for i, var in enumerate(self.crosstalk_channel_vars):
                var.set(str(i + 1) in crosstalk_inputs)
            self.update_crosstalk_channels()
            self.sample_rate.set(preset["sample_rate"])
            self.bit_depth.set(preset["bit_depth"])
            self.crosstalk_freq_var.set(preset.get("THD_frequency", str(THD_frequencies[0])))
            self.frequency_response_freq_var.set(preset.get("frequency_response_start", str(frequency_response_frequencies[0])))
            self.include_thdn.set(preset.get("include_thdn", False))
            self.input_level_db.set(preset.get("input_level_db", "-6"))
            self.log_to_text(f"Preset loaded from {file_path}")
            messagebox.showinfo("Success", f"Preset loaded from {file_path}")

    def save_preset(self):
        preset = {
            "input_device": self.input_device.get(),
            "output_device": self.output_device.get(),
            "input_channel": self.input_channel.get(),
            "output_channel": self.output_channel.get(),
            "crosstalk_inputs": [str(i + 1) for i, var in enumerate(self.crosstalk_channel_vars) if var.get()],
            "sample_rate": self.sample_rate.get(),
            "bit_depth": self.bit_depth.get(),
            "THD_frequency": self.crosstalk_freq_var.get(),
            "frequency_response_start": self.frequency_response_freq_var.get(),
            "include_thdn": self.include_thdn.get(),
            "input_level_db": self.input_level_db.get()
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(preset, f, indent=4)
            self.log_to_text(f"Preset saved to {file_path}")
            messagebox.showinfo("Success", f"Preset saved to {file_path}")

    def copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.result_text.get("1.0", tk.END))
        self.log_to_text("Log copied to clipboard")

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = AudioMeasurementApp(root)
    root.mainloop()
