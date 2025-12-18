import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel


MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

import torch

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False, device=None):
    """
    Computes linear spectrogram (STFT) from waveform tensor
    Safe version for VITS training on Windows + single sample support.
    """
    # Convert 1D tensor to [1, 1, T]
    if y.dim() == 1:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.dim() == 2:
        y = y.unsqueeze(1)  # [B, 1, T]

    y = y.float()
    if device is not None:
        y = y.to(device)

    # -----------------------
    # Safe reflect padding
    # -----------------------
    pad_amount = int((n_fft - hop_size) / 2)
    if pad_amount > 0:
        # If batch size is 1, slicing still works
        left = torch.flip(y[:, :, :pad_amount], [2])
        right = torch.flip(y[:, :, -pad_amount:], [2])
        y = torch.cat([left, y, right], dim=2)

    # -----------------------
    # Compute STFT
    # -----------------------
    spec = torch.stft(
        y.squeeze(1),  # [B, T]
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        center=center,
        return_complex=True
    )

    spec = torch.view_as_real(spec)  # [B, F, T, 2]
    spec = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-9)  # magnitude
    return spec



import librosa


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    
    if fmax_dtype_device not in mel_basis:
        # 1. You calculate the mel filter bank and store it in 'mel_filter'
        mel_filter = librosa.filters.mel(sr=sampling_rate,
                               n_fft=n_fft,
                               n_mels=num_mels,
                               fmin=fmin,
                               fmax=fmax)
        
        # 2. FIX: You must use the 'mel_filter' variable here, not 'mel'
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel_filter).to(
                                            dtype=spec.dtype, 
                                            device=spec.device
                                       )
    
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


# def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
#     # 1. Input Checks (Optional, keeps current behavior)
#     if torch.min(y) < -1.:
#         print('min value is ', torch.min(y))
#     if torch.max(y) > 1.:
#         print('max value is ', torch.max(y))

#     global mel_basis, hann_window
#     dtype_device = str(y.dtype) + '_' + str(y.device)
#     fmax_dtype_device = str(fmax) + '_' + dtype_device
#     wnsize_dtype_device = str(win_size) + '_' + dtype_device

#     # 2. Cache Mel Basis (FIXED LOGIC)
#     if fmax_dtype_device not in mel_basis:
#         # Calculate the Mel filter bank array using librosa
#         mel_filter = librosa.filters.mel(sr=sampling_rate,
#                                n_fft=n_fft,
#                                n_mels=num_mels,
#                                fmin=fmin,
#                                fmax=fmax)
        
#         # Convert the NumPy array to a PyTorch tensor and store it in the global cache
#         mel_basis[fmax_dtype_device] = torch.from_numpy(mel_filter).to(dtype=y.dtype, device=y.device)
#         # mel_basis[fmax_dtype_device] = torch.from_numpy(mel_filter).T.to(
#         #                             dtype=y.dtype,
#         #                             device=y.device
#         #                        )
#         # 🛑 REMOVED PREMATURE RETURN STATEMENT (mel = torch.matmul(...) followed by return mel)
        
#     # 3. Cache Hann Window
#     if wnsize_dtype_device not in hann_window:
#         hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

#     is_batched = y.dim() == 2
#     if is_batched:
#         batch_size = y.size(0)
#     else:
#         batch_size = 1
#         y = y.unsqueeze(0) 

#     # 4. Padding
#     pad_amount = int((n_fft - hop_size) / 2)
#     min_audio_length = n_fft

#     if y.size(-1) < min_audio_length:
#         y = torch.nn.functional.pad(y, (0, min_audio_length - y.size(-1)), mode='constant', value=0)
#     y = torch.nn.functional.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode='reflect')
#     y = y.squeeze(1)

#     # 5. STFT Calculation
#     # Note: Adding return_complex=True is now recommended/required for real inputs
#     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
#                       center=center, pad_mode='reflect', normalized=False, onesided=True,
#                       return_complex=True)

#     # 6. Magnitude Calculation
#     # Convert complex STFT output to real magnitude spectrogram
#     spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

#     if spec.dtype != torch.float:
#         spec = spec.real.float()

#     # 7. Mel Transformation
#     # Apply Mel filter bank matrix multiplication using the cached basis
#         # 7. Mel Transformation
#     # spec shape: [batch, freq_bins, time_steps] or [freq_bins, time_steps]
#     if spec.dim() == 3:
#         # [batch, freq_bins, time] @ [freq_bins, n_mels] -> [batch, n_mels, time]
#         spec = torch.einsum('bft,fm->bmt', spec, mel_basis[fmax_dtype_device])
#     else:
#         spec = spec.transpose(0, 1)
#         spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
#         spec = spec.unsqueeze(0) if spec.dim() == 2 else spec
    
#     # 8. Normalization
#     spec = spectral_normalize_torch(spec)

#     return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # 1. Input Checks
    if torch.min(y) < -1. or torch.max(y) > 1.:
        print(f'WARNING: Audio waveform values outside [-1, 1]. Min: {torch.min(y).item():.4f}, Max: {torch.max(y).item():.4f}')

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device

    # 2. Cache Mel Basis [M, F]
    if fmax_dtype_device not in mel_basis:
        # Calculate the Mel filter bank array using librosa
        mel_filter = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        # Store basis tensor in cache. Shape: [Mels, Freq_Bins] -> [M, F]
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel_filter).to(dtype=y.dtype, device=y.device)
        
    # 3. Cache Hann Window
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # 4. Uniform Batching and Padding
    
    # Track if input was originally batched (2D) or unbatched (1D)
    is_batched = y.dim() == 2
    
    # Ensure y is always [B, T]
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Calculate padding amounts
    pad_amount = int((n_fft - hop_size) / 2)
    min_audio_length = n_fft

    # Pad for min length (Handles cases where audio is too short for n_fft)
    if y.size(-1) < min_audio_length:
        y = torch.nn.functional.pad(y, (0, min_audio_length - y.size(-1)), mode='constant', value=0)
        
    # Apply reflect padding for STFT (Input is [B, T])
    y = torch.nn.functional.pad(y, (pad_amount, pad_amount), mode='reflect')
    
    # 5. STFT Calculation & Magnitude
    
    # STFT: [B, T] -> [B, F, Time] (if onesided=True and center=False)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)

    # Magnitude: Convert complex [B, F, Time] to real float magnitude [B, F, Time]
    # torch.abs handles the magnitude calculation (sqrt(real^2 + imag^2))
    spec = torch.abs(spec) 

    # 6. Mel Transformation (Robust Batch MatMul)
    
    mel_basis_tensor = mel_basis[fmax_dtype_device] # Shape [M, F] (e.g., 80, 513)
    
    # 1. Transpose Spectrogram: [B, F, T] -> [B, T, F] 
    # This aligns Freq (F) for batch matrix multiplication.
    spec = spec.transpose(1, 2) 

    # 2. Transpose Mel Basis: [M, F] -> [F, M] (e.g., 513, 80)
    mel_basis_T = mel_basis_tensor.transpose(0, 1)

    # 3. Batch MatMul: [B, T, F] @ [F, M] -> [B, T, M]
    spec = torch.matmul(spec, mel_basis_T) 

    # 4. Final Transpose: [B, T, M] -> [B, M, T] (Standard Mel Spectrogram format)
    spec = spec.transpose(1, 2) 
    
    # 7. Normalization
    spec = spectral_normalize_torch(spec)
    
    # 8. Final Output Reshaping
    # Remove the batch dimension if it was added for a single input
    if not is_batched:
        spec = spec.squeeze(0)

    return spec

