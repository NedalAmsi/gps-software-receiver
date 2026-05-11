# acquisition.py
# GPS L1 C/A Acquisition — FFT-based Parallel Code-Phase Search
# Operates on complex baseband signal (output of signal_simulator.py)
# Author: Nedal Amsi

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prn_generator import generate_PRN

CODE_RATE   = 1.023e6
CODE_LENGTH = 1023


def acquire(signal, sv_id, fs,
            doppler_range=5000,
            doppler_step=500):
    """
    FFT-based parallel code-phase acquisition for complex baseband signal.

    For each Doppler bin:
      1. Wipe off residual Doppler from baseband signal
      2. Correlate with local PRN replica using FFT (all code phases at once)
      3. Store correlation power across all phases

    The true Doppler bin + code phase shows a dominant peak.

    Args:
        signal       : Complex baseband signal (numpy array)
        sv_id        : Satellite PRN to search for (1-32)
        fs           : Sampling frequency (Hz)
        doppler_range: Search ± Hz around baseband (default ±5000 Hz)
        doppler_step : Doppler resolution (Hz)

    Returns:
        dict with detection result, Doppler estimate, code phase, corr map
    """
    # Use exactly 1ms of signal — one PRN period
    samples_per_ms  = int(fs / 1000)
    signal_1ms      = signal[:samples_per_ms]
    t               = np.arange(samples_per_ms) / fs

    # Local PRN replica (upsampled to match fs)
    prn              = generate_PRN(sv_id)
    samples_per_chip = fs / CODE_RATE
    prn_up = np.array([
        prn[int(i / samples_per_chip) % CODE_LENGTH]
        for i in range(samples_per_ms)
    ], dtype=np.float64)
    prn_up = (1 - 2 * prn_up)         # bipolar: {0→+1, 1→-1}

    # Pre-compute FFT of PRN replica (conjugate for correlation)
    prn_fft_conj = np.conj(np.fft.fft(prn_up))

    # Doppler search grid
    doppler_bins = np.arange(-doppler_range,
                              doppler_range + doppler_step,
                              doppler_step)
    corr_map = np.zeros((len(doppler_bins), samples_per_ms))

    print(f"  PRN {sv_id} — searching {len(doppler_bins)} Doppler bins "
          f"[{-doppler_range} to +{doppler_range} Hz, "
          f"step {doppler_step} Hz]")

    for k, doppler in enumerate(doppler_bins):
        # Wipe off residual Doppler from baseband signal
        wipeoff  = np.exp(-1j * 2 * np.pi * doppler * t)
        baseband = signal_1ms * wipeoff

        # FFT-based circular correlation over all code phases
        bb_fft  = np.fft.fft(np.real(baseband))   # use I channel
        corr    = np.abs(np.fft.ifft(bb_fft * prn_fft_conj))**2
        corr_map[k, :] = corr

    # Find peak
    peak_bin, peak_samp = np.unravel_index(
        np.argmax(corr_map), corr_map.shape)
    peak_doppler  = doppler_bins[peak_bin]
    peak_value    = corr_map[peak_bin, peak_samp]
    mean_value    = np.mean(corr_map)
    peak_ratio    = peak_value / (mean_value + 1e-12)

    THRESHOLD = 2.5
    detected  = peak_ratio > THRESHOLD
    peak_chip = int(peak_samp / samples_per_chip)

    print(f"  Peak Doppler  : {peak_doppler:+.0f} Hz  "
          f"(true: +1500 Hz)")
    print(f"  Peak code phase: chip {peak_chip}")
    print(f"  Peak/mean ratio: {peak_ratio:.2f}  "
          f"(threshold = {THRESHOLD})")
    print(f"  Result        : "
          f"{'✅ DETECTED' if detected else '❌ NOT FOUND'}")

    return {
        'sv_id'       : sv_id,
        'detected'    : detected,
        'peak_doppler': peak_doppler,
        'peak_chip'   : peak_chip,
        'peak_samp'   : peak_samp,
        'peak_ratio'  : peak_ratio,
        'corr_map'    : corr_map,
        'doppler_bins': doppler_bins,
        'fs'          : fs,
    }


def plot_acquisition(r):
    sv           = r['sv_id']
    corr_map     = r['corr_map']
    doppler_bins = r['doppler_bins']
    peak_doppler = r['peak_doppler']
    peak_samp    = r['peak_samp']
    fs           = r['fs']

    samples_per_chip = (fs / 1000) / CODE_LENGTH
    phases_chips     = np.arange(corr_map.shape[1]) / samples_per_chip

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'GPS L1 C/A Acquisition — PRN {sv}', fontsize=13)

    # 2D colour map
    im = axes[0].imshow(
        corr_map,
        aspect='auto',
        extent=[0, CODE_LENGTH,
                doppler_bins[-1], doppler_bins[0]],
        cmap='viridis'
    )
    axes[0].axhline(peak_doppler, color='red', lw=1.5,
                    ls='--', label=f'Peak Doppler: {peak_doppler:+.0f} Hz')
    axes[0].set_xlabel('Code phase (chips)')
    axes[0].set_ylabel('Doppler frequency (Hz)')
    axes[0].set_title('2D Acquisition search grid')
    axes[0].legend(fontsize=9)
    plt.colorbar(im, ax=axes[0], label='Correlation power')

    # 1D slice at best Doppler
    best_bin = np.where(doppler_bins == peak_doppler)[0][0]
    axes[1].plot(phases_chips, corr_map[best_bin, :],
                 color='#1D9E75', lw=1.0)
    axes[1].axvline(peak_samp / samples_per_chip,
                    color='red', lw=1.5, ls='--',
                    label=f'Peak: chip {r["peak_chip"]}')
    axes[1].set_xlabel('Code phase (chips)')
    axes[1].set_ylabel('Correlation power')
    axes[1].set_title(f'Correlation at Doppler = {peak_doppler:+.0f} Hz')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn{sv}_acquisition.png', dpi=150)
    print(f"Saved → OUT/figures/prn{sv}_acquisition.png")


def plot_3d_surface(r):
    sv           = r['sv_id']
    corr_map     = r['corr_map']
    doppler_bins = r['doppler_bins']
    fs           = r['fs']

    samples_per_chip = (fs / 1000) / CODE_LENGTH
    phase_chips      = np.arange(corr_map.shape[1]) / samples_per_chip

    # Downsample for 3D performance
    step      = 8
    map_ds    = corr_map[:, ::step]
    phases_ds = phase_chips[::step]
    D, P      = np.meshgrid(doppler_bins, phases_ds, indexing='ij')

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, D, map_ds, cmap='viridis',
                    alpha=0.9, linewidth=0, antialiased=True)
    ax.set_xlabel('Code phase (chips)', labelpad=10)
    ax.set_ylabel('Doppler (Hz)',       labelpad=10)
    ax.set_zlabel('Correlation power',  labelpad=10)
    ax.set_title(f'3D Acquisition Surface — PRN {sv}', fontsize=13)

    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn{sv}_acquisition_3d.png', dpi=150)
    print(f"Saved → OUT/figures/prn{sv}_acquisition_3d.png")


if __name__ == "__main__":

    print("Loading signal (1ms window for acquisition)...")
    signal_full = np.load('OUT/prn1_rawsignal.npy')
    fs          = 5e6
    print(f"Signal: {len(signal_full)} samples = "
          f"{len(signal_full)/fs*1000:.0f} ms total\n")

    # Acquire PRN 1 — should detect at Doppler = +1500 Hz
    print("Acquiring PRN 1...")
    r1 = acquire(signal_full, sv_id=1, fs=fs)

    plot_acquisition(r1)
    plot_3d_surface(r1)

    # Attempt PRN 5 — not in signal, should fail
    print("\nAcquiring PRN 5 (not present — expect NOT FOUND)...")
    r5 = acquire(signal_full, sv_id=5, fs=fs)
    print(f"  PRN 5 result: "
          f"{'DETECTED (false alarm!)' if r5['detected'] else 'NOT FOUND ✅'}")

    print("\nDone. Check OUT/figures/ for acquisition plots.")
