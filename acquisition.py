# acquisition.py
# GPS L1 C/A Signal Acquisition — FFT-based Parallel Code-Phase Search
# Searches for a satellite signal across all code phases and Doppler bins
# Author: Nedal Amsi

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prn_generator import generate_PRN

# ── Constants ─────────────────────────────────────────────────
CODE_RATE   = 1.023e6   # C/A chipping rate (chips/sec)
CODE_LENGTH = 1023      # chips per PRN period
L1_FREQ     = 1575.42e6 # L1 carrier frequency (Hz)


def acquire(signal, sv_id, fs, doppler_range=5000, doppler_step=500):
    """
    FFT-based parallel code-phase search acquisition.

    For each Doppler bin:
      1. Wipe off the carrier at that Doppler frequency
      2. FFT the result
      3. Correlate with local PRN replica in frequency domain
      4. IFFT → get correlation peak across all 1023 code phases at once

    Args:
        signal       : Received signal (numpy array)
        sv_id        : Satellite PRN to search for (1-32)
        fs           : Sampling frequency (Hz)
        doppler_range: Search range ± Hz (default ±5000 Hz)
        doppler_step : Frequency resolution (default 500 Hz)

    Returns:
        results dict with peak info and full correlation map
    """

    num_samples  = len(signal)
    t            = np.arange(num_samples) / fs

    # ── Local PRN replica (upsampled) ─────────────────────────
    prn          = generate_PRN(sv_id)
    samples_per_chip = fs / CODE_RATE
    prn_upsampled = np.array([
        prn[int(i / samples_per_chip) % CODE_LENGTH]
        for i in range(num_samples)
    ])

    # FFT of local PRN replica — computed once, reused for all Doppler bins
    prn_fft_conj = np.conj(np.fft.fft(prn_upsampled))

    # ── Doppler search grid ───────────────────────────────────
    doppler_bins  = np.arange(-doppler_range,
                               doppler_range + doppler_step,
                               doppler_step)
    num_bins      = len(doppler_bins)
    num_phases    = num_samples

    # Store correlation results: rows = Doppler bins, cols = code phases
    corr_map      = np.zeros((num_bins, num_phases))

    print(f"  Searching {num_bins} Doppler bins "
          f"({-doppler_range} to +{doppler_range} Hz, step {doppler_step} Hz)")

    for i, doppler in enumerate(doppler_bins):

        # Step 1: Wipe off carrier at this Doppler frequency
        # This mixes the signal down to baseband assuming this Doppler
        IF_FREQ = L1_FREQ % fs
        carrier_wipeoff = np.exp(-1j * 2 * np.pi * (IF_FREQ + doppler) * t)
        baseband        = signal * carrier_wipeoff

        # Step 2: FFT of baseband signal
        baseband_fft    = np.fft.fft(baseband)

        # Step 3: Multiply with conjugate of PRN FFT (circular correlation)
        correlation_fft = baseband_fft * prn_fft_conj

        # Step 4: IFFT → correlation peak at correct code phase
        correlation     = np.abs(np.fft.ifft(correlation_fft))

        corr_map[i, :]  = correlation

    # ── Find the peak ─────────────────────────────────────────
    peak_bin, peak_phase = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    peak_value           = corr_map[peak_bin, peak_phase]
    peak_doppler         = doppler_bins[peak_bin]

    # ── Detection metric: Peak-to-mean ratio ──────────────────
    # If satellite is present: ratio >> 1 (typically > 3.0)
    # If satellite absent:     ratio ≈ 1.0 (just noise)
    mean_value = np.mean(corr_map)
    peak_ratio = peak_value / mean_value

    # Detection threshold
    THRESHOLD  = 2.5
    detected   = peak_ratio > THRESHOLD

    results = {
        'sv_id'        : sv_id,
        'detected'     : detected,
        'peak_doppler' : peak_doppler,
        'peak_phase'   : peak_phase,
        'peak_value'   : peak_value,
        'peak_ratio'   : peak_ratio,
        'corr_map'     : corr_map,
        'doppler_bins' : doppler_bins,
    }

    return results


def plot_acquisition(results):
    """
    Plot the 2D acquisition correlation map.
    X-axis: code phase (chips)
    Y-axis: Doppler frequency (Hz)
    Z-axis: correlation power (colour)
    The bright peak is the satellite — this is what acquisition looks like.
    """
    sv_id        = results['sv_id']
    corr_map     = results['corr_map']
    doppler_bins = results['doppler_bins']
    peak_doppler = results['peak_doppler']
    peak_phase   = results['peak_phase']

    # Convert sample index to chips
    samples_per_chip = corr_map.shape[1] / CODE_LENGTH
    phases_chips     = np.arange(corr_map.shape[1]) / samples_per_chip

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'GPS L1 C/A Acquisition — PRN {sv_id}', fontsize=13)

    # ── Left: 2D colour map ───────────────────────────────────
    im = axes[0].imshow(
        corr_map,
        aspect='auto',
        extent=[0, CODE_LENGTH, doppler_bins[-1], doppler_bins[0]],
        cmap='viridis'
    )
    axes[0].set_xlabel('Code phase (chips)')
    axes[0].set_ylabel('Doppler frequency (Hz)')
    axes[0].set_title('2D Acquisition search grid')
    axes[0].axhline(y=peak_doppler, color='red',
                    linewidth=1, linestyle='--', alpha=0.7,
                    label=f'Peak Doppler: {peak_doppler} Hz')
    axes[0].legend(fontsize=9)
    plt.colorbar(im, ax=axes[0], label='Correlation power')

    # ── Right: 1D slice at best Doppler bin ──────────────────
    best_bin = np.where(doppler_bins == peak_doppler)[0][0]
    axes[1].plot(phases_chips, corr_map[best_bin, :],
                 color='#1D9E75', linewidth=1.0)
    axes[1].axvline(x=peak_phase / samples_per_chip,
                    color='red', linewidth=1.5, linestyle='--',
                    label=f'Peak phase: chip {peak_phase // int(samples_per_chip)}')
    axes[1].set_xlabel('Code phase (chips)')
    axes[1].set_ylabel('Correlation power')
    axes[1].set_title(f'Correlation at Doppler = {peak_doppler} Hz')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn{sv_id}_acquisition.png', dpi=150)
    plt.show()
    print(f"Acquisition plot saved → OUT/figures/prn{sv_id}_acquisition.png")


def plot_3d_surface(results):
    """
    3D surface plot of the correlation map.
    This is the classic GNSS textbook visualization —
    the sharp peak rising out of the noise floor.
    """
    sv_id        = results['sv_id']
    corr_map     = results['corr_map']
    doppler_bins = results['doppler_bins']

    samples_per_chip = corr_map.shape[1] / CODE_LENGTH
    phases_chips     = np.arange(corr_map.shape[1]) / samples_per_chip

    # Downsample for 3D plot performance
    step      = 5
    map_ds    = corr_map[:, ::step]
    phases_ds = phases_chips[::step]

    D, P = np.meshgrid(doppler_bins, phases_ds, indexing='ij')

    fig  = plt.figure(figsize=(12, 7))
    ax   = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, D, map_ds,
                           cmap='viridis', alpha=0.9,
                           linewidth=0, antialiased=True)
    ax.set_xlabel('Code phase (chips)', labelpad=10)
    ax.set_ylabel('Doppler (Hz)',       labelpad=10)
    ax.set_zlabel('Correlation power',  labelpad=10)
    ax.set_title(f'3D Acquisition Surface — PRN {sv_id}', fontsize=13)
    fig.colorbar(surf, ax=ax, shrink=0.4, label='Correlation power')

    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn{sv_id}_acquisition_3d.png', dpi=150)
    plt.show()
    print(f"3D surface saved → OUT/figures/prn{sv_id}_acquisition_3d.png")


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load the signal we generated in Day 2
    print("Loading signal from Day 2...")
    signal = np.load('OUT/prn1_rawsignal.npy')
    fs     = 5e6   # must match what signal_simulator used

    # ── Acquire PRN 1 (should be detected) ───────────────────
    SV_TARGET = 1
    print(f"\nAcquiring PRN {SV_TARGET}...")
    results = acquire(signal, sv_id=SV_TARGET, fs=fs)

    print(f"\n{'='*45}")
    print(f"  PRN {results['sv_id']:>2} — "
          f"{'✅ DETECTED' if results['detected'] else '❌ NOT FOUND'}")
    print(f"  Doppler estimate : {results['peak_doppler']:+.0f} Hz")
    print(f"  Code phase       : chip {results['peak_phase']}")
    print(f"  Peak/mean ratio  : {results['peak_ratio']:.2f}  "
          f"(threshold = 2.5)")
    print(f"{'='*45}\n")

    plot_acquisition(results)
    plot_3d_surface(results)

    # ── Try to acquire PRN 5 (not in signal — should fail) ───
    print(f"Acquiring PRN 5 (not in signal — expect NOT FOUND)...")
    results_5 = acquire(signal, sv_id=5, fs=fs)
    print(f"\n  PRN  5 — "
          f"{'✅ DETECTED' if results_5['detected'] else '❌ NOT FOUND'}")
    print(f"  Peak/mean ratio  : {results_5['peak_ratio']:.2f}")
    print(f"\nDone. Check OUT/figures/ for all plots.")