# signal_simulator.py
# GPS L1 C/A Baseband Signal Simulator
# Generates a complex baseband signal (I + jQ)
# This is how real software receivers process GPS signals
# Author: Nedal Amsi

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prn_generator import generate_PRN

CODE_RATE   = 1.023e6
CODE_LENGTH = 1023


def generate_baseband_signal(sv_id,
                              doppler_hz  = 1500,
                              cn0_db_hz   = 45,
                              fs          = 5e6,
                              duration_ms = 200):
    """
    Generate a complex baseband GPS L1 C/A signal.

    At baseband the carrier is removed — only the Doppler offset remains.
    Signal model:  s(t) = prn(t) * exp(j * 2π * doppler * t) + noise

    Args:
        sv_id       : Satellite PRN (1-32)
        doppler_hz  : Doppler frequency offset in Hz
        cn0_db_hz   : Carrier-to-noise density ratio in dB-Hz
        fs          : Sampling frequency in Hz
        duration_ms : Duration in milliseconds

    Returns:
        t      : time vector (s)
        signal : complex baseband signal (I + jQ)
        clean  : noise-free signal
    """
    num_samples      = int(fs * duration_ms / 1000)
    t                = np.arange(num_samples) / fs
    samples_per_chip = fs / CODE_RATE

    # Generate and upsample PRN
    prn = generate_PRN(sv_id)
    prn_up = np.array([
        prn[int(i / samples_per_chip) % CODE_LENGTH]
        for i in range(num_samples)
    ], dtype=np.float64)

    # Doppler carrier (baseband — no L1 frequency)
    doppler_carrier = np.exp(1j * 2 * np.pi * doppler_hz * t)

    # Clean complex baseband signal
    clean = prn_up * doppler_carrier

    # AWGN noise (complex)
    cn0_linear  = 10 ** (cn0_db_hz / 10)
    noise_power = CODE_RATE / cn0_linear
    noise_std   = np.sqrt(noise_power / 2)
    noise       = noise_std * (np.random.randn(num_samples)
                             + 1j * np.random.randn(num_samples))

    signal = clean + noise
    return t, signal, clean


def plot_signal(t, signal, clean, sv_id):
    samples_per_chip = len(t) / (CODE_RATE * t[-1])
    n = int(10 * len(t) / (CODE_RATE * t[-1]))  # first 10 chips

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f'GPS L1 C/A Baseband Signal — PRN {sv_id}', fontsize=13)

    axes[0].plot(t[:n]*1e6, np.real(clean[:n]),
                 color='#1D9E75', linewidth=1.2, label='I (clean)')
    axes[0].plot(t[:n]*1e6, np.imag(clean[:n]),
                 color='#185FA5', linewidth=1.2, label='Q (clean)',
                 alpha=0.7)
    axes[0].set_title('Clean baseband signal — first 10 chips')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[:n]*1e6, np.real(signal[:n]),
                 color='#E24B4A', linewidth=0.6,
                 alpha=0.7, label='I (with noise)')
    axes[1].set_title('Received signal — noise obscures the GPS signal')
    axes[1].set_xlabel('Time (μs)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn{sv_id}_signal_timedomain.png', dpi=150)
    print(f"Saved → OUT/figures/prn{sv_id}_signal_timedomain.png")


if __name__ == "__main__":
    SV_ID      = 1
    DOPPLER    = 1500
    CN0        = 45
    FS         = 5e6
    DURATION   = 200   # 200 ms — enough for tracking

    print(f"Generating {DURATION}ms baseband GPS signal for PRN {SV_ID}...")
    print(f"  Doppler : {DOPPLER} Hz  |  C/N0 : {CN0} dB-Hz  |  fs : {FS/1e6} MHz")

    t, signal, clean = generate_baseband_signal(
        sv_id       = SV_ID,
        doppler_hz  = DOPPLER,
        cn0_db_hz   = CN0,
        fs          = FS,
        duration_ms = DURATION
    )

    print(f"  Samples : {len(signal)} = {len(signal)/FS*1000:.0f} ms ✓")

    plot_signal(t, signal, clean, SV_ID)

    np.save('OUT/prn1_rawsignal.npy', signal)
    np.save('OUT/time_vector.npy', t)
    print(f"  Saved   : OUT/prn1_rawsignal.npy")
    print(f"  Ready for acquisition and tracking.")
