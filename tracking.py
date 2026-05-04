# tracking.py
# GPS L1 C/A DLL + PLL Tracking — Borre formulation
# Reference: Borre et al. "A Software-Defined GPS and Galileo Receiver"
# Author: Nedal Amsi

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prn_generator import generate_PRN

CODE_RATE   = 1.023e6
CODE_LENGTH = 1023


def design_loop(bandwidth_hz, zeta=0.7071):
    """
    Compute tau1 and tau2 for a 2nd order loop filter.
    Returns (tau1, tau2) in seconds.
    """
    wn   = bandwidth_hz * 8 * zeta / (4 * zeta**2 + 1)
    tau1 = 1.0 / (wn ** 2)
    tau2 = 2.0 * zeta / wn
    return tau1, tau2


def track(signal, sv_id, fs,
          init_doppler    = 1500.0,
          init_code_phase = 0,
          num_ms          = 200):

    T                = 1e-3          # integration time (s)
    samples_per_ms   = int(fs * T)
    samples_per_chip = fs / CODE_RATE
    early_late_space = 0.5           # chips

    prn = generate_PRN(sv_id)

    def get_prn(phase_chips):
        idx = (np.arange(samples_per_ms) / samples_per_chip
               + phase_chips) % CODE_LENGTH
        return (1 - 2 * prn[idx.astype(int)]).astype(np.float64)

    # ── Loop filter design (Borre) ────────────────────────────
    tau1_pll, tau2_pll = design_loop(bandwidth_hz=15.0)
    tau1_dll, tau2_dll = design_loop(bandwidth_hz=1.0)

    # ── Initial state ─────────────────────────────────────────
    carr_freq    = float(init_doppler)   # Hz — our Doppler estimate
    carr_nco     = 0.0                   # carrier NCO accumulator (Hz)
    code_phase   = float(init_code_phase) / samples_per_chip
    code_nco     = 0.0                   # code NCO accumulator
    carrier_phase = 0.0                  # radians

    old_pll_disc = 0.0
    old_dll_disc = 0.0

    store = {k: [] for k in
             ['epochs','I_P','Q_P','dll_disc','pll_disc',
              'carr_freq','code_phase','cn0']}

    print(f"  Tracking PRN {sv_id} — {num_ms} ms | "
          f"PLL BW=15Hz | DLL BW=1Hz | "
          f"init Doppler={init_doppler:.0f} Hz")

    for ms in range(num_ms):
        s = ms * samples_per_ms
        e = s + samples_per_ms
        if e > len(signal):
            print(f"  Signal ended at {ms} ms")
            break

        chunk = signal[s:e]

        # ── Carrier wipe-off ──────────────────────────────────
        t_ms     = np.arange(samples_per_ms) / fs
        wipeoff  = np.exp(-1j * (2*np.pi * carr_freq * t_ms
                                 + carrier_phase))
        bb       = chunk * wipeoff

        # ── Correlators ───────────────────────────────────────
        def corr(replica):
            c = np.dot(bb, replica) / samples_per_ms
            return float(np.real(c)), float(np.imag(c))

        IE, QE = corr(get_prn(code_phase - early_late_space))
        IP, QP = corr(get_prn(code_phase))
        IL, QL = corr(get_prn(code_phase + early_late_space))

        # ── PLL Costas discriminator (cycles) ─────────────────
        pll_disc = np.arctan2(QP, IP) / (2 * np.pi)

        # ── PLL loop filter — Borre bilinear IIR ─────────────
        # carr_nco accumulates frequency correction in Hz
        carr_nco += ((tau2_pll / tau1_pll) * (pll_disc - old_pll_disc)
                     + (T / tau1_pll) * pll_disc)
        old_pll_disc  = pll_disc
        carr_freq     = init_doppler + carr_nco   # Hz

        # Update carrier phase for next epoch
        carrier_phase += 2 * np.pi * carr_freq * T

        # ── DLL normalised early-minus-late discriminator ─────
        E_pow    = np.sqrt(IE**2 + QE**2)
        L_pow    = np.sqrt(IL**2 + QL**2)
        denom    = E_pow + L_pow
        dll_disc = (E_pow - L_pow) / (2 * denom) if denom > 1e-12 else 0.0

        # ── DLL loop filter — Borre bilinear IIR ─────────────
        code_nco += ((tau2_dll / tau1_dll) * (dll_disc - old_dll_disc)
                     + (T / tau1_dll) * dll_disc)
        old_dll_disc = dll_disc
        code_phase  -= code_nco * T   # chips

        # ── C/N0 estimate ─────────────────────────────────────
        # Beaulieu C/N0 estimator
        sig_est = IP + 1j*QP
        P_s = IP**2 + QP**2
        P_n = float(np.mean(np.abs(bb - sig_est)**2))
        cn0 = 10 * np.log10(max(P_s / (P_n * T + 1e-12), 1.0))

        store['epochs'].append(ms)
        store['I_P'].append(IP)
        store['Q_P'].append(QP)
        store['dll_disc'].append(dll_disc)
        store['pll_disc'].append(pll_disc)
        store['carr_freq'].append(carr_freq)
        store['code_phase'].append(code_phase)
        store['cn0'].append(cn0)

    n          = len(store['epochs'])
    f_final    = store['carr_freq'][-1]
    dll_final  = np.mean(np.abs(store['dll_disc'][-20:]))
    cn0_mean   = np.mean(store['cn0'])

    print(f"  Processed         : {n} epochs")
    print(f"  Final Doppler     : {f_final:.2f} Hz  "
          f"(true: {init_doppler:.0f}, error: {f_final-init_doppler:.2f} Hz)")
    print(f"  DLL (last 20ms)   : {dll_final:.5f} chips")
    print(f"  Mean C/N0         : {cn0_mean:.1f} dB-Hz  (true: 45)")

    r          = {k: np.array(v) for k, v in store.items()}
    r['sv_id'] = sv_id
    return r


def plot_tracking(r):
    sv = r['sv_id']
    ms = r['epochs']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'DLL + PLL Tracking — PRN {sv} (Borre 2nd order)',
                 fontsize=14)

    # I/Q constellation
    axes[0,0].scatter(r['I_P'], r['Q_P'], s=8, alpha=0.6, color='#1D9E75')
    axes[0,0].axhline(0, color='gray', lw=0.5)
    axes[0,0].axvline(0, color='gray', lw=0.5)
    axes[0,0].set_xlabel('I'); axes[0,0].set_ylabel('Q')
    axes[0,0].set_title('I/Q Constellation\n(locked = clusters on I-axis ±1)')
    axes[0,0].grid(True, alpha=0.3)

    # I_P
    axes[0,1].plot(ms, r['I_P'], color='#185FA5', lw=0.8)
    axes[0,1].axhline(0, color='gray', lw=0.5, ls='--')
    axes[0,1].set_xlabel('Time (ms)'); axes[0,1].set_ylabel('Amplitude')
    axes[0,1].set_title('Prompt I — (locked = flat ±1, not oscillating)')
    axes[0,1].grid(True, alpha=0.3)

    # DLL
    axes[1,0].plot(ms, r['dll_disc'], color='#E24B4A', lw=0.8)
    axes[1,0].axhline(0, color='gray', lw=0.8, ls='--')
    axes[1,0].set_xlabel('Time (ms)'); axes[1,0].set_ylabel('chips')
    axes[1,0].set_title('DLL discriminator → 0 when locked')
    axes[1,0].grid(True, alpha=0.3)

    # PLL
    axes[1,1].plot(ms, r['pll_disc'], color='#EF9F27', lw=0.8)
    axes[1,1].axhline(0, color='gray', lw=0.8, ls='--')
    axes[1,1].set_xlabel('Time (ms)'); axes[1,1].set_ylabel('cycles')
    axes[1,1].set_title('PLL Costas → 0 when locked')
    axes[1,1].grid(True, alpha=0.3)

    # Doppler
    axes[2,0].plot(ms, r['carr_freq'], color='#534AB7', lw=1.0)
    axes[2,0].axhline(1500, color='red', lw=1.0, ls='--',
                      label='True: 1500 Hz')
    axes[2,0].set_xlabel('Time (ms)'); axes[2,0].set_ylabel('Hz')
    axes[2,0].set_title('Doppler estimate (should converge to 1500 Hz)')
    axes[2,0].legend(fontsize=9); axes[2,0].grid(True, alpha=0.3)

    # C/N0
    axes[2,1].plot(ms, r['cn0'], color='#1D9E75', lw=0.8)
    axes[2,1].axhline(45, color='red', lw=1.0, ls='--',
                      label='True: 45 dB-Hz')
    axes[2,1].set_xlabel('Time (ms)'); axes[2,1].set_ylabel('dB-Hz')
    axes[2,1].set_title('C/N0 estimate')
    axes[2,1].legend(fontsize=9); axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn{sv}_tracking.png', dpi=150)
    print(f"Saved → OUT/figures/prn{sv}_tracking.png")


if __name__ == "__main__":
    signal = np.load('OUT/prn1_rawsignal.npy')
    fs     = 5e6
    print(f"Signal: {len(signal)} samples = {len(signal)/fs*1000:.0f} ms\n")

    r = track(signal, sv_id=1, fs=fs,
              init_doppler=1500.0,
              init_code_phase=0,
              num_ms=200)

    plot_tracking(r)
    np.save('OUT/tracking_results.npy', r)
    print("\nDone.")
