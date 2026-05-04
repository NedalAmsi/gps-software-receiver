# prn_generator.py
# GPS L1 C/A Gold Code Generator
# IS-GPS-200 compliant
# Author: Nedal Amsi

import numpy as np
import matplotlib.pyplot as plt

def generate_G1():
    register = [1] * 10
    output = []
    for _ in range(1023):
        output.append(register[9])
        feedback = register[2] ^ register[9]
        register = [feedback] + register[:-1]
    return np.array(output)

def generate_G2():
    register = [1] * 10
    output = []
    for _ in range(1023):
        output.append(register[9])
        feedback = (register[1] ^ register[2] ^ register[5] ^
                    register[7] ^ register[8] ^ register[9])
        register = [feedback] + register[:-1]
    return np.array(output)

G2_DELAYS = {
    1: 5,   2: 6,   3: 7,   4: 8,   5: 17,  6: 18,  7: 139, 8: 140,
    9: 141, 10: 251,11: 252,12: 254,13: 255,14: 256,15: 257,16: 258,
    17: 469,18: 470,19: 471,20: 472,21: 473,22: 474,23: 509,24: 512,
    25: 513,26: 514,27: 515,28: 516,29: 859,30: 860,31: 861,32: 862
}

def generate_PRN(sv_id):
    if sv_id not in G2_DELAYS:
        raise ValueError(f"SV ID must be between 1 and 32, got {sv_id}")
    g1 = generate_G1()
    g2 = generate_G2()
    delay = G2_DELAYS[sv_id]
    g2_delayed = np.roll(g2, delay)
    gold_code = (g1 ^ g2_delayed)
    prn = 1 - 2 * gold_code
    return prn

def plot_autocorrelation(sv_id):
    prn = generate_PRN(sv_id)
    autocorr = np.correlate(prn, prn, mode='full')
    lags = np.arange(-(len(prn)-1), len(prn))
    plt.figure(figsize=(12, 4))
    plt.plot(lags, autocorr, linewidth=0.8, color='#1D9E75')
    plt.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    plt.title(f'PRN {sv_id} Autocorrelation — GPS L1 C/A Gold Code', fontsize=13)
    plt.xlabel('Lag (chips)')
    plt.ylabel('Correlation value')
    plt.xlim(-100, 100)
    plt.tight_layout()
    plt.savefig(f'OUT/figures/prn_{sv_id}_autocorrelation.png', dpi=150)
    plt.show()
    peak = autocorr[len(prn)-1]
    sidelobes = autocorr[autocorr != peak]
    print(f"PRN {sv_id}: Peak = {peak}, Max sidelobe = {np.max(np.abs(sidelobes))}")

def plot_cross_correlation(sv1, sv2):
    prn1 = generate_PRN(sv1)
    prn2 = generate_PRN(sv2)
    crosscorr = np.correlate(prn1, prn2, mode='full')
    lags = np.arange(-(len(prn1)-1), len(prn1))
    plt.figure(figsize=(12, 4))
    plt.plot(lags, crosscorr, linewidth=0.8, color='#E24B4A')
    plt.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    plt.title(f'Cross-correlation: PRN {sv1} vs PRN {sv2}', fontsize=13)
    plt.xlabel('Lag (chips)')
    plt.ylabel('Correlation value')
    plt.tight_layout()
    plt.savefig(f'OUT/figures/crosscorr_prn{sv1}_prn{sv2}.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    prn1 = generate_PRN(1)
    print(f"PRN 1 — first 10 chips : {prn1[:10]}")
    print(f"PRN 1 — code length    : {len(prn1)} chips")
    print(f"PRN 1 — chip balance   : {np.sum(prn1==1)} ones, {np.sum(prn1==-1)} minus-ones")
    plot_autocorrelation(1)
    plot_cross_correlation(1, 2)
    print("\nAll PRN codes generated successfully.")
