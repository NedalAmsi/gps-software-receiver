# position_solver.py
# GPS Pseudorange Position Solver — Weighted Least Squares
# Author: Nedal Amsi

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c = 299792458.0   # speed of light (m/s)


def ecef_to_geodetic(x, y, z):
    a  = 6378137.0
    f  = 1 / 298.257223563
    b  = a * (1 - f)
    e2 = 1 - (b/a)**2
    p  = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        N   = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)
    N   = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    lon = np.arctan2(y, x)
    alt = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), alt


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0):
    a  = 6378137.0
    f  = 1 / 298.257223563
    e2 = 2*f - f**2
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N   = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    X   = (N + alt_m) * np.cos(lat) * np.cos(lon)
    Y   = (N + alt_m) * np.cos(lat) * np.sin(lon)
    Z   = (N*(1-e2) + alt_m) * np.sin(lat)
    return np.array([X, Y, Z])


def solve_position(sv_xyz, pseudoranges, weights=None, max_iter=20, tol=1e-3):
    """
    Weighted least-squares GPS position solver.

    Pseudorange equation: ρᵢ = ‖xᵤ − xˢᵢ‖ + c·δt

    Linearised design matrix:
        H[i, :3] = (xᵤ − xˢᵢ) / ‖xᵤ − xˢᵢ‖   (unit vector, SV → user)
        H[i,  3] = 1                              (clock bias)

    State vector: [X, Y, Z, c·δt]
    """
    N = len(pseudoranges)
    W = np.diag(weights if weights is not None else np.ones(N))

    # Start on Earth's surface beneath satellite centroid
    sv_c = sv_xyz.mean(axis=0)
    pos  = np.append(6371000.0 * sv_c / np.linalg.norm(sv_c), 0.0)

    for _ in range(max_iter):
        # Vectors and ranges: satellite → user
        diff = pos[:3] - sv_xyz          # (N, 3)  user minus satellite
        rng  = np.linalg.norm(diff, axis=1)  # (N,)

        # ── Design matrix (CORRECT signs) ────────────────────
        H        = np.zeros((N, 4))
        H[:, :3] = diff / rng[:, np.newaxis]  # unit vectors SV→user
        H[:,  3] = 1.0                          # clock column

        # Pseudorange residuals
        dp   = pseudoranges - (rng + pos[3])

        # WLS update
        HTW  = H.T @ W
        HTWH = HTW @ H
        dx   = np.linalg.solve(HTWH, HTW @ dp)
        pos += dx

        if np.linalg.norm(dx[:3]) < tol:
            break

    # DOP from final HTWH
    try:
        Q    = np.linalg.inv(HTWH)
        GDOP = float(np.sqrt(abs(np.trace(Q))))
        PDOP = float(np.sqrt(abs(Q[0,0] + Q[1,1] + Q[2,2])))
        TDOP = float(np.sqrt(abs(Q[3,3])))
    except Exception:
        GDOP = PDOP = TDOP = float('nan')

    # Post-fit residuals at final position
    diff_f = pos[:3] - sv_xyz
    rng_f  = np.linalg.norm(diff_f, axis=1)
    resid  = pseudoranges - (rng_f + pos[3])

    dop = {'GDOP': GDOP, 'PDOP': PDOP, 'TDOP': TDOP}
    return pos[:3], pos[3], dop, resid


def simulate_scenario():
    """
    Simulate GPS positioning at ESTEC, Noordwijk, Netherlands.
    Satellite positions chosen for good geometric coverage over Europe.
    """
    # ── True position — ESTEC, ESA Technical Centre ──────────
    TRUE_LAT = 52.2185    # °N
    TRUE_LON =  4.4199    # °E  (positive = East)
    TRUE_ALT = 10.0       # m
    true_ecef = geodetic_to_ecef(TRUE_LAT, TRUE_LON, TRUE_ALT)

    print(f"  Location       : ESTEC, Noordwijk, Netherlands (ESA)")
    print(f"  True position  : {TRUE_LAT:.4f}°N, {TRUE_LON:.4f}°E, "
          f"{TRUE_ALT:.0f}m")
    print(f"  True ECEF (km) : X={true_ecef[0]/1e3:.1f}, "
          f"Y={true_ecef[1]/1e3:.1f}, Z={true_ecef[2]/1e3:.1f}")

    # ── Satellite ECEF positions (m) — real GPS orbital altitudes ─
    # Selected for good European sky coverage (elevation > 15°)
    sv_xyz = np.array([
        [ 20184919,  16278768,   3843784],   # SV  2
        [ 23218050,   8206905,  11241062],   # SV  5
        [-11253792,  21122753,  12318978],   # SV  9
        [  6591694, -25325011,   3494392],   # SV 15
        [-16927618,  15052909,  16134932],   # SV 21
        [ 15178769, -21028659,  10353695],   # SV 24
    ], dtype=np.float64)

    N = len(sv_xyz)

    # Verify orbital altitudes (~26,560 km from Earth centre)
    altitudes = np.linalg.norm(sv_xyz, axis=1) / 1e3
    print(f"\n  SV orbital altitudes: "
          f"{', '.join(f'{a:.0f}km' for a in altitudes)}")

    # ── Compute true geometric ranges ────────────────────────
    true_ranges = np.linalg.norm(true_ecef - sv_xyz, axis=1)
    print(f"  True ranges (km): "
          f"{', '.join(f'{r/1e3:.0f}' for r in true_ranges)}")

    # ── Add realistic errors ──────────────────────────────────
    np.random.seed(7)
    clock_bias  = 45000.0                       # 150 μs clock bias (m)
    iono        = np.random.uniform(2, 8, N)    # ionospheric (m)
    tropo       = np.random.uniform(1, 3, N)    # tropospheric (m)
    noise       = np.random.randn(N) * 1.5      # receiver noise (m)
    iono_corr   = iono * 0.75                   # 75% iono correction

    pseudoranges = (true_ranges + clock_bias
                    + iono - iono_corr
                    + tropo + noise)

    # Elevation-dependent weights
    elevations = np.array([62, 78, 45, 38, 55, 41], dtype=float)
    weights    = np.sin(np.radians(elevations))**2

    print(f"\n  Clock bias     : {clock_bias:.0f} m "
          f"({clock_bias/c*1e6:.1f} μs)")
    print(f"  Iono errors    : {iono.mean():.1f} m avg "
          f"→ {(iono-iono_corr).mean():.1f} m residual after correction")
    print(f"  Tropo errors   : {tropo.mean():.1f} m avg (uncorrected)")
    print(f"  Noise          : ±{noise.std():.1f} m (1σ)")

    return (sv_xyz, pseudoranges, weights,
            true_ecef, TRUE_LAT, TRUE_LON, TRUE_ALT)


def plot_results(true_ecef, solved_ecef, residuals, dop,
                 true_lat, true_lon, sol_lat, sol_lon):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'GPS Position Solver — Weighted Least Squares\n'
        'ESTEC, Noordwijk, Netherlands (ESA)', fontsize=13)

    # ── Position errors ───────────────────────────────────────
    e3d = np.linalg.norm(solved_ecef - true_ecef)
    eh  = np.linalg.norm((solved_ecef - true_ecef)[:2])
    ev  = abs(solved_ecef[2] - true_ecef[2])

    bars = axes[0].bar(['3D', 'Horizontal', 'Vertical'],
                       [e3d, eh, ev],
                       color=['#E24B4A', '#EF9F27', '#185FA5'],
                       width=0.5, edgecolor='white')
    for bar, v in zip(bars, [e3d, eh, ev]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     v + 0.3, f'{v:.1f} m',
                     ha='center', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Error (metres)')
    axes[0].set_title('Position error')
    axes[0].set_ylim(0, max(e3d, eh, ev) * 1.5)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ── DOP ───────────────────────────────────────────────────
    dv = [dop['GDOP'], dop['PDOP'], dop['TDOP']]
    b2 = axes[1].bar(['GDOP', 'PDOP', 'TDOP'], dv,
                     color=['#1D9E75', '#185FA5', '#534AB7'],
                     width=0.5, edgecolor='white')
    for bar, v in zip(b2, dv):
        if not np.isnan(v):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         v + 0.02, f'{v:.2f}',
                         ha='center', fontsize=11)
    axes[1].axhline(2.0, color='red', lw=1.0, ls='--',
                    label='DOP=2 (excellent)')
    axes[1].set_ylabel('DOP value')
    axes[1].set_title('Dilution of Precision')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    # ── Post-fit residuals ────────────────────────────────────
    sv_ids  = [f'SV{i+1}' for i in range(len(residuals))]
    colors3 = ['#1D9E75' if abs(r) < 5 else '#E24B4A'
               for r in residuals]
    axes[2].bar(sv_ids, residuals,
                color=colors3, width=0.5, edgecolor='white')
    axes[2].axhline(0, color='gray', lw=0.8, ls='--')
    axes[2].axhline( 5, color='orange', lw=0.8, ls=':', alpha=0.7)
    axes[2].axhline(-5, color='orange', lw=0.8, ls=':', alpha=0.7,
                    label='±5 m threshold')
    axes[2].set_ylabel('Residual (m)')
    axes[2].set_title('Post-fit pseudorange residuals')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('OUT/figures/position_solution.png', dpi=150)
    print("Saved → OUT/figures/position_solution.png")


if __name__ == "__main__":

    print("="*55)
    print("  GPS Weighted Least-Squares Position Solver")
    print("="*55 + "\n")

    (sv_xyz, pseudoranges, weights,
     true_ecef, true_lat, true_lon,
     true_alt) = simulate_scenario()

    print("\nRunning weighted least-squares solver...")
    solved_ecef, cdt, dop, residuals = solve_position(
        sv_xyz, pseudoranges, weights=weights)

    sol_lat, sol_lon, sol_alt = ecef_to_geodetic(*solved_ecef)
    e3d = np.linalg.norm(solved_ecef - true_ecef)

    print(f"\n{'='*55}")
    print(f"  SOLUTION")
    print(f"{'='*55}")
    print(f"  Solved  : {sol_lat:.6f}°N  {sol_lon:.6f}°E  "
          f"{sol_alt:.1f}m")
    print(f"  True    : {true_lat:.6f}°N  {true_lon:.6f}°E  "
          f"{true_alt:.1f}m")
    print(f"  Lat error : {(sol_lat-true_lat)*111320:.1f} m")
    print(f"  Lon error : "
          f"{(sol_lon-true_lon)*111320*np.cos(np.radians(true_lat)):.1f} m")
    print(f"  3D error  : {e3d:.2f} m")
    print(f"  Clock bias: {cdt/c*1e6:.2f} μs  "
          f"(true: {45000/c*1e6:.2f} μs)")
    print(f"  GDOP: {dop['GDOP']:.2f}  "
          f"PDOP: {dop['PDOP']:.2f}  "
          f"TDOP: {dop['TDOP']:.2f}")
    print(f"  Post-fit residuals (m): {np.round(residuals, 2)}")
    print(f"{'='*55}")

    plot_results(true_ecef, solved_ecef, residuals, dop,
                 true_lat, true_lon, sol_lat, sol_lon)
    print("\n✅ Position solved. Day 5 complete.")
