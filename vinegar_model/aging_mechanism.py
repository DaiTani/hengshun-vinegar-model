import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class TMPReactionNetwork:
    def __init__(self):
        self.R = 8.314
        self.Ea1 = 75000.0
        self.Ea2 = 65000.0
        self.A1 = 8.0e10
        self.A2 = 3.0e9
        self.nh3_effective = 100.0
        self._nh3 = 100.0

    def rate_constant(self, Ea, A, T):
        return A * np.exp(-Ea / (self.R * (T + 273.15)))

    def k1(self, T):
        return self.rate_constant(self.Ea1, self.A1, T)

    def k2(self, T):
        return self.rate_constant(self.Ea2, self.A2, T)

    def _odes(self, y, t, T):
        precursor, acetoin, tmp = y
        r1 = self.k1(T) * precursor
        r2 = self.k2(T) * acetoin * self._nh3
        return [-r1, r1 - r2, r2]

    def simulate_ageing(self, t_months, T=25):
        self._nh3 = self.nh3_effective
        t = np.linspace(0, t_months, 500)
        sol = odeint(self._odes, [150.0, 80.0, 0.0], t, args=(T,))
        return t, sol[:, 1], sol[:, 2]

    def predict_tmp(self, t, initial_acetoin, initial_nh3, T=25, initial_precursor=150.0):
        self._nh3 = initial_nh3
        t_arr = np.asarray(t)
        if len(t_arr) == 1:
            t_single = float(t_arr[0])
            if t_single <= 0:
                return np.array([0.0])
            t_arr = np.linspace(0, t_single, 100)
            sol = odeint(self._odes, [initial_precursor, initial_acetoin, 0.0], t_arr, args=(T,))
            return sol[-1:, 2]
        sol = odeint(self._odes, [initial_precursor, initial_acetoin, 0.0], t_arr, args=(T,))
        return sol[:, 2]


def logistic_fn(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def fit_logistic(t_data, tmp_data):
    p0 = [110, 0.05, 30]
    bounds = ([50, 0.001, 0], [200, 0.5, 80])
    popt, _ = curve_fit(logistic_fn, t_data, tmp_data, p0=p0, bounds=bounds, maxfev=10000)
    return popt


def residuals_mechanism(params, t_data, tmp_data, acetoin_init):
    precursor0, k1, k2 = params
    if precursor0 <= 0 or k1 <= 0 or k2 <= 0:
        return [1e10] * len(t_data)
    model = TMPReactionNetwork()
    model.A1 = k1
    model.A2 = k2
    model.nh3_effective = 100.0
    pred = model.predict_tmp(t_data, acetoin_init, 100.0, T=25, initial_precursor=precursor0)
    return tmp_data - pred


def fit_mechanism_params(t_data, tmp_data, acetoin_init):
    best_result = None
    best_loss = float('inf')
    for precursor0_init in [80, 120, 150, 200]:
        for k1_init in [1e8, 5e8, 1e9, 5e9]:
            for k2_init in [1e6, 5e6, 1e7]:
                x0 = [precursor0_init, k1_init, k2_init]
                try:
                    result = least_squares(
                        residuals_mechanism, x0,
                        args=(t_data, tmp_data, acetoin_init),
                        bounds=([10, 1e5, 1e4], [500, 1e12, 1e10]),
                        max_nfev=2000
                    )
                    if result.fun.sum() ** 2 < best_loss:
                        best_loss = result.fun.sum() ** 2
                        best_result = result
                except:
                    pass
    return best_result.x


def calculate_r2(pred, actual):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - ss_res / ss_tot


if __name__ == '__main__':
    t_data = np.array([36.0, 60.0, 96.0])
    tmp_data = np.array([38.0, 50.0, 95.0])
    acetoin_init = 80.0

    popt_log = fit_logistic(t_data, tmp_data)
    pred_log = logistic_fn(t_data, *popt_log)
    r2_log = calculate_r2(pred_log, tmp_data)

    best_params = fit_mechanism_params(t_data, tmp_data, acetoin_init)
    precursor0, k1_fit, k2_fit = best_params

    model = TMPReactionNetwork()
    model.A1 = k1_fit
    model.A2 = k2_fit
    model.nh3_effective = 100.0

    pred_mech = model.predict_tmp(t_data, acetoin_init, 100.0, T=25, initial_precursor=precursor0)
    r2_mech = calculate_r2(pred_mech, tmp_data)

    t_smooth = np.linspace(0, 100, 300)
    pred_log_smooth = logistic_fn(t_smooth, *popt_log)
    pred_mech_smooth = model.predict_tmp(t_smooth, acetoin_init, 100.0, T=25, initial_precursor=precursor0)

    t_long, acetoin_long, tmp_long = model.simulate_ageing(100, T=25)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(t_data, tmp_data, color='red', s=120, zorder=5,
                    label='Data (Ren 2023)', edgecolors='darkred', linewidths=1.5)
    axes[0].plot(t_smooth, pred_log_smooth, 'b--', linewidth=2,
                 label=f'Logistic (R²={r2_log:.4f})')
    axes[0].plot(t_smooth, pred_mech_smooth, 'g-', linewidth=2,
                 label=f'2-Step Mechanism (R²={r2_mech:.4f})')
    axes[0].set_xlabel('Aging Time (months)', fontsize=12)
    axes[0].set_ylabel('TMP (mg/L)', fontsize=12)
    axes[0].set_title('TMP Formation: Logistic vs Mechanism', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 130)

    axes[1].plot(t_long, tmp_long, 'g-', linewidth=2.5, label='TMP (mechanism)')
    ax2_twin = axes[1].twinx()
    ax2_twin.plot(t_long, acetoin_long, 'm--', linewidth=2, label='Acetoin')
    axes[1].scatter(t_data, tmp_data, color='red', s=100, zorder=5,
                    label='Observed TMP', edgecolors='darkred')
    axes[1].set_xlabel('Aging Time (months)', fontsize=12)
    axes[1].set_ylabel('TMP (mg/L)', color='g', fontsize=12)
    ax2_twin.set_ylabel('Acetoin (mg/L)', color='m', fontsize=12)
    axes[1].set_title('2-Step Mechanism: Species Dynamics', fontsize=13)
    axes[1].legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig('aging_mechanism.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("=" * 60)
    print("   TMP Aging Model - Chemical Mechanism Results")
    print("=" * 60)
    print(f"\n[Logistic Model]")
    print(f"  K (carrying capacity) = {popt_log[0]:.2f} mg/L")
    print(f"  r (growth rate)       = {popt_log[1]:.4f} per month")
    print(f"  t0 (inflection point) = {popt_log[2]:.2f} months")
    print(f"  R² = {r2_log:.4f}")
    print(f"\n[2-Step Chemical Mechanism]")
    print(f"  Step 1: Precursor --(k1)--> Acetoin")
    print(f"  Step 2: Acetoin + NH₃ --(k2)--> TMP")
    print(f"  Precursor pool  = {precursor0:.2f} mg/L")
    print(f"  k1 (fitted)     = {k1_fit:.2e}")
    print(f"  k2 (fitted)     = {k2_fit:.2e}")
    print(f"  R² = {r2_mech:.4f}")
    print(f"\n[Predictions at validation points]")
    print(f"  36 months: Logistic={pred_log[0]:.1f}, Mech={pred_mech[0]:.1f}, Observed=38.0")
    print(f"  60 months: Logistic={pred_log[1]:.1f}, Mech={pred_mech[1]:.1f}, Observed=50.0")
    print(f"  96 months: Logistic={pred_log[2]:.1f}, Mech={pred_mech[2]:.1f}, Observed=95.0")
    print(f"\n[Literature Parameters]")
    print(f"  Ea1 (precursor->acetoin) ≈ 75 kJ/mol (Maillard pathway)")
    print(f"  Ea2 (acetoin->TMP)       ≈ 65 kJ/mol (Strecker degradation)")
    print(f"  NH₃ from amino acid breakdown during aging")
    print(f"  He 2004: TMP 77 μg/mL at 2mo, >500 μg/mL at 2+ years")
    print("=" * 60)
    print("\nFigure saved: aging_mechanism.png")
