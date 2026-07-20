import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from adaptspec import adaptspec, summarise

# ── CHANGE THIS ───────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_260514\global_platform_data\share_price\USA"
FILE     = "_AAPL_price.csv"
# ─────────────────────────────────────────────────────────────────────────────

ticker = FILE.replace("_price.csv", "").lstrip("_")
df = pd.read_csv(os.path.join(DATA_DIR, FILE), parse_dates=["Date"])
df = df.sort_values("Date").dropna(subset=["Price"])
df = df[df["Date"] >= pd.Timestamp("2005-01-01")].reset_index(drop=True)

log_returns = np.diff(np.log(df["Price"].values))
dates = df["Date"].values[1:]

result  = adaptspec(log_returns, nexp_max=20, nbeta=20, niter=10000, nburn=5000, tmin=120, seed=42, verbose=True)
summary = summarise(result)

changepoints = [b[1] for b in summary["segment_boundaries"][:-1]]
cp_dates     = [pd.Timestamp(dates[cp]).date() for cp in changepoints]

# ── PLOT 1: log returns + changepoint proba ───────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax1.plot(dates, log_returns, lw=0.7, color="#2c3e50")
for cp, cpd in zip(changepoints, cp_dates):
    ax1.axvline(pd.Timestamp(dates[cp]), color="red", lw=1.5, ls="--")
    ax1.text(pd.Timestamp(dates[cp]), ax1.get_ylim()[1], str(cpd),
             rotation=90, fontsize=7, color="red", va="top")
ax1.set_title(f"{ticker}  |  modal segments={summary['modal_nexp']}  mean={summary['mean_nexp']:.2f}")
ax1.set_ylabel("Log return")

ax2.fill_between(dates, summary["changepoint_proba"], color="#e74c3c", alpha=0.6)
ax2.axhline(0.05, color="grey", lw=0.8, ls=":")
for cp in changepoints:
    ax2.axvline(pd.Timestamp(dates[cp]), color="red", lw=1.5, ls="--", alpha=0.6)
ax2.set_ylabel("P(changepoint)")

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, f"{ticker}_returns.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── PLOT 2: time-varying power spectrum surface ───────────────────────────────
tvpsd = summary["tvpsd"]
freq  = summary["freq_hat"]
time  = np.linspace(pd.Timestamp(dates[0]).year + pd.Timestamp(dates[0]).month/12,
                    pd.Timestamp(dates[-1]).year + pd.Timestamp(dates[-1]).month/12,
                    len(dates))

# Downsample time axis for surface plot (every 10th point)
step = 10
T = time[::step]
Z = tvpsd[::step, :]
X, Y = np.meshgrid(freq, T)

fig = plt.figure(figsize=(14, 8))
ax  = fig.add_subplot(111, projection="3d")
ax.view_init(35, 310)
ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0.2, antialiased=True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Time")
ax.set_zlabel("Log PSD")
ax.set_title(f"{ticker} — Time-Varying Power Spectrum")
ax.yaxis.set_major_locator(plt.MaxNLocator(4))

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, f"{ticker}_tvpsd.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── TABLE ─────────────────────────────────────────────────────────────────────
rows = [{"changepoint": i+1, "date": cpd, "obs_index": cp,
         "P(cp)": round(summary["changepoint_proba"][cp], 3)}
        for i, (cp, cpd) in enumerate(zip(changepoints, cp_dates))]
table = pd.DataFrame(rows)
print(f"\n--- {ticker} ---")
print(table.to_string(index=False))
print(f"\nnexp distribution: {summary['nexp_probs']}")
table.to_csv(os.path.join(DATA_DIR, f"{ticker}_changepoints.csv"), index=False)