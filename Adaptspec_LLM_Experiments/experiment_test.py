import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adaptspec import adaptspec, summarise

# ── 1. Simulate piecewise AR series ─────────────────────────────────────────
rng = np.random.default_rng(42)
n = 500

def ar1(phi, sigma, n, rng):
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2))
    for t in range(1, n):
        x[t] = phi * x[t-1] + rng.normal(0, sigma)
    return x

seg1 = ar1(phi= 0.90, sigma=1.0, n=n, rng=rng)   # strong positive persistence
seg2 = ar1(phi=-0.85, sigma=1.0, n=n, rng=rng)   # fast oscillation
seg3 = ar1(phi= 0.00, sigma=0.3, n=n, rng=rng)   # low-vol white noise

x = np.concatenate([seg1, seg2, seg3])
true_cps = [500, 1000]
print(f"Series length : {len(x)}")
print(f"True CPs      : {true_cps}")

# ── 2. Run AdaptSPEC ─────────────────────────────────────────────────────────
result = adaptspec(
    x,
    nexp_max = 20,
    nbeta    = 6,
    niter    = 5000,
    nburn    = 2000,
    tmin     = 90,
    seed     = 42,
    verbose  = True,
)

# ── 3. Summarise ─────────────────────────────────────────────────────────────
s = summarise(result)

print(f"\nPosterior mean segments : {s['mean_nexp']:.2f}")
print(f"Modal nexp              : {s['modal_nexp']}")
print(f"nexp distribution       : {s['nexp_probs']}")
print(f"Modal boundaries        : {s['segment_boundaries']}")

detected_cps = [b[1] for b in s['segment_boundaries'][:-1]]
print(f"\nDetected change-points  : {detected_cps}")
print(f"True change-points      : {true_cps}")

cp = s['changepoint_proba']
print("\nTop change-point probabilities:")
top = sorted(range(len(cp)), key=lambda t: -cp[t])[:8]
for t in sorted(top):
    print(f"  t={t:5d}   P(cp) = {cp[t]:.3f}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
colours = ["#d6eaf8", "#d5f5e3", "#fef9e7"]
labels  = ["AR(1) φ=+0.90, σ=1.0", "AR(1) φ=−0.85, σ=1.0", "White noise σ=0.3"]

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

# Panel 1 — time series
ax1 = fig.add_subplot(gs[0:2])
ax1.plot(x, lw=0.6, color="#2c3e50", alpha=0.9)

for i, (s0, e0) in enumerate(zip([0] + true_cps, true_cps + [len(x)])):
    ax1.axvspan(s0, e0, alpha=0.2, color=colours[i], label=labels[i])

for cp_t in true_cps:
    ax1.axvline(cp_t, color="#e74c3c", lw=1.5, ls="--", label="True CP" if cp_t == true_cps[0] else "")

for cp_t in detected_cps:
    ax1.axvline(cp_t, color="#2980b9", lw=1.8, ls="-", label="Detected CP" if cp_t == detected_cps[0] else "")

ax1.set_title(f"Piecewise AR  |  True CPs: {true_cps}  |  Detected: {detected_cps}", fontsize=11)
ax1.set_ylabel("Value")
ax1.legend(loc="upper right", fontsize=8, ncol=3)

# Panel 2 — change-point probability
cp_proba = s['changepoint_proba']
ax2 = fig.add_subplot(gs[2])
ax2.fill_between(range(len(cp_proba)), cp_proba, color="#e74c3c", alpha=0.55)
for cp_t in true_cps:
    ax2.axvline(cp_t, color="#e74c3c", lw=1.5, ls="--", alpha=0.7)
ax2.axhline(0.05, color="grey", lw=0.8, ls=":", label="5% threshold")
ax2.set_xlabel("Time index")
ax2.set_ylabel("P(change-point)")
ax2.set_title("Marginal change-point probabilities")
ax2.legend(fontsize=8)
ax2.set_xlim(0, len(x))

plt.savefig("output.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to output.png")
plt.show()