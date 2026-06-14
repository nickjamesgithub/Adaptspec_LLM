import numpy as np
from adaptspec import adaptspec, summarise, plot_results

log_returns = np.random.randn(500)   # your series

result = adaptspec(
    log_returns,
    nexp_max = 10,    # max segments
    nbeta    = 7,     # spline order
    niter    = 5000,  # MCMC iterations
    nburn    = 2000,  # burn-in
    tmin     = 20,    # min segment length
    seed     = 42,
)

s = summarise(result)
print("Mean segments:", s["mean_nexp"])
print("Change-points:", s["modal_changepoints"][:5])
print("Boundaries:",   s["segment_boundaries"])

fig = plot_results(log_returns, result)
fig.savefig("output.png", dpi=150)