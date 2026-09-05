# Real bugs / surprises hit during this build (raw log for the README)

1. **Phase 1 — `atol=0` in the unbroadcast reference test.**
   `test_unbroadcast_values[(1,) <- (2,3,4)]` failed intermittently. `unbroadcast`
   reduces in two passes (sum leading axes, then sum size-1 axes with keepdims);
   the reference summed all replicated axes at once. Same math, different
   float-addition order, ~1e-16 disagreement. Caught only on some random seeds.
   Fix: rtol=1e-12 in the test. Engine was correct; the *test* was wrong.
