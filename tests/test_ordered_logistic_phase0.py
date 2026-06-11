"""
Phase 0 tests for Ordered Logistic model implementation.

Tests:
1. Model creation and compilation
2. Adaptive prior parameter computation
3. Sampling and convergence
4. Benchmark vs Dirichlet-Multinomial
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import time
import pytest


def generate_synthetic_ordinal_data(
    n_items: int,
    n_obs_per_item: int,
    n_categories: int,
    true_mu: float = 0.0,
    true_sigma: float = 1.0,
    seed: int = 42
) -> dict:
    """Generate synthetic ordinal data with known parameters."""
    np.random.seed(seed)

    # Cutpoints: evenly spaced starting at 0
    spread = 2.0 * np.log(n_categories)
    true_cutpoints = np.linspace(0, spread, n_categories - 1)

    # Item latent values
    true_eta = true_mu + true_sigma * np.random.randn(n_items)

    # Generate observations
    item_counts = np.zeros((n_items, n_categories), dtype=int)

    for i in range(n_items):
        eta_i = true_eta[i]

        # Compute category probabilities
        cum_probs = 1.0 / (1.0 + np.exp(-(true_cutpoints - eta_i)))
        probs = np.zeros(n_categories)
        probs[0] = cum_probs[0]
        probs[1:-1] = cum_probs[1:] - cum_probs[:-1]
        probs[-1] = 1 - cum_probs[-1]
        probs = np.clip(probs, 1e-10, 1)
        probs /= probs.sum()

        # Sample observations
        scores = np.random.choice(n_categories, size=n_obs_per_item, p=probs)
        for s in scores:
            item_counts[i, s] += 1

    return {
        'item_counts': item_counts,
        'item_ns': item_counts.sum(axis=1),
        'true_mu': true_mu,
        'true_sigma': true_sigma,
        'true_cutpoints': true_cutpoints,
        'true_eta': true_eta
    }


class TestAdaptivePriors:
    """Test adaptive prior parameter computation."""

    def test_prior_params_k5(self):
        from optstop.ordinal_model import _compute_adaptive_cutpoint_prior_params
        params = _compute_adaptive_cutpoint_prior_params(5)

        assert params['n_increments'] == 3  # K-2 = 5-2 = 3
        assert params['expected_spread'] > 0
        assert params['expected_increment'] > 0
        assert 'delta_mu' in params
        assert 'delta_sigma' in params

    def test_prior_params_k11(self):
        from optstop.ordinal_model import _compute_adaptive_cutpoint_prior_params
        params = _compute_adaptive_cutpoint_prior_params(11)

        assert params['n_increments'] == 9  # K-2 = 11-2 = 9
        # K=11 should have larger spread than K=5
        params_k5 = _compute_adaptive_cutpoint_prior_params(5)
        assert params['expected_spread'] > params_k5['expected_spread']

    def test_prior_params_scaling(self):
        """Verify spread scales with log(K)."""
        from optstop.ordinal_model import _compute_adaptive_cutpoint_prior_params

        ks = [3, 5, 11, 21]
        spreads = [_compute_adaptive_cutpoint_prior_params(k)['expected_spread'] for k in ks]

        # Spread should increase monotonically
        for i in range(len(spreads) - 1):
            assert spreads[i+1] > spreads[i], f"Spread should increase: {spreads}"


class TestModelCreation:
    """Test model creation and structure."""

    def test_create_model_k11(self):
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        model = _create_ordered_logistic_hierarchical(
            n_categories=11,
            n_items=5
        )

        # Check model has expected variables.
        # Default mode is pre-allocation: the likelihood is applied via a masked
        # Potential ('obs_likelihood') with 'obs_weight', and items use a fixed
        # shape - so there is no 'obs' observed RV and no dynamic 'n_items' Data.
        var_names = set(model.named_vars.keys())
        expected = {'mu_group', 'sigma_group', 'z', 'eta', 'cutpoints',
                    'probs', 'probs_group', 'modal_group', 'entropy_group',
                    'delta_raw', 'obs_likelihood', 'obs_weight',
                    'item_counts', 'item_ns'}

        for expected_var in expected:
            assert expected_var in var_names, f"Missing variable: {expected_var}"

    def test_create_model_k3(self):
        """Test edge case with K=3 (only 1 free increment)."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        model = _create_ordered_logistic_hierarchical(
            n_categories=3,
            n_items=3
        )

        assert 'delta_raw' in model.named_vars
        assert 'cutpoints' in model.named_vars
        assert 'item_ns' in model.named_vars  # Pre-allocated item-count data

    def test_create_model_k2(self):
        """Test edge case with K=2 (no free increments, single fixed cutpoint)."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        model = _create_ordered_logistic_hierarchical(
            n_categories=2,
            n_items=3
        )

        # K=2 should have cutpoints but no delta_raw
        assert 'cutpoints' not in model.named_vars  # It's a constant, not a variable
        assert 'delta_raw' not in model.named_vars
        assert 'item_ns' in model.named_vars  # Pre-allocated item-count data


class TestSampling:
    """Test model sampling and convergence."""

    @pytest.mark.slow
    def test_sampling_converges(self):
        """Test that sampling converges on synthetic data."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical
        import arviz as az

        # Generate synthetic data
        data = generate_synthetic_ordinal_data(
            n_items=5,
            n_obs_per_item=20,
            n_categories=11,
            true_mu=1.0,
            true_sigma=0.5
        )

        # Create model
        model = _create_ordered_logistic_hierarchical(
            n_categories=11,
            n_items=5
        )

        # Set data
        with model:
            pm.set_data({
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })

        # Sample (short run for testing)
        with model:
            trace = pm.sample(
                draws=200,
                tune=200,
                chains=2,
                cores=1,
                random_seed=42,
                progressbar=False,
                return_inferencedata=True
            )

        # Check convergence
        rhat = az.rhat(trace)
        rhat_max = rhat.to_array().max().item()

        assert rhat_max < 1.1, f"R-hat too high: {rhat_max}"

    @pytest.mark.slow
    def test_parameter_recovery(self):
        """Test that model recovers true parameters."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        # Generate synthetic data with known parameters
        true_mu = 2.0
        true_sigma = 0.8
        data = generate_synthetic_ordinal_data(
            n_items=10,
            n_obs_per_item=30,
            n_categories=11,
            true_mu=true_mu,
            true_sigma=true_sigma
        )

        # Create and fit model
        model = _create_ordered_logistic_hierarchical(
            n_categories=11,
            n_items=10
        )

        with model:
            pm.set_data({
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                random_seed=42,
                progressbar=False,
                return_inferencedata=True
            )

        # Check parameter recovery
        mu_samples = trace.posterior['mu_group'].values.flatten()
        sigma_samples = trace.posterior['sigma_group'].values.flatten()

        mu_mean = np.mean(mu_samples)
        sigma_mean = np.mean(sigma_samples)

        # Parameters should be within reasonable range of true values
        # (allowing for sampling uncertainty)
        assert abs(mu_mean - true_mu) < 1.0, f"mu_mean={mu_mean}, true={true_mu}"
        assert abs(sigma_mean - true_sigma) < 0.5, f"sigma_mean={sigma_mean}, true={true_sigma}"


class TestBenchmark:
    """Benchmark Ordered Logistic vs Dirichlet-Multinomial."""

    def create_dirichlet_multinomial(self, n_categories: int, n_items: int) -> pm.Model:
        """Create Dirichlet-Multinomial model for comparison."""
        with pm.Model() as model:
            # Group-level
            alpha_prior = np.ones(n_categories)
            alpha_group = pm.Dirichlet("alpha_group", a=alpha_prior)
            kappa = pm.Gamma("kappa", alpha=2, beta=0.1)

            # Data containers
            item_counts_data = pm.Data(
                "item_counts",
                np.ones((n_items, n_categories), dtype="int64")
            )
            item_ns_data = pm.Data(
                "item_ns",
                np.ones(n_items, dtype="int64") * n_categories
            )

            # Item-level
            alpha_item = alpha_group * kappa
            p_item = pm.Dirichlet("p_item", a=alpha_item, shape=(n_items, n_categories))

            # Likelihood
            obs = pm.Multinomial("obs", n=item_ns_data, p=p_item, observed=item_counts_data)

            # Derived quantities
            modal_group = pm.Deterministic("modal_group", pt.argmax(alpha_group))
            p_group_normalized = alpha_group / pm.math.sum(alpha_group)
            entropy_group = pm.Deterministic(
                "entropy_group",
                -pm.math.sum(p_group_normalized * pm.math.log(p_group_normalized + 1e-10))
            )

        return model

    @pytest.mark.slow
    def test_benchmark_timing(self):
        """Compare sampling time between models."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        # Generate data
        data = generate_synthetic_ordinal_data(
            n_items=5,
            n_obs_per_item=20,
            n_categories=11,
            true_mu=1.0,
            true_sigma=0.5
        )

        sampling_kwargs = {
            'draws': 200,
            'tune': 200,
            'chains': 2,
            'cores': 1,
            'random_seed': 42,
            'progressbar': False,
            'return_inferencedata': True
        }

        # Benchmark Dirichlet-Multinomial
        dm_model = self.create_dirichlet_multinomial(11, 5)
        with dm_model:
            pm.set_data({
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })

        start = time.time()
        with dm_model:
            dm_trace = pm.sample(**sampling_kwargs)
        dm_time = time.time() - start

        # Benchmark Ordered Logistic
        ol_model = _create_ordered_logistic_hierarchical(11, 5)
        with ol_model:
            pm.set_data({
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })

        start = time.time()
        with ol_model:
            ol_trace = pm.sample(**sampling_kwargs)
        ol_time = time.time() - start

        # Report results
        ratio = ol_time / dm_time
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Dirichlet-Multinomial: {dm_time:.2f}s")
        print(f"Ordered Logistic:      {ol_time:.2f}s")
        print(f"Ratio (OL/DM):         {ratio:.2f}x")
        print(f"{'='*60}\n")

        # Store results for reporting
        assert dm_time > 0
        assert ol_time > 0
        # Expected: OL is 1-3x slower, but should complete
        assert ratio < 10, f"OL too slow: {ratio}x"


if __name__ == "__main__":
    # Run quick tests
    print("Running Phase 0 tests...")

    print("\n1. Testing adaptive prior parameters...")
    test_priors = TestAdaptivePriors()
    test_priors.test_prior_params_k5()
    test_priors.test_prior_params_k11()
    test_priors.test_prior_params_scaling()
    print("   PASSED")

    print("\n2. Testing model creation...")
    test_model = TestModelCreation()
    test_model.test_create_model_k11()
    test_model.test_create_model_k3()
    test_model.test_create_model_k2()
    print("   PASSED")

    print("\n3. Running benchmark (this may take a minute)...")
    test_bench = TestBenchmark()
    test_bench.test_benchmark_timing()

    print("\nAll Phase 0 tests passed!")
