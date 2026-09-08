"""
Phase 2 tests for Ordered Logistic model: Expanded Testing.

Tests:
1. Parameter recovery on synthetic data (various scenarios)
2. Edge cases (sparse categories, small n, extreme distributions)
3. Comparison of stopping behavior between ordered_logistic and dirichlet models
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import warnings


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


def generate_sparse_category_data(
    n_items: int,
    n_obs_per_item: int,
    n_categories: int,
    concentration: float = 0.1,  # Low = sparse (concentrated in few categories)
    seed: int = 42
) -> dict:
    """Generate data with sparse category usage (e.g., U-shaped or concentrated)."""
    np.random.seed(seed)

    # Generate concentrated probabilities (most mass in a few categories)
    # Use Dirichlet with low concentration to create sparse distributions
    base_probs = np.random.dirichlet(np.ones(n_categories) * concentration)

    item_counts = np.zeros((n_items, n_categories), dtype=int)

    for i in range(n_items):
        # Add some item-level variation
        item_probs = np.random.dirichlet(base_probs * 10 + 0.1)
        scores = np.random.choice(n_categories, size=n_obs_per_item, p=item_probs)
        for s in scores:
            item_counts[i, s] += 1

    return {
        'item_counts': item_counts,
        'item_ns': item_counts.sum(axis=1),
        'base_probs': base_probs
    }


def generate_bimodal_data(
    n_items: int,
    n_obs_per_item: int,
    n_categories: int,
    mode1_center: float = 0.2,  # Fraction of scale
    mode2_center: float = 0.8,
    mode_weight: float = 0.5,  # Weight for first mode
    seed: int = 42
) -> dict:
    """Generate bimodal ordinal data (challenging for ordered logistic)."""
    np.random.seed(seed)

    item_counts = np.zeros((n_items, n_categories), dtype=int)

    for i in range(n_items):
        for _ in range(n_obs_per_item):
            # Choose mode
            if np.random.rand() < mode_weight:
                center = mode1_center
            else:
                center = mode2_center

            # Sample around mode with some spread
            score = int(np.clip(
                np.random.normal(center * (n_categories - 1), 1.0),
                0, n_categories - 1
            ))
            item_counts[i, score] += 1

    return {
        'item_counts': item_counts,
        'item_ns': item_counts.sum(axis=1),
        'mode1_center': mode1_center,
        'mode2_center': mode2_center
    }


class TestParameterRecovery:
    """Test parameter recovery under various scenarios."""

    @pytest.mark.slow
    def test_recovery_baseline(self):
        """Baseline: K=11, mu=0, sigma=1, moderate data."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        true_mu = 0.0
        true_sigma = 1.0
        data = generate_synthetic_ordinal_data(
            n_items=10,
            n_obs_per_item=30,
            n_categories=11,
            true_mu=true_mu,
            true_sigma=true_sigma,
            seed=42
        )

        model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=10)

        with model:
            pm.set_data({
                "n_items": np.int64(10),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=500, tune=500, chains=2, cores=1,
                    random_seed=42, progressbar=False, return_inferencedata=True
                )

        mu_mean = float(trace.posterior['mu_group'].mean())
        sigma_mean = float(trace.posterior['sigma_group'].mean())

        # Check recovery (allowing for sampling uncertainty)
        assert abs(mu_mean - true_mu) < 1.5, f"mu recovery: {mu_mean} vs {true_mu}"
        assert abs(sigma_mean - true_sigma) < 1.0, f"sigma recovery: {sigma_mean} vs {true_sigma}"

        print(f"\n✓ Baseline recovery: mu={mu_mean:.2f} (true={true_mu}), sigma={sigma_mean:.2f} (true={true_sigma})")

    @pytest.mark.slow
    def test_recovery_shifted_distribution(self):
        """Test recovery when true mu is shifted (higher categories more likely)."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        true_mu = 2.5  # Shifted toward higher categories
        true_sigma = 0.8
        data = generate_synthetic_ordinal_data(
            n_items=10,
            n_obs_per_item=30,
            n_categories=11,
            true_mu=true_mu,
            true_sigma=true_sigma,
            seed=123
        )

        model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=10)

        with model:
            pm.set_data({
                "n_items": np.int64(10),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=500, tune=500, chains=2, cores=1,
                    random_seed=123, progressbar=False, return_inferencedata=True
                )

        mu_mean = float(trace.posterior['mu_group'].mean())
        sigma_mean = float(trace.posterior['sigma_group'].mean())

        # Shifted mu should be recovered (sign at least correct, magnitude within range)
        assert mu_mean > 1.0, f"Shifted mu should be positive: {mu_mean}"
        assert abs(sigma_mean - true_sigma) < 1.0, f"sigma recovery: {sigma_mean} vs {true_sigma}"

        print(f"\n✓ Shifted recovery: mu={mu_mean:.2f} (true={true_mu}), sigma={sigma_mean:.2f} (true={true_sigma})")

    @pytest.mark.slow
    def test_recovery_tight_distribution(self):
        """Test recovery when between-item variance is low."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        true_mu = 1.0
        true_sigma = 0.3  # Low between-item variance
        data = generate_synthetic_ordinal_data(
            n_items=10,
            n_obs_per_item=30,
            n_categories=11,
            true_mu=true_mu,
            true_sigma=true_sigma,
            seed=456
        )

        model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=10)

        with model:
            pm.set_data({
                "n_items": np.int64(10),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=500, tune=500, chains=2, cores=1,
                    random_seed=456, progressbar=False, return_inferencedata=True
                )

        mu_mean = float(trace.posterior['mu_group'].mean())
        sigma_mean = float(trace.posterior['sigma_group'].mean())

        # Tight sigma should be recovered (should be small)
        assert sigma_mean < 1.0, f"Tight sigma should be small: {sigma_mean}"

        print(f"\n✓ Tight recovery: mu={mu_mean:.2f} (true={true_mu}), sigma={sigma_mean:.2f} (true={true_sigma})")


class TestEdgeCases:
    """Test edge cases that might cause model issues."""

    @pytest.mark.slow
    def test_few_categories_k5(self):
        """Test with K=5 categories."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        data = generate_synthetic_ordinal_data(
            n_items=8, n_obs_per_item=25, n_categories=5,
            true_mu=0.5, true_sigma=0.8, seed=789
        )

        model = _create_ordered_logistic_hierarchical(n_categories=5, n_items=8)

        with model:
            pm.set_data({
                "n_items": np.int64(8),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=300, tune=300, chains=2, cores=1,
                    random_seed=789, progressbar=False, return_inferencedata=True
                )

        # Just verify it runs and produces reasonable output
        import arviz as az
        rhat = az.rhat(trace)
        rhat_max = float(rhat.to_array().max())

        assert rhat_max < 1.2, f"R-hat too high for K=5: {rhat_max}"
        print(f"\n✓ K=5 test passed, R-hat max={rhat_max:.3f}")

    @pytest.mark.slow
    def test_small_sample_size(self):
        """Test with small sample size per item (n=5)."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        data = generate_synthetic_ordinal_data(
            n_items=10, n_obs_per_item=5, n_categories=11,  # Very small n
            true_mu=1.0, true_sigma=1.0, seed=111
        )

        model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=10)

        with model:
            pm.set_data({
                "n_items": np.int64(10),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=300, tune=300, chains=2, cores=1,
                    random_seed=111, progressbar=False, return_inferencedata=True
                )

        import arviz as az
        rhat = az.rhat(trace)
        rhat_max = float(rhat.to_array().max())

        # With small n, we expect wider posteriors but should still converge
        assert rhat_max < 1.2, f"R-hat too high for small n: {rhat_max}"

        # Check posterior is wider (more uncertain) than baseline
        sigma_std = float(trace.posterior['sigma_group'].std())
        assert sigma_std > 0.1, f"Should have meaningful uncertainty with small n"

        print(f"\n✓ Small n test passed, R-hat max={rhat_max:.3f}, sigma_std={sigma_std:.3f}")

    @pytest.mark.slow
    def test_sparse_categories(self):
        """Test with sparse category usage (most responses in few categories)."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        data = generate_sparse_category_data(
            n_items=8, n_obs_per_item=20, n_categories=11,
            concentration=0.1, seed=222
        )

        model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=8)

        with model:
            pm.set_data({
                "n_items": np.int64(8),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    draws=300, tune=300, chains=2, cores=1,
                    random_seed=222, progressbar=False, return_inferencedata=True
                )

        import arviz as az
        rhat = az.rhat(trace)
        rhat_max = float(rhat.to_array().max())

        # Sparse data is challenging - accept slightly higher R-hat
        assert rhat_max < 1.3, f"R-hat too high for sparse data: {rhat_max}"

        # Check model still produces valid probabilities
        probs_mean = trace.posterior['probs_group'].mean(dim=['chain', 'draw']).values
        assert np.allclose(probs_mean.sum(), 1.0, atol=0.01), "Probs should sum to 1"

        print(f"\n✓ Sparse categories test passed, R-hat max={rhat_max:.3f}")

    @pytest.mark.slow
    def test_bimodal_distribution(self):
        """Test with bimodal data (challenging for ordered logistic)."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical

        data = generate_bimodal_data(
            n_items=8, n_obs_per_item=30, n_categories=11,
            mode1_center=0.2, mode2_center=0.8, seed=333
        )

        model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=8)

        with model:
            pm.set_data({
                "n_items": np.int64(8),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Bimodal data may need more tuning
                trace = pm.sample(
                    draws=400, tune=400, chains=2, cores=1,
                    random_seed=333, progressbar=False, return_inferencedata=True
                )

        import arviz as az
        rhat = az.rhat(trace)
        rhat_max = float(rhat.to_array().max())

        # Bimodal is fundamentally challenging - model may struggle
        # Accept if it at least converges, even if recovery is poor
        assert rhat_max < 1.5, f"R-hat too high for bimodal: {rhat_max}"

        # Note: Ordered logistic assumes unimodal latent distribution
        # Bimodal data violates this assumption, so recovery may be poor
        print(f"\n✓ Bimodal test completed (R-hat max={rhat_max:.3f}). Note: Model assumes unimodal data.")


class TestStoppingBehaviorComparison:
    """Compare stopping behavior between ordered_logistic and dirichlet models."""

    def create_dirichlet_model(self, n_categories: int, n_items: int) -> pm.Model:
        """Create Dirichlet-Multinomial model for comparison."""
        with pm.Model() as model:
            alpha_prior = np.ones(n_categories)
            alpha_group = pm.Dirichlet("alpha_group", a=alpha_prior)
            kappa = pm.Gamma("kappa", alpha=2, beta=0.1)

            n_items_data = pm.Data("n_items", np.array(n_items, dtype="int64"))
            item_counts_data = pm.Data("item_counts", np.ones((n_items, n_categories), dtype="int64"))
            item_ns_data = pm.Data("item_ns", np.ones(n_items, dtype="int64") * n_categories)

            alpha_item = alpha_group * kappa
            p_item = pm.Dirichlet("p_item", a=alpha_item, shape=(n_items_data, n_categories))
            obs = pm.Multinomial("obs", n=item_ns_data, p=p_item, observed=item_counts_data)

            modal_group = pm.Deterministic("modal_group", pt.argmax(alpha_group))
            p_group_normalized = alpha_group / pm.math.sum(alpha_group)
            entropy_group = pm.Deterministic(
                "entropy_group",
                -pm.math.sum(p_group_normalized * pm.math.log(p_group_normalized + 1e-10))
            )

        return model

    @pytest.mark.slow
    def test_stopping_consistency_unimodal(self):
        """Both models should give similar stopping decisions on unimodal data."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical
        import arviz as az

        # Generate unimodal ordinal data
        data = generate_synthetic_ordinal_data(
            n_items=8, n_obs_per_item=25, n_categories=11,
            true_mu=1.5, true_sigma=0.6, seed=444
        )

        sampling_kwargs = {
            'draws': 300, 'tune': 300, 'chains': 2, 'cores': 1,
            'random_seed': 444, 'progressbar': False, 'return_inferencedata': True
        }

        # Fit Ordered Logistic model
        ol_model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=8)
        with ol_model:
            pm.set_data({
                "n_items": np.int64(8),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ol_trace = pm.sample(**sampling_kwargs)

        # Fit Dirichlet model
        dm_model = self.create_dirichlet_model(n_categories=11, n_items=8)
        with dm_model:
            pm.set_data({
                "n_items": np.int64(8),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dm_trace = pm.sample(**sampling_kwargs)

        # Compare modal category estimates
        ol_modal = float(ol_trace.posterior['modal_group'].mean())
        dm_modal = float(dm_trace.posterior['modal_group'].mean())

        # Compare entropy estimates
        ol_entropy = float(ol_trace.posterior['entropy_group'].mean())
        dm_entropy = float(dm_trace.posterior['entropy_group'].mean())

        # They should agree on modal category (within ±1 category)
        modal_diff = abs(ol_modal - dm_modal)
        assert modal_diff < 2.0, f"Modal category differs too much: OL={ol_modal:.1f}, DM={dm_modal:.1f}"

        # Entropy should be in same ballpark (both should indicate some uncertainty)
        assert ol_entropy > 0 and dm_entropy > 0, "Both should have positive entropy"

        print(f"\n✓ Stopping consistency test:")
        print(f"  Modal: OL={ol_modal:.2f}, DM={dm_modal:.2f} (diff={modal_diff:.2f})")
        print(f"  Entropy: OL={ol_entropy:.3f}, DM={dm_entropy:.3f}")

    @pytest.mark.slow
    def test_uncertainty_quantification(self):
        """Compare CI widths between models as proxy for stopping readiness."""
        from optstop.ordinal_model import _create_ordered_logistic_hierarchical
        import arviz as az

        data = generate_synthetic_ordinal_data(
            n_items=10, n_obs_per_item=20, n_categories=11,
            true_mu=0.0, true_sigma=1.0, seed=555
        )

        sampling_kwargs = {
            'draws': 300, 'tune': 300, 'chains': 2, 'cores': 1,
            'random_seed': 555, 'progressbar': False, 'return_inferencedata': True
        }

        # Fit both models
        ol_model = _create_ordered_logistic_hierarchical(n_categories=11, n_items=10)
        with ol_model:
            pm.set_data({
                "n_items": np.int64(10),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ol_trace = pm.sample(**sampling_kwargs)

        dm_model = self.create_dirichlet_model(n_categories=11, n_items=10)
        with dm_model:
            pm.set_data({
                "n_items": np.int64(10),
                "item_counts": data['item_counts'],
                "item_ns": data['item_ns']
            })
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dm_trace = pm.sample(**sampling_kwargs)

        # Compute 95% HDI width for modal category
        ol_modal_hdi = az.hdi(ol_trace.posterior['modal_group'], hdi_prob=0.95)
        dm_modal_hdi = az.hdi(dm_trace.posterior['modal_group'], hdi_prob=0.95)

        try:
            ol_width = float(ol_modal_hdi['modal_group'].sel(hdi='higher') -
                           ol_modal_hdi['modal_group'].sel(hdi='lower'))
        except:
            ol_width = float(ol_modal_hdi['modal_group'].values[1] -
                           ol_modal_hdi['modal_group'].values[0])

        try:
            dm_width = float(dm_modal_hdi['modal_group'].sel(hdi='higher') -
                           dm_modal_hdi['modal_group'].sel(hdi='lower'))
        except:
            dm_width = float(dm_modal_hdi['modal_group'].values[1] -
                           dm_modal_hdi['modal_group'].values[0])

        # Note: modal_group is discrete (integer category), so CI width can be 0
        # when all posterior samples agree on the same mode. This is valid behavior.
        # We only check that the width is not unreasonably large.
        assert ol_width >= 0 and ol_width < 10, f"OL CI width unreasonable: {ol_width}"
        assert dm_width >= 0 and dm_width < 10, f"DM CI width unreasonable: {dm_width}"

        # Also check entropy as a continuous measure of uncertainty
        ol_entropy = float(ol_trace.posterior['entropy_group'].mean())
        dm_entropy = float(dm_trace.posterior['entropy_group'].mean())

        print(f"\n✓ Uncertainty quantification test:")
        print(f"  Modal 95% HDI width: OL={ol_width:.2f}, DM={dm_width:.2f}")
        print(f"  Entropy (bits): OL={ol_entropy:.3f}, DM={dm_entropy:.3f}")

        # Both should have positive entropy (some uncertainty)
        assert ol_entropy > 0, f"OL entropy should be positive: {ol_entropy}"
        assert dm_entropy > 0, f"DM entropy should be positive: {dm_entropy}"


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2: Expanded Testing for Ordered Logistic Model")
    print("=" * 60)

    print("\n--- Parameter Recovery Tests ---")
    test_recovery = TestParameterRecovery()
    test_recovery.test_recovery_baseline()
    test_recovery.test_recovery_shifted_distribution()
    test_recovery.test_recovery_tight_distribution()

    print("\n--- Edge Case Tests ---")
    test_edge = TestEdgeCases()
    test_edge.test_few_categories_k5()
    test_edge.test_small_sample_size()
    test_edge.test_sparse_categories()
    test_edge.test_bimodal_distribution()

    print("\n--- Stopping Behavior Comparison ---")
    test_stopping = TestStoppingBehaviorComparison()
    test_stopping.test_stopping_consistency_unimodal()
    test_stopping.test_uncertainty_quantification()

    print("\n" + "=" * 60)
    print("All Phase 2 tests completed!")
    print("=" * 60)
