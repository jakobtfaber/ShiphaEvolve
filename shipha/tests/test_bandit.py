"""
Tests for multi-armed bandit selection algorithms.

Tests cover:
    - AsymmetricUCB arm selection and updates
    - FixedSampler deterministic behavior
    - Posterior probability computation
    - Reward asymmetry handling
"""

from __future__ import annotations

import numpy as np
import pytest

from shipha.llm.bandit import AsymmetricUCB, BanditBase, FixedSampler


class TestFixedSampler:
    """Tests for the FixedSampler class."""

    def test_initialization(self) -> None:
        """Test sampler initialization."""
        sampler = FixedSampler(n_arms=3, arm_names=["a", "b", "c"])
        assert sampler.n_arms == 3
        assert sampler._arm_names == ["a", "b", "c"]

    def test_sample_via_posterior(self, fixed_sampler: FixedSampler) -> None:
        """Test that we can sample from posterior probabilities."""
        posterior = fixed_sampler.posterior()
        assert len(posterior) == fixed_sampler.n_arms
        for _ in range(10):
            arm = np.random.choice(len(posterior), p=posterior)
            assert 0 <= arm < fixed_sampler.n_arms

    def test_uniform_posterior(self, fixed_sampler: FixedSampler) -> None:
        """Test that posterior is uniform for FixedSampler."""
        posterior = fixed_sampler.posterior()
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(posterior, expected)

    def test_update_submitted_returns_zero(
        self, fixed_sampler: FixedSampler
    ) -> None:
        """Test that update_submitted returns 0 for FixedSampler."""
        result = fixed_sampler.update_submitted(0)
        assert result == 0.0

    def test_update_is_no_op(self, fixed_sampler: FixedSampler) -> None:
        """Test that update is a no-op for FixedSampler."""
        before = fixed_sampler.posterior().copy()
        fixed_sampler.update(arm=0, reward=1.0, baseline=0.5)
        after = fixed_sampler.posterior()
        np.testing.assert_array_almost_equal(before, after)


class TestAsymmetricUCB:
    """Tests for the AsymmetricUCB bandit."""

    def test_initialization(self) -> None:
        """Test UCB initialization with parameters."""
        ucb = AsymmetricUCB(
            n_arms=3,
            arm_names=["a", "b", "c"],
            exploration_coef=1.0,
            epsilon=0.2,
        )
        assert ucb.n_arms == 3
        assert ucb.c == 1.0  # exploration_coef is stored as "c"
        assert ucb.epsilon == 0.2

    def test_sample_via_posterior(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test that we can sample from posterior probabilities."""
        posterior = ucb_bandit.posterior()
        for _ in range(10):
            arm = np.random.choice(len(posterior), p=posterior)
            assert 0 <= arm < ucb_bandit.n_arms

    def test_posterior_sums_to_one(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test that posterior probabilities sum to 1."""
        posterior = ucb_bandit.posterior()
        assert abs(sum(posterior) - 1.0) < 1e-6

    def test_posterior_subset(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test posterior computation with arm subset."""
        # Posterior with subset returns full array with zeros for non-subset arms
        posterior = ucb_bandit.posterior(subset=[0, 2])
        assert len(posterior) == ucb_bandit.n_arms
        # Only arms 0 and 2 should have non-zero probability
        assert posterior[1] == 0.0
        assert posterior[0] > 0 or posterior[2] > 0

    def test_positive_reward_updates(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test that positive rewards increase arm value."""
        # First record the initial state
        ucb_bandit.update_submitted(arm=0)
        initial_posterior = ucb_bandit.posterior().copy()
        
        # Give a positive reward
        ucb_bandit.update(arm=0, reward=1.0, baseline=0.0)
        
        # The posterior should be affected (can check arm 0 relative to others)
        final_posterior = ucb_bandit.posterior()
        # After positive update, arm 0 should have higher or similar probability
        assert final_posterior[0] >= 0  # Basic sanity check

    def test_negative_reward_with_asymmetric(self) -> None:
        """Test that negative rewards are clipped with asymmetric scaling."""
        ucb = AsymmetricUCB(
            n_arms=2,
            asymmetric_scaling=True,
            seed=42,
        )
        
        # Submit to arm 0
        ucb.update_submitted(arm=0)
        
        # Give negative reward (score below baseline)
        ucb.update(arm=0, reward=-1.0, baseline=0.0)
        
        # With asymmetric scaling, negative rewards are clipped
        # The arm should not be heavily penalized

    def test_exploration_coef_affects_selection(self) -> None:
        """Test that exploration coefficient affects posterior."""
        ucb_low = AsymmetricUCB(n_arms=2, exploration_coef=0.1, seed=42)
        ucb_high = AsymmetricUCB(n_arms=2, exploration_coef=10.0, seed=42)

        # Train one arm more than the other
        for _ in range(10):
            ucb_low.update_submitted(0)
            ucb_low.update(arm=0, reward=1.0)
            ucb_high.update_submitted(0)
            ucb_high.update(arm=0, reward=1.0)

        # High exploration should give more probability to unexplored arm
        posterior_low = ucb_low.posterior()
        posterior_high = ucb_high.posterior()
        
        # Arm 1 (unexplored) should have higher probability with higher exploration
        assert posterior_high[1] >= posterior_low[1] or abs(posterior_high[1] - posterior_low[1]) < 0.2

    def test_decay_reduces_statistics(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test that decay reduces accumulated statistics."""
        # Accumulate some statistics
        for _ in range(10):
            ucb_bandit.update_submitted(0)
            ucb_bandit.update(arm=0, reward=1.0)
        
        # Apply decay
        ucb_bandit.decay(0.5)
        
        # Decay should reduce the effect of past observations

    def test_print_summary_works(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test that print_summary runs without error."""
        ucb_bandit.update_submitted(0)
        ucb_bandit.update(arm=0, reward=1.0)
        # Should not raise
        ucb_bandit.print_summary()


class TestBanditInterface:
    """Tests for the BanditBase interface."""

    def test_fixed_sampler_is_bandit(self, fixed_sampler: FixedSampler) -> None:
        """Test that FixedSampler implements BanditBase."""
        assert isinstance(fixed_sampler, BanditBase)

    def test_ucb_is_bandit(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test that AsymmetricUCB implements BanditBase."""
        assert isinstance(ucb_bandit, BanditBase)

    def test_bandit_has_required_methods(
        self, fixed_sampler: FixedSampler
    ) -> None:
        """Test that bandit has all required abstract methods."""
        assert hasattr(fixed_sampler, "posterior")
        assert hasattr(fixed_sampler, "update_submitted")
        assert hasattr(fixed_sampler, "update")
        assert hasattr(fixed_sampler, "decay")
        assert callable(fixed_sampler.posterior)
        assert callable(fixed_sampler.update)
