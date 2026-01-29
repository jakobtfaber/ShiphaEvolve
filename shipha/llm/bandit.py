"""
Multi-Armed Bandit for LLM Model Selection

Ported from ShinkaEvolve's dynamic_sampling.py - implements AsymmetricUCB
for intelligent LLM selection based on evolutionary performance.

The AsymmetricUCB algorithm treats improvements over a baseline asymmetrically:
- Positive rewards (improvements) are tracked and used for selection
- Negative rewards (regressions) are treated as zero
- This encourages exploration while not penalizing models for occasional failures
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence, List, Any
from scipy.special import logsumexp
from rich.table import Table
from rich.console import Console
import rich.box

Arm = Union[int, str]
Subset = Optional[Union[np.ndarray, Sequence[Arm]]]


def _logadd(x_log: float, y_log: float, w1: float = 1.0, w2: float = 1.0) -> float:
    """Log-domain addition: log(w1*exp(x) + w2*exp(y))"""
    out, _ = logsumexp([x_log, y_log], b=[w1, w2], return_sign=True)
    return float(out)


def _logdiffexp(a_log: float, b_log: float) -> float:
    """Log-domain subtraction: log(exp(a) - exp(b)), requires a > b"""
    if a_log <= b_log:
        return -np.inf
    return a_log + np.log1p(-np.exp(b_log - a_log))


def _logexpm1(z: float) -> float:
    """Compute log(exp(z) - 1) numerically stably"""
    if z > 0:
        return z + np.log1p(-np.exp(-z))
    else:
        return np.log(np.expm1(z)) if z > -30 else z


class BanditBase(ABC):
    """Abstract base class for multi-armed bandit algorithms."""

    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = None,
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
    ):
        """
        Initialize bandit.

        Args:
            n_arms: Number of arms (models to choose from)
            seed: Random seed for reproducibility
            arm_names: Human-readable names for each arm (e.g., model names)
            auto_decay: If set, automatically decay statistics after each update
            shift_by_baseline: Whether to shift rewards by baseline score
            shift_by_parent: Whether to shift rewards by parent program score
        """
        self.rng = np.random.default_rng(seed)

        if arm_names is None and n_arms is None:
            raise ValueError("provide n_arms or arm_names")
        if arm_names is not None:
            if n_arms is not None and int(n_arms) != len(arm_names):
                raise ValueError("len(arm_names) must equal n_arms")
            self._arm_names = list(arm_names)
            self._name_to_idx = {n: i for i, n in enumerate(self._arm_names)}
            self._n_arms = len(self._arm_names)
        else:
            self._arm_names = None
            self._name_to_idx = {}
            self._n_arms = int(n_arms)

        self._baseline = 0.0
        self._shift_by_baseline = bool(shift_by_baseline)
        self._shift_by_parent = bool(shift_by_parent)
        if auto_decay is not None and not (0.0 < auto_decay <= 1.0):
            raise ValueError("auto_decay must be in (0, 1]")
        self._auto_decay = auto_decay

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def set_baseline_score(self, baseline: float) -> None:
        """Set the baseline score for reward computation."""
        self._baseline = float(baseline)

    def _resolve_arm(self, arm: Arm) -> int:
        """Convert arm name or index to integer index."""
        if isinstance(arm, int):
            return int(arm)
        if self._arm_names is None:
            try:
                return int(arm)
            except Exception as e:
                raise ValueError("string arm requires arm_names") from e
        if arm not in self._name_to_idx:
            raise ValueError(f"unknown arm name '{arm}'")
        return self._name_to_idx[arm]

    def _resolve_subset(self, subset: Subset) -> np.ndarray:
        """Convert subset specification to array of indices."""
        if subset is None:
            return np.arange(self.n_arms, dtype=np.int64)
        if isinstance(subset, np.ndarray) and np.issubdtype(subset.dtype, np.integer):
            return subset.astype(np.int64)
        idxs = [self._resolve_arm(a) for a in subset]
        return np.asarray(idxs, dtype=np.int64)

    def _maybe_decay(self) -> None:
        """Apply auto-decay if configured."""
        if self._auto_decay is not None:
            self.decay(self._auto_decay)

    @abstractmethod
    def update_submitted(self, arm: Arm) -> float:
        """Record that a sample was submitted to this arm (before result known)."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        arm: Arm,
        reward: Optional[float],
        baseline: Optional[float] = None,
    ) -> None:
        """Update arm statistics after receiving reward."""
        raise NotImplementedError

    @abstractmethod
    def posterior(
        self,
        subset: Subset = None,
        samples: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get selection probabilities for each arm."""
        raise NotImplementedError

    @abstractmethod
    def decay(self, factor: float) -> None:
        """Decay accumulated statistics by factor."""
        raise NotImplementedError

    @abstractmethod
    def print_summary(self) -> None:
        """Print current bandit statistics."""
        raise NotImplementedError


class AsymmetricUCB(BanditBase):
    """
    Asymmetric UCB1 with ε-exploration and adaptive scaling.

    This algorithm is optimized for LLM selection in evolutionary optimization:
    - Uses UCB1 for exploration/exploitation balance
    - Asymmetric scaling: only counts positive improvements (above baseline)
    - Adaptive scaling: normalizes rewards based on observed range
    - ε-greedy fallback: ensures minimum exploration

    The asymmetric treatment is key for evolution - we want to find models
    that occasionally produce breakthroughs, not models that are consistently
    mediocre. A model that produces 9 failures and 1 breakthrough is valuable.
    """

    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        exploration_coef: float = 1.0,
        epsilon: float = 0.2,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = 0.95,
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
        adaptive_scale: bool = True,
        asymmetric_scaling: bool = True,
        exponential_base: Optional[float] = 1.0,
    ):
        """
        Initialize AsymmetricUCB.

        Args:
            n_arms: Number of arms (LLM models)
            seed: Random seed
            exploration_coef: UCB exploration coefficient (c in UCB1)
            epsilon: Minimum exploration probability
            arm_names: Names of LLM models
            auto_decay: Decay factor applied after each update (0.95 = 5% decay)
            shift_by_baseline: Shift rewards by global baseline
            shift_by_parent: Shift rewards by parent program score
            adaptive_scale: Normalize rewards by observed range
            asymmetric_scaling: Only count positive improvements
            exponential_base: Base for exponential scaling (None = linear)
        """
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            arm_names=arm_names,
            auto_decay=auto_decay,
            shift_by_baseline=shift_by_baseline,
            shift_by_parent=shift_by_parent,
        )
        if asymmetric_scaling:
            assert shift_by_baseline or shift_by_parent, (
                "asymmetric scaling requires at least one of "
                "shift_by_baseline or shift_by_parent to be True"
            )
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")

        self.c = float(exploration_coef)
        self.epsilon = float(epsilon)
        self.adaptive_scale = bool(adaptive_scale)
        self.asymmetric_scaling = bool(asymmetric_scaling)
        self.exponential_base = exponential_base
        self.use_exponential_scaling = self.exponential_base is not None

        if self.exponential_base is not None:
            assert self.exponential_base > 0.0, "exponential_base must be > 0"
            self.exponential_base = float(exponential_base)

        n = self.n_arms
        self.n_submitted = np.zeros(n, dtype=np.float64)
        self.n_completed = np.zeros(n, dtype=np.float64)

        if self.use_exponential_scaling:
            self.s = np.full(n, -np.inf, dtype=np.float64)
        else:
            self.s = np.zeros(n, dtype=np.float64)
        self.divs = np.zeros(n, dtype=np.float64)

        if self.asymmetric_scaling:
            if self.use_exponential_scaling:
                self._obs_max = -np.inf
                self._obs_min = -np.inf
            else:
                self._obs_min = 0.0
                self._obs_max = 0.0
        else:
            self._obs_max = -np.inf
            self._obs_min = np.inf

    @property
    def n(self) -> np.ndarray:
        """Get effective sample counts per arm."""
        return np.maximum(self.n_submitted, self.n_completed)

    def _add_to_reward(
        self, r: float, value: float, coeff_r: float = 1, coeff_value: float = 1
    ) -> float:
        """Add reward in appropriate space (log or linear)."""
        if self.use_exponential_scaling:
            out, _ = logsumexp([r, value], b=[coeff_r, coeff_value], return_sign=True)
            return float(out)
        else:
            return coeff_r * r + coeff_value * value

    def _multiply_reward(self, r: float, value: float) -> float:
        """Multiply reward in appropriate space."""
        if self.use_exponential_scaling:
            assert value > 0, "Multipliers in log space must be > 0"
            return r + np.log(value)
        else:
            return r * value

    def _mean(self) -> np.ndarray:
        """Get mean reward per arm."""
        denom = np.maximum(self.divs, 1e-7)
        if self.use_exponential_scaling:
            return self.s - np.log(denom)
        else:
            return self.s / denom

    def _update_obs_range(self, r: float) -> None:
        """Update observed reward range."""
        if r > self._obs_max:
            self._obs_max = r
        if not (self.use_exponential_scaling and self.asymmetric_scaling):
            if r < self._obs_min:
                self._obs_min = r

    def _have_obs_range(self) -> bool:
        """Check if we have valid observation range."""
        if self.use_exponential_scaling and self.asymmetric_scaling:
            return np.isfinite(self._obs_max)
        return (
            np.isfinite(self._obs_min)
            and np.isfinite(self._obs_max)
            and (self._obs_max - self._obs_min) > 0.0
        )

    def _impute_worst_reward(self) -> float:
        """Impute reward for failed evaluations."""
        if self.asymmetric_scaling:
            return -np.inf if self.use_exponential_scaling else 0.0

        seen = self.n > 0
        if not np.any(seen):
            return 0.0

        denom = np.maximum(self.divs[seen], 1e-7)
        mu = self.s[seen] / denom
        mu_min = float(mu.min())
        if mu.size >= 2:
            s = float(mu.std(ddof=1))
            sigma = 1.0 if (not np.isfinite(s) or s <= 0.0) else s
        else:
            sigma = 1.0
        return mu_min - sigma

    def _normalized_means(self, idx: np.ndarray) -> np.ndarray:
        """Get normalized mean rewards for arms in idx."""
        if not self.adaptive_scale or not self._have_obs_range():
            m = self._mean()[idx]
            return np.exp(m) if self.use_exponential_scaling else m
        elif self.use_exponential_scaling and self.asymmetric_scaling:
            mlog = self._mean()[idx]
            return np.exp(mlog - self._obs_max)
        elif self.use_exponential_scaling:
            means_log = self._mean()[idx]
            rng_log = _logdiffexp(self._obs_max, self._obs_min)
            num_log = _logdiffexp(means_log, self._obs_min)
            return np.exp(num_log - rng_log)
        else:
            means = self._mean()[idx]
            rng = max(self._obs_max - self._obs_min, 1e-9)
            return (means - self._obs_min) / rng

    def update_submitted(self, arm: Arm) -> float:
        """Record that a sample was submitted to this arm."""
        arm_idx = self._resolve_arm(arm)
        self.n_submitted[arm_idx] += 1.0
        return self.n[arm_idx]

    def update(
        self,
        arm: Arm,
        reward: Optional[float],
        baseline: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Update arm statistics after receiving reward.

        Args:
            arm: Arm that was pulled (model name or index)
            reward: Score achieved (None if evaluation failed)
            baseline: Parent program score (for relative improvement)

        Returns:
            Tuple of (processed_reward, effective_baseline)
        """
        i = self._resolve_arm(arm)
        is_real = reward is not None
        r_raw = float(reward) if is_real else self._impute_worst_reward()

        # Determine effective baseline
        if self._shift_by_parent and self._shift_by_baseline:
            baseline = (
                self._baseline if baseline is None else max(baseline, self._baseline)
            )
        elif self._shift_by_baseline:
            baseline = self._baseline
        elif not self._shift_by_parent:
            baseline = 0.0
        if baseline is None:
            raise ValueError("baseline required when shifting is active")

        r = r_raw - baseline

        # Asymmetric: only count positive improvements
        if self.asymmetric_scaling:
            r = max(r, 0.0)

        self.divs[i] += 1.0
        self.n_completed[i] += 1.0

        if self.use_exponential_scaling and self.asymmetric_scaling:
            z = r * self.exponential_base
            if self._shift_by_baseline:
                contrib_log = _logexpm1(z)
            else:
                contrib_log = z
            self.s[i] = _logadd(self.s[i], contrib_log)
            if self.adaptive_scale and is_real:
                self._update_obs_range(contrib_log)
        else:
            self.s[i] += r
            if self.adaptive_scale and is_real:
                self._update_obs_range(r)

        self._maybe_decay()
        return r, baseline

    def posterior(
        self,
        subset: Subset = None,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get selection probabilities for each arm.

        Args:
            subset: Subset of arms to consider (None = all)
            samples: Number of samples to allocate (for batch)

        Returns:
            Array of selection probabilities, one per arm
        """
        idx = self._resolve_subset(subset)
        if samples is None or int(samples) <= 1:
            n_sub = self.n[idx]
            probs = np.zeros(self._n_arms, dtype=np.float64)

            # If no arms have been tried, uniform
            if np.all(n_sub <= 0.0):
                p = np.ones(idx.size) / idx.size
                probs[idx] = p
                return probs

            # Prioritize unseen arms
            unseen = np.where(n_sub <= 0.0)[0]
            if unseen.size > 0:
                p = np.ones(unseen.size) / unseen.size
                probs[idx[unseen]] = p
                return probs

            # UCB1 scores
            t = float(self.n.sum())
            base = self._normalized_means(idx)
            num = 2.0 * np.log(max(t, 2.0))
            bonus = self.c * np.sqrt(num / n_sub)
            scores = base + bonus

            # ε-greedy on top of UCB
            winners = np.where(scores == scores.max())[0]
            rem = idx.size - winners.size
            p_sub = np.zeros(idx.size, dtype=np.float64)
            if rem == 0:
                p_sub[:] = 1.0 / idx.size
            else:
                p_sub[winners] = (1.0 - self.epsilon) / winners.size
                mask = np.ones(idx.size, dtype=bool)
                mask[winners] = False
                p_sub[mask] = self.epsilon / rem
            probs[idx] = p_sub
            return probs
        else:
            return self._posterior_batch(idx, samples)

    def _posterior_batch(self, idx: np.ndarray, k: int) -> np.ndarray:
        """Get probabilities for batch allocation of k samples."""
        A = idx.size
        probs = np.zeros(self._n_arms, dtype=np.float64)
        if k <= 0 or A == 0:
            return probs

        n_sub = self.n[idx].astype(np.float64)
        v = np.zeros(A, dtype=np.int64)

        if np.all(n_sub <= 0.0):
            p = np.ones(A, dtype=np.float64) / A
            probs[idx] = p
            return probs

        # Handle unseen arms first
        unseen = np.where(n_sub <= 0.0)[0]
        if unseen.size > 0:
            if k >= unseen.size:
                v[unseen] += 1
                k -= unseen.size
            else:
                take = int(k)
                sel = self.rng.choice(unseen, size=take, replace=False)
                v[sel] += 1
                k = 0
            if k == 0:
                alloc = v.astype(np.float64)
                probs[idx] = alloc / alloc.sum()
                return probs

        base = self._normalized_means(idx)
        t0 = float(self.n.sum())
        step = int(v.sum()) + 1

        # Simulate remaining k virtual pulls with epsilon-greedy
        while k > 0:
            num = 2.0 * np.log(max(t0 + step, 2.0))
            den = np.maximum(n_sub + v, 1.0)
            scores = base + self.c * np.sqrt(num / den)

            winners = np.where(scores == scores.max())[0]
            p = np.zeros(A, dtype=np.float64)
            if winners.size == A:
                p[:] = 1.0 / A
            else:
                p[winners] = (1.0 - self.epsilon) / winners.size
                mask = np.ones(A, dtype=bool)
                mask[winners] = False
                others = np.where(mask)[0]
                if others.size > 0:
                    p[others] = self.epsilon / others.size

            i = int(self.rng.choice(A, p=p))
            v[i] += 1
            step += 1
            k -= 1

        alloc = v.astype(np.float64)
        probs[idx] = alloc / alloc.sum()
        return probs

    def decay(self, factor: float) -> None:
        """Decay accumulated statistics by factor."""
        if not (0.0 < factor <= 1.0):
            raise ValueError("factor must be in (0, 1]")

        self.divs = self.divs * factor
        one_minus_factor = 1.0 - factor

        if self.use_exponential_scaling and self.asymmetric_scaling:
            s = self.s
            with np.errstate(divide="ignore", invalid="ignore"):
                log1p_term = np.where(
                    s > 0.0,
                    s + np.log(one_minus_factor + np.exp(-s)),
                    np.log1p(one_minus_factor * np.exp(s)),
                )
                self.s = s + np.log(factor) - log1p_term

            if self.adaptive_scale and np.isfinite(self._obs_max):
                means_log = self._mean()
                mmax = float(np.max(means_log))
                om = self._obs_max
                log1p_obs = (
                    om + np.log(one_minus_factor + np.exp(-om))
                    if om > 0.0
                    else np.log1p(one_minus_factor * np.exp(om))
                )
                obs_new = om + np.log(factor) - log1p_obs
                self._obs_max = max(obs_new, mmax)
        else:
            self.s = self.s * factor
            if self.adaptive_scale and self._have_obs_range():
                means = self._mean()
                self._obs_max = max(
                    self._obs_max * factor + one_minus_factor * np.max(means),
                    np.max(means),
                )
                self._obs_min = min(
                    self._obs_min * factor + one_minus_factor * np.min(means),
                    np.min(means),
                )

    def print_summary(self) -> None:
        """Print rich table showing current bandit statistics."""
        names = self._arm_names or [str(i) for i in range(self._n_arms)]
        post = self.posterior()
        n = self.n.astype(int)
        mean = self._mean()
        if self.use_exponential_scaling:
            mean_disp = mean
            mean_label = "log mean"
        else:
            mean_disp = mean
            mean_label = "mean"
        idx = np.arange(self._n_arms)

        # Exploitation and exploration components
        exploitation = self._normalized_means(idx)
        t = float(self.n.sum())
        num = 2.0 * np.log(max(t, 2.0))
        n_sub = np.maximum(self.n[idx], 1.0)
        exploration = self.c * np.sqrt(num / n_sub)
        score = exploitation + exploration

        # Create header information
        exp_base_str = (
            f"{self.exponential_base:.3f}"
            if self.exponential_base is not None
            else "None"
        )
        header_info = (
            f"AsymmetricUCB (c={self.c:.3f}, eps={self.epsilon:.3f}, "
            f"adaptive={self.adaptive_scale}, asym={self.asymmetric_scaling}, "
            f"exp_base={exp_base_str}, shift_base={self._shift_by_baseline}, "
            f"shift_parent={self._shift_by_parent}, "
            f"log_sum={self.use_exponential_scaling})"
        )

        additional_info = []
        if self._auto_decay is not None:
            additional_info.append(f"auto_decay={self._auto_decay:.3f}")
        additional_info.append(f"baseline={self._baseline:.6f}")

        if np.isfinite(self._obs_min) and np.isfinite(self._obs_max):
            if self.use_exponential_scaling:
                obs_min = np.exp(self._obs_min)
                obs_max = np.exp(self._obs_max)
            else:
                obs_min = self._obs_min
                obs_max = self._obs_max
            rng = obs_max - obs_min
            additional_info.append(
                f"obs_range=[{obs_min:.6f},{obs_max:.6f}] (w={rng:.6f})"
            )

        # Create rich table
        table = Table(
            title=header_info,
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            width=120,
        )

        # Add columns
        table.add_column("arm", style="white", width=24)
        table.add_column("n", justify="right", style="green")
        table.add_column("div", justify="right", style="yellow")
        table.add_column(mean_label, justify="right", style="blue")
        table.add_column("exploit", justify="right", style="magenta")
        table.add_column("explore", justify="right", style="cyan")
        table.add_column("score", justify="right", style="bold white")
        table.add_column("post", justify="right", style="bright_green")

        # Add rows
        for i, name in enumerate(names):
            if isinstance(name, str):
                display_name = name.split("/")[-1][-25:]
            else:
                display_name = str(name)
            table.add_row(
                display_name,
                f"{n[i]:d}",
                f"{self.divs[i]:.3f}",
                f"{mean_disp[i]:.4f}",
                f"{exploitation[i]:.4f}",
                f"{exploration[i]:.4f}",
                f"{score[i]:.4f}",
                f"{post[i]:.4f}",
            )

        console = Console()
        console.print(table)


class FixedSampler(BanditBase):
    """
    Fixed probability sampler (no learning).

    Samples from fixed prior probabilities without any learning or decay.
    Useful as a baseline or when you want deterministic model distribution.
    """

    def __init__(
        self,
        n_arms: Optional[int] = None,
        seed: Optional[int] = None,
        prior_probs: Optional[np.ndarray] = None,
        arm_names: Optional[List[str]] = None,
        auto_decay: Optional[float] = None,
        shift_by_baseline: bool = True,
        shift_by_parent: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            n_arms=n_arms,
            seed=seed,
            arm_names=arm_names,
            auto_decay=auto_decay,
            shift_by_baseline=shift_by_baseline,
            shift_by_parent=shift_by_parent,
        )
        n = self.n_arms
        if prior_probs is None:
            self.p = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            p = np.asarray(prior_probs, dtype=np.float64)
            if p.ndim != 1 or p.size != n:
                raise ValueError("prior_probs must be length n_arms")
            if np.any(p < 0.0):
                raise ValueError("prior_probs must be >= 0")
            s = p.sum()
            if s <= 0.0:
                raise ValueError("prior_probs must sum to > 0")
            self.p = p / s

    def update_submitted(self, arm: Arm) -> float:
        return 0.0

    def update(
        self,
        arm: Arm,
        reward: Optional[float],
        baseline: Optional[float] = None,
    ) -> tuple[float, float]:
        self._maybe_decay()
        return 0.0, baseline if baseline is not None else 0.0

    def posterior(
        self,
        subset: Subset = None,
        samples: Optional[int] = None,
    ) -> np.ndarray:
        if subset is None:
            return self.p.copy()
        idx = self._resolve_subset(subset)
        probs = self.p[idx]
        s = probs.sum()
        if s <= 0.0:
            raise ValueError("subset probs sum to 0")
        probs = probs / s
        out = np.zeros(self.n_arms, dtype=np.float64)
        out[idx] = probs
        return out

    def decay(self, factor: float) -> None:
        return None

    def print_summary(self) -> None:
        names = self._arm_names or [str(i) for i in range(self._n_arms)]
        post = self.posterior()

        table = Table(
            title="FixedSampler (fixed prior probs)",
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            width=120,
        )

        table.add_column("arm", style="white", width=28)
        table.add_column("base", justify="right", style="blue")
        table.add_column("prob", justify="right", style="bright_green")

        for i, name in enumerate(names):
            if isinstance(name, str):
                display_name = name.split("/")[-1][-28:]
            else:
                display_name = str(name)
            table.add_row(
                display_name,
                f"{self.p[i]:.4f}",
                f"{post[i]:.4f}",
            )

        console = Console()
        console.print(table)
