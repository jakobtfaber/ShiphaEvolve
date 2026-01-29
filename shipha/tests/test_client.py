"""
Tests for async LLM client.

Tests cover:
    - LLMClient initialization and configuration
    - Query execution with mocked litellm
    - Batch query concurrent execution
    - Bandit integration for model selection
    - Cost tracking and token counting
    - Integration tests with real APIs (marked slow)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shipha.llm import LLMClient, QueryResult, SyncLLMClient
from shipha.llm.bandit import AsymmetricUCB, FixedSampler
from shipha.llm.models import calculate_cost, get_model_pricing, REASONING_MODELS

from .conftest import skip_without_openai


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_single_model_string(self) -> None:
        """Test initialization with single model string."""
        client = LLMClient(model_names="gpt-4o", verbose=False)
        assert client.model_names == ["gpt-4o"]

    def test_multiple_models_list(self) -> None:
        """Test initialization with multiple models."""
        models = ["gpt-4o", "claude-3-5-sonnet-20241022"]
        client = LLMClient(model_names=models, verbose=False)
        assert client.model_names == models

    def test_temperature_normalization(self) -> None:
        """Test that single temperature is normalized to list."""
        client = LLMClient(temperatures=0.8, verbose=False)
        assert client.temperatures == [0.8]

    def test_default_bandit_is_fixed_sampler(self) -> None:
        """Test that default bandit is FixedSampler."""
        client = LLMClient(model_names=["a", "b"], verbose=False)
        assert isinstance(client.bandit, FixedSampler)

    def test_custom_bandit(self, ucb_bandit: AsymmetricUCB) -> None:
        """Test initialization with custom bandit."""
        client = LLMClient(
            model_names=["gpt-4o", "claude", "mini"],
            model_selection=ucb_bandit,
            verbose=False,
        )
        assert client.bandit is ucb_bandit


class TestLLMClientQuery:
    """Tests for LLMClient.query method."""

    @pytest.mark.asyncio
    async def test_query_returns_result(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that query returns QueryResult."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            result = await client.query(
                msg="Test message",
                system_msg="You are helpful",
            )
            
            assert result is not None
            assert isinstance(result, QueryResult)
            assert result.content == "def improved(x): return x * 3"

    @pytest.mark.asyncio
    async def test_query_tracks_tokens(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that query tracks token usage."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            result = await client.query(msg="Test")
            
            assert result is not None
            assert result.input_tokens == 100
            assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_query_updates_total_cost(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that query updates total cost."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            initial_cost = client.total_cost
            
            await client.query(msg="Test")
            
            assert client.total_cost > initial_cost

    @pytest.mark.asyncio
    async def test_query_increments_count(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that query increments query count."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            
            await client.query(msg="Test 1")
            await client.query(msg="Test 2")
            
            assert client.query_count == 2

    @pytest.mark.asyncio
    async def test_query_records_bandit_submission(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that query records arm submission to bandit."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            bandit = FixedSampler(n_arms=2, arm_names=["a", "b"])
            client = LLMClient(
                model_names=["a", "b"],
                model_selection=bandit,
                verbose=False,
            )
            
            # Query should complete without error - bandit.update_submitted is called
            await client.query(msg="Test")
            # FixedSampler.update_submitted returns 0.0, so we just verify no error


class TestLLMClientBatchQuery:
    """Tests for LLMClient.batch_query method."""

    @pytest.mark.asyncio
    async def test_batch_query_returns_list(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that batch_query returns list of results."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            results = await client.batch_query(
                num_samples=3,
                msg="Test",
                system_msg="Helper",
            )
            
            assert isinstance(results, list)
            assert len(results) == 3
            assert all(isinstance(r, QueryResult) for r in results)

    @pytest.mark.asyncio
    async def test_batch_query_concurrent_execution(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that batch queries execute concurrently."""
        call_times: list[float] = []
        
        async def slow_completion(*args, **kwargs):
            import time
            call_times.append(time.time())
            await asyncio.sleep(0.1)
            return mock_litellm_response
        
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = slow_completion
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            
            import time
            start = time.time()
            await client.batch_query(num_samples=3, msg="Test")
            elapsed = time.time() - start
            
            # If sequential: ~0.3s, if concurrent: ~0.1s
            assert elapsed < 0.25  # Allow some overhead

    @pytest.mark.asyncio
    async def test_batch_query_with_different_messages(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test batch query with different messages per sample."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            captured_messages: list[str] = []
            
            async def capture_completion(*args, **kwargs):
                messages = kwargs.get("messages", [])
                if messages:
                    captured_messages.append(messages[-1]["content"])
                return mock_litellm_response
            
            mock_litellm.acompletion = capture_completion
            
            client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
            await client.batch_query(
                num_samples=3,
                msg=["Msg1", "Msg2", "Msg3"],
            )
            
            assert set(captured_messages) == {"Msg1", "Msg2", "Msg3"}


class TestLLMClientBanditReward:
    """Tests for bandit reward feedback."""

    def test_update_bandit_reward(self) -> None:
        """Test that update_bandit_reward calls bandit."""
        bandit = AsymmetricUCB(n_arms=2, arm_names=["a", "b"])
        client = LLMClient(
            model_names=["a", "b"],
            model_selection=bandit,
            verbose=False,
        )
        
        # Record a submission first
        bandit.update_submitted(0)
        
        # Update with reward
        client.update_bandit_reward(
            arm_index=0,
            reward=1.0,
            parent_score=0.0,
        )
        
        # Should not raise an error

    def test_get_bandit_stats(self) -> None:
        """Test that get_bandit_stats returns string."""
        client = LLMClient(model_names=["gpt-4o"], verbose=False)
        stats = client.get_bandit_stats()
        assert isinstance(stats, str)
        assert "Bandit" in stats


class TestSyncLLMClient:
    """Tests for synchronous wrapper."""

    def test_sync_client_query(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that SyncLLMClient.query works."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = SyncLLMClient(model_names=["gpt-4o-mini"], verbose=False)
            result = client.query(msg="Test")
            
            assert result is not None
            assert isinstance(result, QueryResult)

    def test_sync_client_properties(
        self, mock_litellm_response: MagicMock
    ) -> None:
        """Test that SyncLLMClient exposes cost and count."""
        with patch("shipha.llm.client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
            
            client = SyncLLMClient(model_names=["gpt-4o-mini"], verbose=False)
            client.query(msg="Test")
            
            assert client.query_count == 1
            assert client.total_cost > 0


class TestModelPricing:
    """Tests for model pricing utilities."""

    def test_get_pricing_known_model(self) -> None:
        """Test pricing lookup for known model."""
        pricing = get_model_pricing("gpt-4o")
        assert "input_price" in pricing
        assert "output_price" in pricing
        assert pricing["input_price"] > 0

    def test_get_pricing_unknown_model(self) -> None:
        """Test fallback pricing for unknown model."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert "input_price" in pricing
        assert "output_price" in pricing

    def test_calculate_cost(self) -> None:
        """Test cost calculation."""
        input_cost, output_cost, total = calculate_cost(
            "gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        
        assert input_cost > 0
        assert output_cost > 0
        assert total == input_cost + output_cost

    def test_reasoning_models_list(self) -> None:
        """Test that reasoning models are defined."""
        assert len(REASONING_MODELS) > 0
        assert "o1" in REASONING_MODELS


# =============================================================================
# Integration Tests (require API keys)
# =============================================================================


@pytest.mark.slow
@pytest.mark.integration
class TestLLMClientIntegration:
    """Integration tests with real LLM APIs."""

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_real_openai_query(self) -> None:
        """Test real query to OpenAI API."""
        client = LLMClient(
            model_names=["gpt-4o-mini"],
            max_tokens=50,
            verbose=True,
        )
        
        result = await client.query(
            msg="What is 2+2? Reply with just the number.",
            system_msg="You are a helpful math assistant.",
        )
        
        assert result is not None
        assert "4" in result.content
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.cost > 0

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_real_batch_query(self) -> None:
        """Test real concurrent batch queries."""
        client = LLMClient(
            model_names=["gpt-4o-mini"],
            max_tokens=20,
            verbose=True,
        )
        
        results = await client.batch_query(
            num_samples=2,
            msg=["What is 1+1?", "What is 2+2?"],
            system_msg="Reply with just the number.",
        )
        
        assert len(results) == 2
        # Check both queries returned valid results
        contents = [r.content for r in results]
        assert any("2" in c for c in contents)
        assert any("4" in c for c in contents)

    @skip_without_openai
    @pytest.mark.asyncio
    async def test_real_bandit_selection(self) -> None:
        """Test bandit model selection with real API."""
        bandit = AsymmetricUCB(
            n_arms=2,
            arm_names=["gpt-4o-mini", "gpt-4o-mini"],  # Same model, testing selection
        )
        
        client = LLMClient(
            model_names=["gpt-4o-mini", "gpt-4o-mini"],
            model_selection=bandit,
            max_tokens=10,
            verbose=True,
        )
        
        result = await client.query(msg="Hi")
        
        assert result is not None
        assert result.arm_index in [0, 1]
        
        # Update bandit with reward
        client.update_bandit_reward(
            arm_index=result.arm_index,
            reward=1.0,
            parent_score=0.0,
        )
        
        # Bandit should have recorded the update
        assert sum(bandit.arm_submission_count) > 0
