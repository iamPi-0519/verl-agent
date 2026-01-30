# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for AffineGameEnvs with Ray.

Tests the distributed environment wrapper that manages multiple workers.
"""

import pytest
from unittest.mock import patch, MagicMock

from agent_system.environments.env_package.affine_game.envs import (
    AffineGameEnvs,
    build_affine_game_envs,
    GAMES_TO_TASK_ID_RANGE
)


@pytest.mark.unit
class TestGamesToTaskIdRange:
    """Test suite for GAMES_TO_TASK_ID_RANGE constant."""

    def test_goofspiel_range(self):
        """Test Goofspiel task ID range."""
        assert "goofspiel" in GAMES_TO_TASK_ID_RANGE
        start, end = GAMES_TO_TASK_ID_RANGE["goofspiel"]
        assert start == 0
        assert end == 99999999

    def test_all_expected_games_present(self):
        """Test that all expected games are in the mapping."""
        expected_games = [
            "goofspiel",
            "liars_dice",
            "leduc_poker",
            "gin_rummy",
            "othello",
            "backgammon",
            "hex",
            "clobber"
        ]
        for game in expected_games:
            assert game in GAMES_TO_TASK_ID_RANGE, f"Missing game: {game}"

    def test_ranges_dont_overlap(self):
        """Test that game task ID ranges don't overlap."""
        ranges = list(GAMES_TO_TASK_ID_RANGE.values())
        for i, (start1, end1) in enumerate(ranges):
            for j, (start2, end2) in enumerate(ranges):
                if i != j:
                    # Check no overlap
                    assert end1 < start2 or end2 < start1, \
                        f"Ranges overlap: ({start1}, {end1}) and ({start2}, {end2})"


@pytest.mark.unit
class TestAffineGameEnvsValidation:
    """Test validation logic in AffineGameEnvs (no Ray required)."""

    def test_invalid_game_name_raises_error(self):
        """Test that invalid game name raises ValueError."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote'):
                    with pytest.raises(ValueError, match="Unknown game"):
                        AffineGameEnvs(
                            server_urls=["http://localhost:5000"],
                            game_name="invalid_game",
                            max_interactions=30,
                            seed=42,
                            env_num=1,
                            group_n=1,
                            opponent="mcts",
                            resources_per_worker={"num_cpus": 0.1}
                        )

    def test_server_urls_string_to_list(self):
        """Test that single server URL string is converted to list."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote') as mock_remote:
                    mock_remote.return_value = MagicMock()

                    envs = AffineGameEnvs(
                        server_urls="http://localhost:5000",  # String, not list
                        game_name="goofspiel",
                        max_interactions=30,
                        seed=42,
                        env_num=1,
                        group_n=1,
                        opponent="mcts",
                        resources_per_worker={"num_cpus": 0.1}
                    )

                    assert isinstance(envs.server_urls, list)
                    assert envs.server_urls == ["http://localhost:5000"]


@pytest.mark.unit
class TestAffineGameEnvsNumProcesses:
    """Test num_processes calculation."""

    def test_num_processes_calculation(self):
        """Test that num_processes equals env_num * group_n."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote') as mock_remote:
                    mock_remote.return_value = MagicMock()

                    envs = AffineGameEnvs(
                        server_urls=["http://localhost:5000"],
                        game_name="goofspiel",
                        max_interactions=30,
                        seed=42,
                        env_num=4,
                        group_n=3,
                        opponent="mcts",
                        resources_per_worker={"num_cpus": 0.1}
                    )

                    assert envs.num_processes == 12  # 4 * 3

    def test_single_env_single_group(self):
        """Test with single environment and single group."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote') as mock_remote:
                    mock_remote.return_value = MagicMock()

                    envs = AffineGameEnvs(
                        server_urls=["http://localhost:5000"],
                        game_name="goofspiel",
                        max_interactions=30,
                        seed=42,
                        env_num=1,
                        group_n=1,
                        opponent="mcts",
                        resources_per_worker={"num_cpus": 0.1}
                    )

                    assert envs.num_processes == 1


@pytest.mark.unit
class TestBuildAffineGameEnvs:
    """Test the build_affine_game_envs factory function."""

    def test_default_parameters(self):
        """Test factory function uses correct defaults."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote') as mock_remote:
                    mock_remote.return_value = MagicMock()

                    envs = build_affine_game_envs(
                        server_urls=["http://localhost:5000"]
                    )

                    assert envs.game_name == "goofspiel"
                    assert envs.max_interactions == 30
                    assert envs.seed == 42
                    assert envs.env_num == 1
                    assert envs.group_n == 1
                    assert envs.opponent == "mcts"
                    assert envs.timeout == 300

    def test_custom_parameters(self):
        """Test factory function accepts custom parameters."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote') as mock_remote:
                    mock_remote.return_value = MagicMock()

                    envs = build_affine_game_envs(
                        server_urls=["http://localhost:8080"],
                        game_name="liars_dice",
                        max_interactions=50,
                        seed=123,
                        env_num=5,
                        group_n=2,
                        opponent="random",
                        timeout=60
                    )

                    assert envs.game_name == "liars_dice"
                    assert envs.max_interactions == 50
                    assert envs.seed == 123
                    assert envs.env_num == 5
                    assert envs.group_n == 2
                    assert envs.opponent == "random"
                    assert envs.timeout == 60

    def test_default_resources_per_worker(self):
        """Test that default resources_per_worker is applied."""
        with patch('ray.is_initialized', return_value=True):
            with patch('ray.init'):
                with patch('ray.remote') as mock_remote:
                    mock_remote.return_value = MagicMock()

                    # Not passing resources_per_worker, should use default
                    envs = build_affine_game_envs(
                        server_urls=["http://localhost:5000"]
                    )

                    # The function should have been called (envs created successfully)
                    assert envs is not None


@pytest.mark.integration
class TestAffineGameEnvsWithRay:
    """Integration tests that require Ray to be running."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        """Initialize Ray before tests and shutdown after."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)
        yield
        # Don't shutdown - let other tests reuse

    def test_worker_distribution_single_server(self, mock_affine_server):
        """Test that workers are created for single server."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=2,
            resources_per_worker={"num_cpus": 0.1}
        )

        assert len(envs.workers) == 4  # 2 * 2
        envs.close()

    def test_worker_distribution_multiple_servers(self, mock_affine_server):
        """Test that workers are distributed across multiple servers."""
        # Use the same mock server URL twice to simulate multiple servers
        server_urls = [mock_affine_server, mock_affine_server]

        envs = build_affine_game_envs(
            server_urls=server_urls,
            env_num=2,
            group_n=2,
            resources_per_worker={"num_cpus": 0.1}
        )

        assert len(envs.workers) == 4
        envs.close()

    def test_reset_returns_correct_structure(self, mock_affine_server):
        """Test that reset returns observations and infos for all workers."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=1,
            resources_per_worker={"num_cpus": 0.1}
        )

        obs_list, info_list = envs.reset()

        assert len(obs_list) == 2
        assert len(info_list) == 2

        # Each observation should be formatted
        for obs in obs_list:
            assert "Legal Actions:" in obs

        # Each info should have required keys
        for info in info_list:
            assert 'episode_id' in info
            assert 'game_name' in info

        envs.close()

    def test_step_requires_correct_action_count(self, mock_affine_server):
        """Test that step requires actions for all environments."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=1,
            resources_per_worker={"num_cpus": 0.1}
        )

        envs.reset()

        # Providing wrong number of actions should raise
        with pytest.raises(AssertionError):
            envs.step(["5"])  # Only 1 action, need 2

        envs.close()

    def test_step_returns_correct_structure(self, mock_affine_server):
        """Test that step returns correct data structure."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=1,
            resources_per_worker={"num_cpus": 0.1}
        )

        envs.reset()
        obs_list, reward_list, done_list, info_list = envs.step(["5", "5"])

        assert len(obs_list) == 2
        assert len(reward_list) == 2
        assert len(done_list) == 2
        assert len(info_list) == 2

        envs.close()

    def test_close_kills_workers(self, mock_affine_server):
        """Test that close properly shuts down Ray actors."""
        import ray

        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=1,
            resources_per_worker={"num_cpus": 0.1}
        )

        workers = envs.workers.copy()
        envs.close()

        # Workers should be killed (checking might raise)
        for worker in workers:
            try:
                ray.get(worker.close.remote(), timeout=1)
            except ray.exceptions.RayActorError:
                pass  # Expected - actor was killed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
