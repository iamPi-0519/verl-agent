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
End-to-end tests for Affine Game environment with mock server.

Tests the complete flow from environment creation through episode completion.
"""

import pytest
import numpy as np
import os
from functools import partial

from agent_system.environments.env_package.affine_game.envs import (
    build_affine_game_envs,
    AffineGameWorker
)
from agent_system.environments.env_package.affine_game.projection import affine_game_projection
from agent_system.environments.env_manager import AffineGameEnvironmentManager
from agent_system.environments.prompts import GOOFSPIEL_SYSTEM_PROMPT


@pytest.mark.e2e
class TestFullEpisodeWithMockServer:
    """End-to-end tests for complete episode cycles."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        """Initialize Ray before tests."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)
        yield

    def test_full_episode_reset_step_close(self, mock_affine_server):
        """Test complete reset -> step -> step -> close cycle."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=1,
            group_n=1,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        # Reset
        obs_list, info_list = envs.reset()
        assert len(obs_list) == 1
        assert len(info_list) == 1
        assert "Legal Actions:" in obs_list[0]

        # Step 1
        obs_list, rewards, dones, infos = envs.step(["5"])
        assert len(obs_list) == 1
        assert len(rewards) == 1
        assert len(dones) == 1

        # Step 2
        obs_list, rewards, dones, infos = envs.step(["3"])
        assert len(obs_list) == 1

        # Close
        envs.close()

    def test_episode_completes_after_max_steps(self, mock_affine_server):
        """Test that episode ends after max_interactions."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=1,
            group_n=1,
            max_interactions=5,  # Low limit for testing
            resources_per_worker={"num_cpus": 0.1}
        )

        envs.reset()

        done = False
        step_count = 0
        while not done and step_count < 10:
            obs_list, rewards, dones, infos = envs.step(["5"])
            done = dones[0]
            step_count += 1

        # Should complete within max_interactions
        assert step_count <= 5

        envs.close()

    def test_game_completion_with_reward(self, mock_affine_server):
        """Test that game completion returns proper reward."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=1,
            group_n=1,
            max_interactions=20,
            resources_per_worker={"num_cpus": 0.1}
        )

        envs.reset()

        # Run until done (mock server ends after 13 steps)
        final_reward = 0
        for _ in range(15):
            obs_list, rewards, dones, infos = envs.step(["5"])
            if dones[0]:
                final_reward = rewards[0]
                break

        # Mock server returns reward 1.0 on completion
        assert final_reward == 1.0

        envs.close()


@pytest.mark.e2e
class TestEnvironmentManagerEpisode:
    """End-to-end tests for AffineGameEnvironmentManager."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        """Initialize Ray before tests."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)
        yield

    @pytest.fixture
    def manager_config(self, mock_affine_server):
        """Create config for environment manager."""
        from omegaconf import OmegaConf
        return OmegaConf.create({
            'env': {
                'env_name': 'affine_game',
                'seed': 42,
                'max_steps': 30,
                'history_length': 5,
                'affine_game': {
                    'server_urls': [mock_affine_server],
                    'game_name': 'goofspiel',
                    'opponent': 'mcts',
                    'timeout': 300
                }
            }
        })

    def test_full_episode_through_manager(self, mock_affine_server, manager_config):
        """Test complete episode through AffineGameEnvironmentManager."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=1,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        projection_f = partial(affine_game_projection)
        manager = AffineGameEnvironmentManager(envs, projection_f, manager_config)

        # Reset - should include system prompt
        observations, infos = manager.reset(kwargs={})
        assert len(observations['text']) == 2
        for obs in observations['text']:
            assert GOOFSPIEL_SYSTEM_PROMPT in obs

        # Step - should NOT include system prompt
        observations, rewards, dones, infos = manager.step(["5", "7"])
        for obs in observations['text']:
            assert GOOFSPIEL_SYSTEM_PROMPT not in obs

        # Verify return types
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)

        manager.close()

    def test_manager_tracks_action_validity(self, mock_affine_server, manager_config):
        """Test that manager tracks action validity correctly."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=1,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        projection_f = partial(affine_game_projection)
        manager = AffineGameEnvironmentManager(envs, projection_f, manager_config)

        manager.reset(kwargs={})

        # One valid, one invalid action
        observations, rewards, dones, infos = manager.step(["5", "invalid"])

        assert infos[0]['is_action_valid'] == 1
        assert infos[1]['is_action_valid'] == 0

        manager.close()


@pytest.mark.e2e
class TestBatchOperations:
    """End-to-end tests for batch operations with multiple environments."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        """Initialize Ray before tests."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)
        yield

    def test_multiple_environments_parallel(self, mock_affine_server):
        """Test multiple environments operating in parallel."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=4,
            group_n=1,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        obs_list, info_list = envs.reset()
        assert len(obs_list) == 4
        assert len(info_list) == 4

        # All should have observations
        for obs in obs_list:
            assert len(obs) > 0
            assert "Legal Actions:" in obs

        # Step all environments
        actions = ["5", "3", "7", "1"]
        obs_list, rewards, dones, infos = envs.step(actions)

        assert len(obs_list) == 4
        assert len(rewards) == 4
        assert len(dones) == 4
        assert len(infos) == 4

        envs.close()

    def test_group_n_creates_copies(self, mock_affine_server):
        """Test that group_n creates multiple copies per environment."""
        envs = build_affine_game_envs(
            server_urls=[mock_affine_server],
            env_num=2,
            group_n=3,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        assert envs.num_processes == 6  # 2 * 3

        obs_list, info_list = envs.reset()
        assert len(obs_list) == 6

        envs.close()


@pytest.mark.e2e
class TestWorkerDirectAccess:
    """End-to-end tests accessing AffineGameWorker directly."""

    def test_worker_episode_without_ray(self, mock_affine_server):
        """Test AffineGameWorker directly without Ray distribution."""
        worker = AffineGameWorker(
            worker_id=0,
            server_url=mock_affine_server,
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=30
        )

        # Reset
        obs, info = worker.reset(task_id=12345, seed=42)
        assert len(obs) > 0
        assert "episode_id" in info

        # Step
        obs, reward, done, info = worker.step("5")
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        # Close
        worker.close()
        assert worker.episode_id is None


@pytest.mark.requires_server
class TestLiveServerValidation:
    """Tests that require a live Affine Game server.

    Run with: pytest -m requires_server
    Set AFFINE_GAME_SERVER_URL environment variable.
    """

    @pytest.fixture
    def live_server_url(self):
        """Get live server URL from environment."""
        url = os.environ.get("AFFINE_GAME_SERVER_URL")
        if not url:
            pytest.skip("AFFINE_GAME_SERVER_URL not set")
        return url

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        """Initialize Ray before tests."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)
        yield

    def test_live_server_reset(self, live_server_url):
        """Test reset endpoint on live server."""
        worker = AffineGameWorker(
            worker_id=0,
            server_url=live_server_url,
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=30
        )

        obs, info = worker.reset(task_id=12345, seed=42)

        assert len(obs) > 0
        assert "error" not in info
        assert "episode_id" in info

        worker.close()

    def test_live_server_step(self, live_server_url):
        """Test step endpoint on live server."""
        worker = AffineGameWorker(
            worker_id=0,
            server_url=live_server_url,
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=30
        )

        worker.reset(task_id=12345, seed=42)
        obs, reward, done, info = worker.step("5")

        assert "error" not in info
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        worker.close()

    def test_live_server_full_episode(self, live_server_url):
        """Test complete episode on live server."""
        envs = build_affine_game_envs(
            server_urls=[live_server_url],
            env_num=1,
            group_n=1,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        envs.reset()

        done = False
        step_count = 0
        while not done and step_count < 30:
            obs_list, rewards, dones, infos = envs.step(["5"])
            done = dones[0]
            step_count += 1

        # Episode should complete
        assert done

        envs.close()

    def test_live_server_concurrent_workers(self, live_server_url):
        """Test multiple concurrent workers on live server."""
        envs = build_affine_game_envs(
            server_urls=[live_server_url],
            env_num=4,
            group_n=1,
            max_interactions=30,
            resources_per_worker={"num_cpus": 0.1}
        )

        obs_list, info_list = envs.reset()

        # All should succeed
        assert len(obs_list) == 4
        for info in info_list:
            assert "error" not in info

        envs.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
