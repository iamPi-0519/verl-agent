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
Integration tests for AffineGameEnvironmentManager.

Tests the environment manager that handles system prompts, observation
formatting, and data flow.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from functools import partial

from agent_system.environments.env_manager import AffineGameEnvironmentManager
from agent_system.environments.prompts import GOOFSPIEL_SYSTEM_PROMPT
from agent_system.environments.env_package.affine_game.projection import affine_game_projection


@pytest.fixture
def mock_envs(sample_observations):
    """Create mock environments that return sample observations."""
    mock = MagicMock()
    mock.reset.return_value = (
        [sample_observations['raw'], sample_observations['raw']],
        [
            {'episode_id': 'ep1', 'game_name': 'goofspiel', 'task_id': 1},
            {'episode_id': 'ep2', 'game_name': 'goofspiel', 'task_id': 2}
        ]
    )
    mock.step.return_value = (
        [sample_observations['step'], sample_observations['step']],
        [0.0, 0.0],
        [False, False],
        [
            {'won': False, 'step_count': 1, 'episode_id': 'ep1'},
            {'won': False, 'step_count': 1, 'episode_id': 'ep2'}
        ]
    )
    return mock


@pytest.fixture
def env_manager(mock_envs, mock_config):
    """Create AffineGameEnvironmentManager with mocks."""
    projection_f = partial(affine_game_projection)
    return AffineGameEnvironmentManager(mock_envs, projection_f, mock_config)


@pytest.mark.unit
class TestAffineGameEnvironmentManagerReset:
    """Test suite for AffineGameEnvironmentManager.reset() method."""

    def test_reset_includes_system_prompt(self, env_manager):
        """Test that first observation contains system prompt."""
        observations, infos = env_manager.reset(kwargs={})

        text_obs = observations['text']
        assert len(text_obs) == 2

        # Each observation should start with system prompt
        for obs in text_obs:
            assert GOOFSPIEL_SYSTEM_PROMPT in obs

    def test_reset_returns_correct_structure(self, env_manager):
        """Test that reset returns correct observation structure."""
        observations, infos = env_manager.reset(kwargs={})

        assert 'text' in observations
        assert 'image' in observations
        assert 'anchor' in observations
        assert observations['image'] is None
        assert isinstance(observations['anchor'], list)

    def test_reset_initializes_memory(self, env_manager):
        """Test that reset initializes the memory buffer."""
        env_manager.reset(kwargs={})

        # Memory should be reset with correct batch size
        assert len(env_manager.memory) == 2


@pytest.mark.unit
class TestAffineGameEnvironmentManagerStep:
    """Test suite for AffineGameEnvironmentManager.step() method."""

    def test_step_excludes_system_prompt(self, env_manager):
        """Test that subsequent observations do NOT include system prompt."""
        env_manager.reset(kwargs={})
        observations, rewards, dones, infos = env_manager.step(["5", "7"])

        text_obs = observations['text']

        # Step observations should NOT have system prompt
        for obs in text_obs:
            assert GOOFSPIEL_SYSTEM_PROMPT not in obs

    def test_step_returns_numpy_arrays(self, env_manager):
        """Test that rewards and dones are numpy arrays."""
        env_manager.reset(kwargs={})
        observations, rewards, dones, infos = env_manager.step(["5", "7"])

        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)

    def test_step_adds_action_validity_to_infos(self, env_manager):
        """Test that is_action_valid is added to infos."""
        env_manager.reset(kwargs={})
        observations, rewards, dones, infos = env_manager.step(["5", "7"])

        for info in infos:
            assert 'is_action_valid' in info
            assert isinstance(info['is_action_valid'], np.ndarray)

    def test_step_with_valid_actions(self, env_manager):
        """Test step with valid action inputs."""
        env_manager.reset(kwargs={})
        observations, rewards, dones, infos = env_manager.step(["5", "7"])

        # Both actions are valid numbers
        for info in infos:
            assert info['is_action_valid'] == 1

    def test_step_with_invalid_actions(self, env_manager):
        """Test step with invalid action inputs."""
        env_manager.reset(kwargs={})
        observations, rewards, dones, infos = env_manager.step(["abc", ""])

        # Both actions are invalid (no numbers)
        for info in infos:
            assert info['is_action_valid'] == 0

    def test_step_stores_in_memory(self, env_manager):
        """Test that step stores observation-action pairs in memory."""
        env_manager.reset(kwargs={})
        env_manager.step(["5", "7"])

        # Memory should have one entry per environment
        assert len(env_manager.memory[0]) == 1
        assert len(env_manager.memory[1]) == 1


@pytest.mark.unit
class TestAffineGameEnvironmentManagerBuildTextObs:
    """Test suite for build_text_obs method."""

    def test_build_text_obs_init_true(self, env_manager, sample_observations):
        """Test build_text_obs with init=True includes system prompt."""
        text_obs = [sample_observations['raw']]
        result = env_manager.build_text_obs(text_obs, init=True)

        assert len(result) == 1
        assert GOOFSPIEL_SYSTEM_PROMPT in result[0]

    def test_build_text_obs_init_false(self, env_manager, sample_observations):
        """Test build_text_obs with init=False excludes system prompt."""
        text_obs = [sample_observations['step']]
        result = env_manager.build_text_obs(text_obs, init=False)

        assert len(result) == 1
        assert GOOFSPIEL_SYSTEM_PROMPT not in result[0]


@pytest.mark.unit
class TestAffineGameEnvironmentManagerProcessBatch:
    """Test suite for _process_batch method."""

    def test_process_batch_extracts_won(self, env_manager):
        """Test that _process_batch correctly extracts won field."""
        # Create mock batch data
        total_batch_list = [
            [
                {'active_masks': False},
                {'active_masks': True},
                {'active_masks': False}
            ]
        ]
        total_infos = [
            [
                {'won': False},
                {'won': True},
                {'won': False}
            ]
        ]
        success = {'success_rate': []}

        env_manager._process_batch(0, total_batch_list, total_infos, success)

        # Should find the last active mask (index 1) with won=True
        assert success['success_rate'] == [1.0]

    def test_process_batch_loss(self, env_manager):
        """Test _process_batch with a losing game."""
        total_batch_list = [
            [{'active_masks': True}]
        ]
        total_infos = [
            [{'won': False}]
        ]
        success = {'success_rate': []}

        env_manager._process_batch(0, total_batch_list, total_infos, success)

        assert success['success_rate'] == [0.0]

    def test_process_batch_missing_won_field(self, env_manager):
        """Test _process_batch when won field is missing."""
        total_batch_list = [
            [{'active_masks': True}]
        ]
        total_infos = [
            [{}]  # No 'won' field
        ]
        success = {'success_rate': []}

        env_manager._process_batch(0, total_batch_list, total_infos, success)

        # Should default to 0 when 'won' is missing
        assert success['success_rate'] == [0.0]


@pytest.mark.unit
class TestAffineGameEnvironmentManagerSuccessEvaluator:
    """Test suite for success_evaluator method."""

    def test_success_evaluator(self, env_manager):
        """Test the success evaluator aggregates results correctly."""
        total_batch_list = [
            [{'active_masks': True}],
            [{'active_masks': True}]
        ]
        total_infos = [
            [{'won': True}],
            [{'won': False}]
        ]

        result = env_manager.success_evaluator(
            total_batch_list=total_batch_list,
            total_infos=total_infos
        )

        assert 'success_rate' in result
        assert isinstance(result['success_rate'], np.ndarray)
        assert len(result['success_rate']) == 2
        assert result['success_rate'][0] == 1.0
        assert result['success_rate'][1] == 0.0


@pytest.mark.integration
class TestMakeEnvsAffineGame:
    """Integration tests for make_envs with affine_game environment."""

    @pytest.fixture(autouse=True)
    def setup_ray(self):
        """Initialize Ray before tests."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=4, ignore_reinit_error=True)
        yield

    def test_make_envs_creates_manager(self, mock_affine_server):
        """Test that make_envs creates AffineGameEnvironmentManager."""
        from agent_system.environments.env_manager import make_envs
        from omegaconf import OmegaConf

        config_dict = {
            'env': {
                'env_name': 'affine_game',
                'seed': 42,
                'max_steps': 30,
                'history_length': 5,
                'rollout': {'n': 1},
                'resources_per_worker': {'num_cpus': 0.1},
                'affine_game': {
                    'server_urls': [mock_affine_server],
                    'game_name': 'goofspiel',
                    'opponent': 'mcts',
                    'timeout': 300
                }
            },
            'data': {
                'train_batch_size': 2,
                'val_batch_size': 1
            }
        }
        config = OmegaConf.create(config_dict)

        train_envs, val_envs = make_envs(config)

        assert isinstance(train_envs, AffineGameEnvironmentManager)
        assert isinstance(val_envs, AffineGameEnvironmentManager)

        # Clean up
        train_envs.close()
        val_envs.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
