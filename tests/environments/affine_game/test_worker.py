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
Unit tests for AffineGameWorker with mocked HTTP requests.

Tests the worker class that communicates with the Affine Game server.
"""

import pytest
from unittest.mock import patch, MagicMock
import requests

from agent_system.environments.env_package.affine_game.envs import AffineGameWorker


@pytest.mark.unit
class TestAffineGameWorkerReset:
    """Test suite for AffineGameWorker.reset() method."""

    @patch('requests.post')
    def test_reset_success(self, mock_post, sample_observations):
        """Test successful reset with valid server response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "episode_id": "test-episode-123",
                "observation": sample_observations['raw']
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        obs, info = worker.reset(task_id=12345, seed=42)

        # Verify HTTP call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:5000/reset"
        assert call_args[1]['json']['task_id'] == 12345
        assert call_args[1]['json']['seed'] == 42
        assert call_args[1]['json']['opponent'] == "mcts"

        # Verify response parsing
        assert info['episode_id'] == "test-episode-123"
        assert info['task_id'] == 12345
        assert info['game_name'] == "goofspiel"
        assert worker.episode_id == "test-episode-123"
        assert worker.current_step_count == 0

        # Verify observation is formatted
        assert "Legal Actions:" in obs

    @patch('requests.post')
    def test_reset_http_timeout(self, mock_post):
        """Test handling of HTTP timeout during reset."""
        mock_post.side_effect = requests.Timeout("Connection timed out")

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        obs, info = worker.reset(task_id=12345, seed=42)

        # Should return empty observation and error info
        assert obs == ""
        assert "error" in info
        assert info['task_id'] == 12345

    @patch('requests.post')
    def test_reset_http_connection_error(self, mock_post):
        """Test handling of connection error during reset."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        obs, info = worker.reset(task_id=12345, seed=42)

        assert obs == ""
        assert "error" in info

    @patch('requests.post')
    def test_reset_server_error(self, mock_post):
        """Test handling of server error (500) during reset."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        obs, info = worker.reset(task_id=12345, seed=42)

        assert obs == ""
        assert "error" in info

    @patch('requests.post')
    def test_reset_clears_step_count(self, mock_post, sample_observations):
        """Test that reset clears the step counter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "episode_id": "test-123",
                "observation": sample_observations['raw']
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        # Simulate some steps were taken
        worker.current_step_count = 5

        worker.reset(task_id=12345, seed=42)

        assert worker.current_step_count == 0


@pytest.mark.unit
class TestAffineGameWorkerStep:
    """Test suite for AffineGameWorker.step() method."""

    @patch('requests.post')
    def test_step_success(self, mock_post, sample_observations):
        """Test successful step with valid server response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "observation": sample_observations['step'],
                "reward": 0.0,
                "done": False
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )
        worker.episode_id = "test-episode-123"

        obs, reward, done, info = worker.step("5")

        # Verify HTTP call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:5000/step"
        assert call_args[1]['json']['action'] == "5"
        assert call_args[1]['json']['episode_id'] == "test-episode-123"

        # Verify response
        assert reward == 0.0
        assert done is False
        assert info['step_count'] == 1
        assert worker.current_step_count == 1

    def test_step_without_reset_raises_error(self):
        """Test that step() raises RuntimeError if reset() wasn't called."""
        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        with pytest.raises(RuntimeError, match="Environment not reset"):
            worker.step("5")

    @patch('requests.post')
    def test_step_game_done(self, mock_post, sample_observations):
        """Test step when game is finished (done=True)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "observation": sample_observations['final'],
                "reward": 1.0,
                "done": True
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )
        worker.episode_id = "test-episode-123"

        obs, reward, done, info = worker.step("5")

        assert done is True
        assert reward == 1.0
        assert info['won'] is True

    @patch('requests.post')
    def test_step_max_interactions_reached(self, mock_post, sample_observations):
        """Test that done=True when max_interactions is reached."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "observation": sample_observations['step'],
                "reward": 0.0,
                "done": False
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=5,  # Low max for testing
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )
        worker.episode_id = "test-episode-123"
        worker.current_step_count = 4  # One step away from max

        obs, reward, done, info = worker.step("5")

        # Should be done due to max_interactions
        assert done is True
        assert worker.current_step_count == 5

    @patch('requests.post')
    def test_step_http_error(self, mock_post):
        """Test handling of HTTP error during step."""
        mock_post.side_effect = requests.Timeout("Connection timed out")

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )
        worker.episode_id = "test-episode-123"

        obs, reward, done, info = worker.step("5")

        # Should return error state
        assert obs == ""
        assert reward == -0.01
        assert done is True
        assert "error" in info
        assert info['won'] is False

    @patch('requests.post')
    def test_step_increments_counter(self, mock_post, sample_observations):
        """Test that step count is incremented correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "observation": sample_observations['step'],
                "reward": 0.0,
                "done": False
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )
        worker.episode_id = "test-episode-123"

        for i in range(5):
            worker.step("5")
            assert worker.current_step_count == i + 1


@pytest.mark.unit
class TestAffineGameWorkerClose:
    """Test suite for AffineGameWorker.close() method."""

    def test_close_resets_state(self):
        """Test that close() resets worker state."""
        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )
        worker.episode_id = "test-episode-123"
        worker.current_step_count = 10

        worker.close()

        assert worker.episode_id is None
        assert worker.current_step_count == 0


@pytest.mark.unit
class TestAffineGameWorkerInit:
    """Test suite for AffineGameWorker initialization."""

    def test_init_with_trailing_slash(self):
        """Test that trailing slash is stripped from server URL."""
        worker = AffineGameWorker(
            worker_id=0,
            server_url="http://localhost:5000/",
            max_interactions=30,
            game_name="goofspiel",
            opponent="mcts",
            timeout=300
        )

        assert worker.server_url == "http://localhost:5000"

    def test_init_stores_parameters(self):
        """Test that all initialization parameters are stored."""
        worker = AffineGameWorker(
            worker_id=5,
            server_url="http://localhost:8080",
            max_interactions=50,
            game_name="liars_dice",
            opponent="random",
            timeout=120
        )

        assert worker.worker_id == 5
        assert worker.server_url == "http://localhost:8080"
        assert worker.max_interactions == 50
        assert worker.game_name == "liars_dice"
        assert worker.opponent == "random"
        assert worker.timeout == 120
        assert worker.episode_id is None
        assert worker.current_step_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
