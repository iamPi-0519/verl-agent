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
Shared fixtures for Affine Game environment tests.
"""

import pytest
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock


# Sample Goofspiel observations for testing
SAMPLE_RAW_OBSERVATION = """Current point card: 7
Remaining Point Cards: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
P0 hand: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
P1 hand: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
(Win sequence: Higher bid wins)
Player 0: 0, Player 1: 0
You are Player 0."""

SAMPLE_STEP_OBSERVATION = """Current point card: 3
Remaining Point Cards: [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13]
P0 hand: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
P1 hand: [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
(Win sequence: Higher bid wins)
Player 0: 7, Player 1: 0
You are Player 0."""

SAMPLE_FINAL_OBSERVATION = """Game Over!
Final Scores - Player 0: 45, Player 1: 46
You lost!"""


class MockAffineGameHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock Affine Game server."""

    # Class-level state shared across requests
    episodes = {}
    step_counts = {}

    def log_message(self, format, *args):
        # Suppress logging in tests
        pass

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode('utf-8'))

        if self.path == '/reset':
            self._handle_reset(request)
        elif self.path == '/step':
            self._handle_step(request)
        else:
            self.send_error(404, 'Not Found')

    def _handle_reset(self, request):
        task_id = request.get('task_id', 0)
        episode_id = f"episode-{task_id}-{id(self)}"

        MockAffineGameHandler.episodes[episode_id] = {
            'task_id': task_id,
            'step_count': 0,
            'done': False
        }
        MockAffineGameHandler.step_counts[episode_id] = 0

        response = {
            'result': {
                'episode_id': episode_id,
                'observation': SAMPLE_RAW_OBSERVATION
            }
        }

        self._send_json_response(response)

    def _handle_step(self, request):
        episode_id = request.get('episode_id', '')
        action = request.get('action', '')

        if episode_id not in MockAffineGameHandler.episodes:
            self.send_error(400, 'Invalid episode_id')
            return

        episode = MockAffineGameHandler.episodes[episode_id]
        episode['step_count'] += 1
        MockAffineGameHandler.step_counts[episode_id] = episode['step_count']

        # Simulate game ending after 13 steps (all cards played)
        if episode['step_count'] >= 13:
            response = {
                'result': {
                    'observation': SAMPLE_FINAL_OBSERVATION,
                    'reward': 1.0,  # Win
                    'done': True
                }
            }
        else:
            response = {
                'result': {
                    'observation': SAMPLE_STEP_OBSERVATION,
                    'reward': 0.0,
                    'done': False
                }
            }

        self._send_json_response(response)

    def _send_json_response(self, data):
        response_bytes = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response_bytes))
        self.end_headers()
        self.wfile.write(response_bytes)


@pytest.fixture(scope="session")
def mock_affine_server():
    """
    Start a mock HTTP server that simulates the Affine Game API.
    Yields the server URL.
    """
    # Find an available port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    server = HTTPServer(('localhost', port), MockAffineGameHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    yield f"http://localhost:{port}"

    server.shutdown()


@pytest.fixture
def mock_config():
    """Create a mock configuration object for environment manager tests."""
    config = MagicMock()
    config.env.affine_game.get.side_effect = lambda key, default=None: {
        'game_name': 'goofspiel',
        'server_urls': ['http://localhost:5000'],
        'opponent': 'mcts',
        'timeout': 300
    }.get(key, default)
    config.env.affine_game.game_name = 'goofspiel'
    config.env.affine_game.server_urls = ['http://localhost:5000']
    config.env.max_steps = 30
    config.env.seed = 42
    config.env.history_length = 5
    config.data.train_batch_size = 2
    config.data.val_batch_size = 1
    config.env.rollout.n = 2
    config.env.resources_per_worker = {'num_cpus': 0.1}
    return config


@pytest.fixture
def sample_observations():
    """Provide sample observations for testing."""
    return {
        'raw': SAMPLE_RAW_OBSERVATION,
        'step': SAMPLE_STEP_OBSERVATION,
        'final': SAMPLE_FINAL_OBSERVATION
    }


# Pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test (no external dependencies)")
    config.addinivalue_line("markers", "integration: mark test as an integration test (requires Ray)")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test (requires mock server)")
    config.addinivalue_line("markers", "requires_server: mark test as requiring a live Affine Game server")
