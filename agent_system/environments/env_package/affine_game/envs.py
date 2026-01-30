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

import re
import random
import requests
import numpy as np
import ray


# Game task ID ranges for different games
GAMES_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}


def format_goofspiel_observation(raw_obs: str) -> str:
    """
    Parse the raw Goofspiel observation from the environment and return a
    normalized format that also includes a Legal Actions block for P0.
    If parsing fails, fall back to the original observation.

    This function matches the evaluation behavior from affine_game.py.
    """
    try:
        point_match = re.search(r"Current point card:\s*(\d+)", raw_obs)
        remaining_match = re.search(r"Remaining Point Cards:\s*([^\n]+)", raw_obs)
        p0_hand_match = re.search(r"P0 hand:\s*([^\n]+)", raw_obs)
        p1_hand_match = re.search(r"P1 hand:\s*([^\n]+)", raw_obs)
        win_seq_comment_match = re.search(r"\(Win sequence:[^\n]*\)", raw_obs)
        score_match = re.search(r"Player 0:\s*[^,\n]+,\s*Player 1:\s*[^\n]+", raw_obs)
        you_are_match = re.search(r"You are Player 0\.", raw_obs)

        if not (point_match and remaining_match and p0_hand_match and p1_hand_match and score_match and win_seq_comment_match):
            return raw_obs

        point_card = point_match.group(1).strip()
        remaining_cards = remaining_match.group(1).strip()
        p0_hand_str = p0_hand_match.group(1).strip()
        p1_hand_str = p1_hand_match.group(1).strip()
        win_seq_comment = win_seq_comment_match.group(0).strip()
        score_line = score_match.group(0).strip()
        you_are_line = you_are_match.group(0).strip() if you_are_match else "You are Player 0."

        # Build Legal Actions from P0 hand.
        p0_cards = [int(x) for x in re.findall(r"\d+", p0_hand_str)]
        p0_cards = sorted(set(p0_cards))
        legal_lines = [f"{card - 1} -> [P0]Bid: {card}" for card in p0_cards]

        lines = [
            f"Current point card: {point_card}",
            f"Remaining Point Cards: {remaining_cards}",
            f"P0 hand: {p0_hand_str}",
            f"P1 hand: {p1_hand_str}",
            "Win sequence:",
            win_seq_comment,
            score_line,
            "",
            "",
            you_are_line,
            "Legal Actions:",
            *legal_lines,
            "\nYour choice (ID only):"
        ]
        return "\n".join(lines)
    except Exception:
        # Never break training due to formatting issues
        return raw_obs


class AffineGameWorker:
    """
    Ray Actor that holds an HTTP client connection to an Affine Game environment
    and operates the environment based on method calls from the main process.
    """
    def __init__(self, worker_id, server_url, max_interactions, game_name, opponent, timeout):
        self.worker_id = worker_id
        self.server_url = server_url.rstrip('/')
        self.max_interactions = max_interactions
        self.game_name = game_name
        self.opponent = opponent
        self.timeout = timeout

        self.episode_id = None
        self.current_step_count = 0

    def reset(self, task_id, seed):
        """Reset the environment with a new task."""
        self.current_step_count = 0

        payload = {
            "task_id": task_id,
            "seed": seed,
            "opponent": self.opponent
        }

        try:
            response = requests.post(
                f"{self.server_url}/reset",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            result = data["result"]

            self.episode_id = result.get("episode_id", "")
            raw_observation = result.get("observation", "")

            # Format the observation for goofspiel
            formatted_obs = format_goofspiel_observation(raw_observation)

            info = {
                "task_id": task_id,
                "episode_id": self.episode_id,
                "game_name": self.game_name,
            }

            return formatted_obs, info

        except Exception as e:
            print(f"[AffineGameWorker {self.worker_id}] Reset failed for task_id {task_id}: {e}")
            return "", {"error": str(e), "task_id": task_id}

    def step(self, action):
        """Execute one step in the environment."""
        if self.episode_id is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1

        payload = {
            "action": action,
            "episode_id": self.episode_id
        }

        try:
            response = requests.post(
                f"{self.server_url}/step",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            result = data["result"]

            raw_observation = result.get("observation", "")
            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            # Format the observation
            formatted_obs = format_goofspiel_observation(raw_observation)

            # Check if max interactions reached
            if self.current_step_count >= self.max_interactions:
                done = True

            info = {
                "won": reward > 0 if done else False,
                "step_count": self.current_step_count,
                "episode_id": self.episode_id,
            }

            return formatted_obs, reward, done, info

        except Exception as e:
            print(f"[AffineGameWorker {self.worker_id}] Step failed: {e}")
            return "", -0.01, True, {"error": str(e), "won": False, "step_count": self.current_step_count}

    def close(self):
        """Close the environment (cleanup)."""
        self.episode_id = None
        self.current_step_count = 0


class AffineGameEnvs:
    """
    A Ray-based distributed wrapper for Affine Game environments.
    - Creates multiple Ray actors, each holding an HTTP client to an environment server.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """
    def __init__(
        self,
        server_urls,
        game_name,
        max_interactions,
        seed,
        env_num,
        group_n,
        opponent,
        resources_per_worker,
        timeout=300
    ):
        super().__init__()

        self.server_urls = server_urls if isinstance(server_urls, list) else [server_urls]
        self.game_name = game_name
        self.max_interactions = max_interactions
        self.seed = seed
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.opponent = opponent
        self.timeout = timeout

        # Get task ID range for the selected game
        if game_name not in GAMES_TO_TASK_ID_RANGE:
            raise ValueError(f"Unknown game: {game_name}. Available games: {list(GAMES_TO_TASK_ID_RANGE.keys())}")
        self.task_id_range = GAMES_TO_TASK_ID_RANGE[game_name]

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors (workers)
        env_worker = ray.remote(**resources_per_worker)(AffineGameWorker)
        self.workers = []

        for i in range(self.num_processes):
            # Distribute workers across available servers
            server_url = self.server_urls[i % len(self.server_urls)]
            worker = env_worker.remote(
                worker_id=i,
                server_url=server_url,
                max_interactions=self.max_interactions,
                game_name=self.game_name,
                opponent=self.opponent,
                timeout=self.timeout
            )
            self.workers.append(worker)

    def reset(self):
        """
        Reset all worker environments simultaneously,
        returning each environment's initial observation and info.
        """
        # Generate random task IDs within the game's range
        task_ids = []
        for _ in range(self.env_num):
            task_id = random.randint(self.task_id_range[0], self.task_id_range[1])
            task_ids.append(task_id)

        # Repeat task_ids group_n times (same task for each group)
        task_ids = np.repeat(task_ids, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(task_ids[i], self.seed)
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        obs_list = []
        info_list = []

        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def step(self, actions):
        """
        Execute actions on all environments.

        Args:
            actions: List of actions, one per environment

        Returns:
            observations, rewards, dones, infos
        """
        assert len(actions) == self.num_processes, \
            f"Expected {self.num_processes} actions, got {len(actions)}"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)

        # Wait for all workers to close
        ray.get(futures)

        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass


def build_affine_game_envs(
    server_urls,
    game_name="goofspiel",
    max_interactions=30,
    seed=42,
    env_num=1,
    group_n=1,
    opponent="mcts",
    resources_per_worker=None,
    timeout=300
):
    """
    Factory function to build Affine Game environments.

    Args:
        server_urls: List of server URLs or single URL string
        game_name: Name of the game (e.g., "goofspiel")
        max_interactions: Maximum steps per episode
        seed: Random seed
        env_num: Number of unique environments
        group_n: Number of copies per environment (for multiple rollouts)
        opponent: Opponent type ("mcts" or "random")
        resources_per_worker: Ray resource configuration per worker
        timeout: HTTP request timeout in seconds

    Returns:
        AffineGameEnvs instance
    """
    if resources_per_worker is None:
        resources_per_worker = {"num_cpus": 0.1}

    return AffineGameEnvs(
        server_urls=server_urls,
        game_name=game_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        opponent=opponent,
        resources_per_worker=resources_per_worker,
        timeout=timeout
    )
