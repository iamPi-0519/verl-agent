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
Unit tests for Goofspiel observation formatting.

Tests the format_goofspiel_observation function that normalizes raw
observations and adds legal actions.
"""

import pytest
from agent_system.environments.env_package.affine_game.envs import format_goofspiel_observation


@pytest.mark.unit
class TestFormatGoofspielObservation:
    """Test suite for format_goofspiel_observation function."""

    def test_valid_observation_formatting(self, sample_observations):
        """Test that valid observations are properly formatted."""
        raw_obs = sample_observations['raw']
        formatted = format_goofspiel_observation(raw_obs)

        # Should include legal actions section
        assert "Legal Actions:" in formatted
        # Should include choice prompt
        assert "Your choice (ID only):" in formatted
        # Should preserve key information
        assert "Current point card: 7" in formatted
        assert "P0 hand:" in formatted
        assert "P1 hand:" in formatted
        assert "Player 0:" in formatted

    def test_legal_actions_generation(self, sample_observations):
        """Test that legal actions are generated from P0 hand cards."""
        raw_obs = sample_observations['raw']
        formatted = format_goofspiel_observation(raw_obs)

        # P0 hand has cards 1-13, so legal actions should be 0-12 (card N -> ID N-1)
        assert "0 -> [P0]Bid: 1" in formatted
        assert "1 -> [P0]Bid: 2" in formatted
        assert "12 -> [P0]Bid: 13" in formatted

    def test_step_observation_with_reduced_hand(self, sample_observations):
        """Test formatting when P0 has played a card (card 7 removed)."""
        raw_obs = sample_observations['step']
        formatted = format_goofspiel_observation(raw_obs)

        # P0 hand is now [1,2,3,4,5,6,8,9,10,11,12,13] (7 removed)
        assert "Legal Actions:" in formatted
        # Card 7 should NOT appear in legal actions
        assert "6 -> [P0]Bid: 7" not in formatted
        # Other cards should still be present
        assert "0 -> [P0]Bid: 1" in formatted
        assert "7 -> [P0]Bid: 8" in formatted

    def test_invalid_observation_returns_original(self):
        """Test that invalid observations are returned unchanged."""
        invalid_obs = "This is not a valid Goofspiel observation"
        formatted = format_goofspiel_observation(invalid_obs)

        assert formatted == invalid_obs

    def test_partial_observation_returns_original(self):
        """Test that partial observations missing required fields return unchanged."""
        partial_obs = "Current point card: 7\nP0 hand: [1, 2, 3]"
        formatted = format_goofspiel_observation(partial_obs)

        # Missing required fields, should return original
        assert formatted == partial_obs

    def test_empty_observation(self):
        """Test handling of empty observation string."""
        formatted = format_goofspiel_observation("")
        assert formatted == ""

    def test_none_like_input(self):
        """Test handling of various edge case inputs."""
        # The function should handle malformed input gracefully
        assert format_goofspiel_observation("   ") == "   "

    def test_observation_preserves_win_sequence(self, sample_observations):
        """Test that win sequence comment is preserved."""
        raw_obs = sample_observations['raw']
        formatted = format_goofspiel_observation(raw_obs)

        assert "(Win sequence:" in formatted

    def test_observation_preserves_scores(self, sample_observations):
        """Test that player scores are preserved."""
        raw_obs = sample_observations['raw']
        formatted = format_goofspiel_observation(raw_obs)

        assert "Player 0:" in formatted
        assert "Player 1:" in formatted

    def test_observation_preserves_player_identity(self, sample_observations):
        """Test that 'You are Player 0.' is preserved."""
        raw_obs = sample_observations['raw']
        formatted = format_goofspiel_observation(raw_obs)

        assert "You are Player 0." in formatted

    def test_legal_actions_sorted(self, sample_observations):
        """Test that legal actions are generated in sorted order."""
        raw_obs = sample_observations['raw']
        formatted = format_goofspiel_observation(raw_obs)

        lines = formatted.split('\n')
        action_lines = [l for l in lines if '->' in l and 'Bid:' in l]

        # Extract action IDs and verify they're in order
        action_ids = []
        for line in action_lines:
            action_id = int(line.split(' -> ')[0].strip())
            action_ids.append(action_id)

        assert action_ids == sorted(action_ids)

    def test_duplicate_cards_handled(self):
        """Test handling of observations with duplicate card numbers (edge case)."""
        # This shouldn't happen in a real game, but the function should handle it
        obs_with_duplicates = """Current point card: 5
Remaining Point Cards: [1, 2, 3]
P0 hand: [1, 1, 2, 3]
P1 hand: [4, 5, 6]
(Win sequence: Higher bid wins)
Player 0: 0, Player 1: 0
You are Player 0."""

        formatted = format_goofspiel_observation(obs_with_duplicates)

        # Should deduplicate and sort
        assert "Legal Actions:" in formatted
        # Count occurrences of each action - should only appear once
        assert formatted.count("0 -> [P0]Bid: 1") == 1

    def test_exception_handling(self):
        """Test that exceptions don't crash the function."""
        # Various inputs that might cause exceptions
        test_inputs = [
            None,  # Will cause AttributeError on .search()
            123,   # Not a string
            [],    # Not a string
        ]

        for inp in test_inputs:
            try:
                # Should either return original or handle gracefully
                result = format_goofspiel_observation(inp)
            except (TypeError, AttributeError):
                # Expected for non-string inputs
                pass

    def test_malformed_numbers_in_observation(self):
        """Test handling of malformed number formats."""
        malformed_obs = """Current point card: abc
Remaining Point Cards: [not, numbers]
P0 hand: [x, y, z]
P1 hand: [1, 2, 3]
(Win sequence: Higher bid wins)
Player 0: 0, Player 1: 0
You are Player 0."""

        # Should return original since parsing will fail
        formatted = format_goofspiel_observation(malformed_obs)
        assert "Legal Actions:" not in formatted or formatted == malformed_obs


@pytest.mark.unit
class TestObservationFormattingEdgeCases:
    """Additional edge case tests for observation formatting."""

    def test_single_card_in_hand(self):
        """Test formatting when P0 has only one card left."""
        obs = """Current point card: 13
Remaining Point Cards: []
P0 hand: [5]
P1 hand: [7]
(Win sequence: Higher bid wins)
Player 0: 45, Player 1: 46
You are Player 0."""

        formatted = format_goofspiel_observation(obs)

        assert "Legal Actions:" in formatted
        assert "4 -> [P0]Bid: 5" in formatted
        # Should only have one legal action
        action_count = formatted.count(" -> [P0]Bid:")
        assert action_count == 1

    def test_all_cards_played_game_over(self, sample_observations):
        """Test handling of game over observation."""
        final_obs = sample_observations['final']
        formatted = format_goofspiel_observation(final_obs)

        # Game over observation doesn't have the required fields,
        # should return original
        assert formatted == final_obs

    def test_whitespace_variations(self):
        """Test handling of various whitespace in observations."""
        obs_with_spaces = """Current point card:    7
Remaining Point Cards:  [1,  2,  3]
P0 hand:  [1,2,3]
P1 hand: [ 4 , 5 , 6 ]
(Win sequence: Higher bid wins)
Player 0:  10 ,  Player 1:  15
You are Player 0."""

        formatted = format_goofspiel_observation(obs_with_spaces)

        # Should still parse correctly
        assert "Legal Actions:" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
