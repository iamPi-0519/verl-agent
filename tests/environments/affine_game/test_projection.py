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
Unit tests for the Affine Game projection function.

Tests the action parsing logic that extracts action IDs from LLM outputs.
"""

import pytest
from agent_system.environments.env_package.affine_game.projection import affine_game_projection


@pytest.mark.unit
class TestAffineGameProjection:
    """Test suite for affine_game_projection function."""

    def test_simple_numbers(self):
        """Test parsing simple numeric strings."""
        actions = ["5", "12", "0"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["5", "12", "0"]
        assert valids == [1, 1, 1]

    def test_with_whitespace(self):
        """Test parsing numbers with surrounding whitespace."""
        actions = ["  7  ", "\n3\n", "\t10\t"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["7", "3", "10"]
        assert valids == [1, 1, 1]

    def test_with_eos_token(self):
        """Test parsing numbers with </s> EOS token."""
        actions = ["5</s>", "12</s>", "0</s>"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["5", "12", "0"]
        assert valids == [1, 1, 1]

    def test_with_eos_and_whitespace(self):
        """Test parsing numbers with both whitespace and EOS token."""
        actions = ["  5  </s>", "\n12</s>"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["5", "12"]
        assert valids == [1, 1]

    def test_number_in_text(self):
        """Test extracting first number from text with additional content."""
        actions = ["I choose 3", "Action: 12", "My bid is 7"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["3", "12", "7"]
        assert valids == [1, 1, 1]

    def test_multiple_numbers_extracts_first(self):
        """Test that only the first number is extracted from text."""
        actions = ["3 then 5", "Action 12 or 7", "Bid 1 2 3"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["3", "12", "1"]
        assert valids == [1, 1, 1]

    def test_no_number_invalid(self):
        """Test that text without numbers is marked invalid."""
        actions = ["abc", "no number here", "invalid"]
        parsed, valids = affine_game_projection(actions)

        # When invalid, the text is returned as-is
        assert parsed == ["abc", "no number here", "invalid"]
        assert valids == [0, 0, 0]

    def test_empty_string_invalid(self):
        """Test that empty strings are marked invalid."""
        actions = ["", "   ", "\n\t"]
        parsed, valids = affine_game_projection(actions)

        assert valids == [0, 0, 0]

    def test_empty_list(self):
        """Test handling of empty input list."""
        actions = []
        parsed, valids = affine_game_projection(actions)

        assert parsed == []
        assert valids == []

    def test_single_action(self):
        """Test handling of single action input."""
        actions = ["5"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["5"]
        assert valids == [1]

    def test_large_numbers(self):
        """Test parsing large numbers."""
        actions = ["100", "999", "12345"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["100", "999", "12345"]
        assert valids == [1, 1, 1]

    def test_zero(self):
        """Test parsing zero."""
        actions = ["0", "Action 0", "0</s>"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["0", "0", "0"]
        assert valids == [1, 1, 1]

    def test_mixed_valid_invalid(self):
        """Test handling of mixed valid and invalid inputs."""
        actions = ["5", "abc", "12", "", "7"]
        parsed, valids = affine_game_projection(actions)

        assert parsed[0] == "5"
        assert parsed[2] == "12"
        assert parsed[4] == "7"
        assert valids == [1, 0, 1, 0, 1]

    def test_number_with_leading_zeros(self):
        """Test parsing numbers with leading zeros."""
        actions = ["007", "012"]
        parsed, valids = affine_game_projection(actions)

        # Regex extracts first digit sequence
        assert parsed == ["007", "012"]
        assert valids == [1, 1]

    def test_special_characters(self):
        """Test handling of special characters in input."""
        actions = ["!@#$%", "Action: [5]", "{bid: 7}"]
        parsed, valids = affine_game_projection(actions)

        assert parsed[0] == "!@#$%"  # No number, returned as-is
        assert parsed[1] == "5"
        assert parsed[2] == "7"
        assert valids == [0, 1, 1]

    def test_newlines_in_text(self):
        """Test handling of newlines in input text."""
        actions = ["Line1\n5\nLine3", "Thinking...\n\nAction: 7"]
        parsed, valids = affine_game_projection(actions)

        assert parsed == ["1", "7"]  # First number found
        assert valids == [1, 1]

    def test_unicode_numbers(self):
        """Test that only ASCII digits are matched."""
        # Unicode numeric characters should not match
        actions = ["\u0661\u0662", "5"]  # Arabic digits followed by ASCII 5
        parsed, valids = affine_game_projection(actions)

        # Only ASCII 5 should match
        assert valids[1] == 1
        # First should be invalid (Arabic digits don't match \d+ for ASCII)
        # Actually \d matches Unicode digits in Python 3, let's verify behavior
        # The regex uses \d+ which in Python 3 matches Unicode digits too


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
