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
from typing import List, Tuple


def affine_game_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Parse LLM outputs to extract action IDs.

    Evaluation format: Model outputs ONLY the action ID (a single number).
    Extract first integer from the text.

    Args:
        actions: List of raw LLM output strings

    Returns:
        Tuple of (parsed_actions, validity_flags)
        - parsed_actions: List of extracted action strings
        - validity_flags: List of 1s (valid) or 0s (invalid)
    """
    parsed_actions = []
    valids = []

    for action in actions:
        text = action.strip()
        if text.endswith("</s>"):
            text = text[:-4].strip()

        # Extract first integer (matching evaluation behavior from affine_game.py)
        match = re.search(r"\d+", text)
        if match:
            parsed_actions.append(match.group(0))
            valids.append(1)
        else:
            # Fallback: use the text as-is
            parsed_actions.append(text)
            valids.append(0)

    return parsed_actions, valids
