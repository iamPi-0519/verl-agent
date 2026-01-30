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
Affine Game prompts and templates.

"""

# System prompt for goofspiel (matches evaluation exactly)
GOOFSPIEL_SYSTEM_PROMPT = '''You are playing goofspiel.

# Game Rules
GOOFSPIEL RULES:
Setup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.
Goal: Win the most points by bidding on prize cards.

Each turn:
1. Reveal top prize card (worth its face value in points)
2. Players simultaneously play one bid card from their hand
3. Highest bidder wins the prize card (adds its value to score)
4. If bids tie, prize card is discarded (no one gets points)

Winning: Player with most points after all rounds wins.


# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.

Examples:
- For action "0 -> roll": respond "0"
- For action "89 -> a3": respond "89"
'''.strip()

# Template for initial observation (first turn includes system prompt)
AFFINE_GAME_TEMPLATE_INIT = "{system_prompt}\n\n{formatted_observation}"

# Template for subsequent observations (just the formatted observation)
AFFINE_GAME_TEMPLATE_STEP = "{formatted_observation}"
