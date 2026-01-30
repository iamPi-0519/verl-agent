# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`verl-agent` is an RL framework for training large language model (LLM) agents, extending the veRL framework. It implements Group-in-Group Policy Optimization (GiGPO) and supports multi-turn, long-horizon agent-environment interactions with step-independent rollouts.

Key architectural innovation: Unlike concatenating full histories, `verl-agent` uses **step-independent multi-turn rollouts** with customizable per-step input structures and memory management, enabling scalable training for tasks requiring 30-50+ steps.

## Environment Architecture

**New Architecture (Docker-based)**: The codebase is transitioning to a Docker-based environment architecture where environments are deployed as containerized services accessed via HTTP APIs. This is the **preferred approach for all new environments**.

**Reference Implementation**: `agent_system/environments/env_package/affine_game/` - This is an experimental implementations and might contains multiple bugs which we will fix if any.

### Docker-Based Environment Pattern
Environments run in Docker containers and expose HTTP APIs with endpoints:
- `POST /create` - Create new environment instance, returns `env_id`
- `POST /step` - Execute action, returns `(observation, reward, done, info)`
- `GET /get` - Get current state
- `POST /reset` - Reset environment to initial state

Client-side components in `verl-agent`:
- **Worker** (`envs.py`): Ray actor managing HTTP connections to environment server
- **Projection** (`projection.py`): Parses LLM outputs into valid action formats
- **Prompts** (`prompts/<env_name>.py`): System prompts and observation templates
- **Environment Manager** (`env_manager.py`): Integrates memory, formatting, and environment interaction
- **Tests** (`tests/environments/<env_name>/`): Comprehensive unit, integration, and e2e tests

## Installation & Setup

### Core Installation
```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### For Docker-Based Environments (Affine Game, future environments)
1. Deploy the environment server (Docker container)
2. Configure server URL in training config: `env.affine_game.server_urls`
3. No additional dependencies needed in verl-agent

### For Legacy Environments (see "Legacy Environments" section below)
Legacy environments (ALFWorld, WebShop, Search, etc.) require complex environment-specific installation. These are being phased out in favor of Docker-based environments.

## Running Tests

### Test Suite Pattern (for Docker-based environments)
All new environments should follow the comprehensive test suite pattern from Affine Game:

```bash
# Install test dependencies
pip install pytest pre-commit py-spy

# Run all tests for an environment
pytest tests/environments/affine_game/ -v

# Run by test category (recommended during development)
pytest tests/environments/affine_game/ -v -m unit          # Fast unit tests, no dependencies
pytest tests/environments/affine_game/ -v -m integration   # Requires Ray initialization
pytest tests/environments/affine_game/ -v -m e2e           # End-to-end with mock HTTP server
pytest tests/environments/affine_game/ -v -m requires_server  # Requires actual environment server

# Run with coverage
pytest tests/environments/affine_game/ -v --cov=agent_system.environments.env_package.affine_game
```

**Test markers** (use `@pytest.mark.<marker>` to categorize tests):
- `unit` - Fast, isolated tests with no external dependencies
- `integration` - Tests requiring Ray or other internal dependencies
- `e2e` - Full integration tests with mock HTTP server
- `requires_server` - Tests requiring actual environment server (optional, set `AFFINE_GAME_SERVER_URL` env var)

**Test file structure** (reference: `tests/environments/affine_game/`):
- `conftest.py` - Shared fixtures, mock HTTP server
- `test_projection.py` - Unit tests for action parsing
- `test_observation_formatting.py` - Unit tests for observation formatting
- `test_worker.py` - Unit tests for HTTP worker with mocked requests
- `test_envs.py` - Integration tests for Ray-distributed environments
- `test_env_manager.py` - Integration tests for environment manager
- `test_e2e.py` - End-to-end tests with mock server

## Training & Execution

### Main Entry Points
- **RL Training**: `python3 -m verl.trainer.main_ppo` with Hydra config
- **Generation**: `python3 -m verl.trainer.main_generation`
- **Evaluation**: `python3 -m verl.trainer.main_eval`

### Configuration
Training uses Hydra configuration files in `verl/trainer/config/`:
- `ppo_trainer.yaml` - Main PPO/GiGPO/GRPO training config (despite name, handles all algorithms)
- `generation.yaml` - Generation config
- `evaluation.yaml` - Evaluation config

### Training Script Pattern
Example training scripts are in `examples/<algorithm>_trainer/` directory:

```bash
# GiGPO (recommended for long-horizon tasks)
bash examples/gigpo_trainer/run_<env_name>.sh

# Other algorithms
bash examples/grpo_trainer/run_<env_name>.sh
bash examples/ppo_trainer/run_<env_name>.sh
bash examples/rloo_trainer/run_<env_name>.sh
bash examples/dapo_trainer/run_<env_name>.sh

# LoRA training (reduced GPU requirements)
bash examples/gigpo_trainer/run_alfworld_lora.sh
```

**Key configuration for Docker-based environments**:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    env.env_name=affine_game/goofspiel \
    env.affine_game.server_urls=['http://localhost:5000'] \
    env.affine_game.game_name=goofspiel \
    env.affine_game.opponent=mcts \
    env.rollout.n=8 \
    # ... other config
```

### Data Preparation
**For Docker-based environments**: No data preparation needed. Environment server handles all game logic.

**For dataset-driven tasks** (e.g., Search-R1): Tasks are loaded from datasets and passed via `env_kwargs`:
```bash
python examples/data_preprocess/preprocess_search_r1_dataset.py
```

Agent inputs come from environment feedback via `env.step()`. Most environments use minimal data prep to indicate modality ("text" or "<image>").

## Architecture Overview

### Core Components

**1. Environment System** (`agent_system/environments/`)

Modern (Docker-based) architecture:
- **HTTP Workers** (`env_package/<env_name>/envs.py`): Ray actors managing HTTP connections to environment servers
  - Each worker maintains connection to Docker-based environment server
  - Handles CREATE, STEP, RESET API calls
  - Returns vectorized observations, rewards, dones for parallel training
- **Projection Functions** (`env_package/<env_name>/projection.py`): Parse LLM free-form text outputs into structured actions
  - Use regex/parsing to extract action IDs or structured formats
  - Return validity flags for invalid action handling
- **Environment Managers** (`env_manager.py`): Orchestrate memory, observation formatting, and environment interaction
  - Extend `EnvironmentManagerBase`
  - `reset()`: Initialize memory, format first observation (includes system prompt)
  - `step()`: Execute action, update memory, format next observation (no system prompt)
  - `build_text_obs()`: Construct agent input from current observation + memory context
- **Prompts** (`prompts/<env_name>.py`): Task-specific system prompts and observation templates
  - System prompt: Game rules, objectives, output format
  - Init template: System prompt + first observation
  - Step template: Subsequent observations only

**2. Memory System** (`agent_system/memory/`)
- `SimpleMemory`: Default implementation for managing interaction history
- `SearchMemory`: Specialized for search tasks
- Customizable for dynamic summarization, selective retention, external knowledge
- Invoked by `env_manager.build_text_obs()` to construct step-wise inputs

**3. Multi-Turn Rollout** (`agent_system/multi_turn_rollout/`)
- `rollout_loop.py`: Core rollout logic for multi-step agent-environment interactions
- `TrajectoryCollector`: Processes observations and organizes data for model input
- **Step-independent design**: Each step has constant context length (observation + memory summary), avoiding full history concatenation

**4. Training System** (`verl/trainer/`)
- `main_ppo.py`: Main training entry point (handles PPO, GiGPO, GRPO, RLOO, DAPO)
- `ppo/ray_trainer.py`: Ray-based distributed trainer (`RayPPOTrainer`)
- `ppo/core_algos.py`: Core RL algorithms (GiGPO, GRPO, etc.)
- `ppo/reward.py`: Reward computation and management

**5. Models & Workers** (`verl/models/`, `verl/workers/`)
- Model wrappers for LLMs (Qwen3, Qwen3-VL, Qwen2.5, LLaMA3.2, etc.)
- FSDP for training, vLLM/sglang for efficient inference
- Actor, critic, reference model management

### Key Data Flow (Docker-based environments)

1. **Environment Reset**:
   - `env_manager.reset()` → HTTP worker `POST /create` to Docker server
   - Initialize memory
   - Format first observation: `TEMPLATE_INIT.format(system_prompt=..., formatted_observation=...)`
   - Return to agent

2. **Rollout Loop**:
   - LLM generates action text from observation
   - `projection_f()` parses LLM text → extract action ID/format
   - HTTP worker `POST /step` to Docker server with action
   - Server returns `(observation, reward, done, info)`
   - Memory stores step information
   - `build_text_obs()` constructs next input: `TEMPLATE_STEP.format(formatted_observation=...)` + memory context
   - Repeat until done

3. **Training**: Collected trajectories → Ray-distributed training (PPO/GiGPO/GRPO) → model update

### Group Environments
`verl-agent` supports "group environments" where environments within a group share identical initial states during `reset()`. Useful for algorithms like GRPO/DAPO requiring multiple rollouts per state. Configure via `env.rollout.n` in config.

## Adding New Docker-Based Environments

Use the Affine Game implementation as your template. The architecture cleanly separates environment logic (runs in Docker) from agent-side integration (in verl-agent).

### 1. Environment Server (Docker side)
Deploy a containerized environment that exposes HTTP API endpoints:
- `POST /create` - Create environment instance, return `env_id`
- `POST /step` - Execute action, return `(observation, reward, done, info)`
- `GET /get` - Get current state
- `POST /reset` - Reset environment

### 2. Client-Side Integration (verl-agent)

**a) Create environment package** in `agent_system/environments/env_package/<env_name>/`:

```python
# __init__.py - Export public interface
from .projection import <env_name>_projection
from .envs import build_<env_name>_envs

# projection.py - Parse LLM outputs to valid actions
def <env_name>_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Extract action IDs/formats from LLM outputs.
    Returns: (parsed_actions, validity_flags)
    """
    # Use regex to extract structured actions from free-form LLM text
    # Return validity flag (1=valid, 0=invalid) for each action

# envs.py - Ray worker managing HTTP connections
class <EnvName>Worker:
    """Ray actor managing HTTP connection to environment server."""
    def __init__(self, worker_id, server_url, ...):
        self.server_url = server_url
        # Initialize HTTP client

    def reset(self, ...):
        # POST to /create or /reset endpoint
        # Format observation for agent

    def step(self, action):
        # POST to /step endpoint with action
        # Return (observation, reward, done, info)

def build_<env_name>_envs(server_urls, num_workers, ...):
    """Create Ray-distributed workers, return wrapper with vectorized interface."""
    workers = [<EnvName>Worker.remote(...) for _ in range(num_workers)]
    return <EnvName>Envs(workers)
```

**b) Define prompts** in `agent_system/environments/prompts/<env_name>.py`:

```python
# System prompt explaining game rules and output format
<ENV_NAME>_SYSTEM_PROMPT = '''Rules, objectives, constraints...
Output Format: Specify exact format for LLM actions...'''

# Template for first observation (includes system prompt)
<ENV_NAME>_TEMPLATE_INIT = "{system_prompt}\n\n{formatted_observation}"

# Template for subsequent observations (no system prompt)
<ENV_NAME>_TEMPLATE_STEP = "{formatted_observation}"
```

**c) Register environment manager** in `agent_system/environments/env_manager.py`:

```python
class <EnvName>EnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        self.memory.reset(batch_size=len(text_obs))
        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': prev_obs, 'action': actions})
        full_text_obs = self.build_text_obs(text_obs, init=False)
        # Add validity flags to infos
        return {'text': full_text_obs, ...}, rewards, dones, infos

    def build_text_obs(self, text_obs, init=False):
        # Format observations using templates + memory
        # init=True: include system prompt; init=False: observation only
```

**d) Add to environment factory** in `agent_system/environments/env_manager.py`:

```python
elif "<env_name>" in config.env.env_name.lower():
    from agent_system.environments.env_package.<env_name> import (
        build_<env_name>_envs, <env_name>_projection
    )
    _envs = build_<env_name>_envs(
        server_urls=config.env.<env_name>.server_urls,
        # ... other config
    )
    projection_f = partial(<env_name>_projection)
    env_manager = <EnvName>EnvironmentManager(_envs, projection_f, config)
```

### 3. Comprehensive Test Suite

Create `tests/environments/<env_name>/` following the Affine Game pattern:
- `conftest.py` - Fixtures, mock HTTP server
- `test_projection.py` - Unit tests for action parsing (18+ tests)
- `test_observation_formatting.py` - Unit tests for observation formatting (20+ tests)
- `test_worker.py` - Unit tests for HTTP worker with mocked requests (15+ tests)
- `test_envs.py` - Integration tests for Ray-distributed environments (10+ tests)
- `test_env_manager.py` - Integration tests for environment manager (12+ tests)
- `test_e2e.py` - End-to-end tests with mock server (15+ tests)

Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.requires_server`

### Reference Implementation
See complete example: `agent_system/environments/env_package/affine_game/` and `tests/environments/affine_game/`

## Code Linting & Formatting

```bash
# Pre-commit hooks (Ruff for linting and formatting)
pre-commit install
pre-commit run --all-files

# Manual Ruff usage
ruff check . --fix
ruff format .
```

Configuration in `pyproject.toml`:
- Line length: 300 (intentionally high)
- Linter: pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, isort
- Excludes `.ipynb` files

## Documentation

```bash
# Build docs
cd docs/
pip install -r requirements-docs.txt
make clean
make html

# View docs
python -m http.server -d _build/html/
# Navigate to http://localhost:8000
```

## Important Constraints & Patterns

### Docker-Based Environment Pattern
- **Server deployment**: Environment runs in Docker container, accessed via HTTP API
- **Clean separation**: Environment logic (server-side) vs. integration logic (verl-agent client-side)
- **No conda conflicts**: No need for environment-specific Python versions or package conflicts
- **Scalability**: Easy to scale by deploying multiple environment servers
- **Server URLs**: Configure via `env.<env_name>.server_urls` in training config

### Memory Usage
- **Step-independent design**: Context length stays constant across turns (observation + memory summary)
- Customize memory in `agent_system/memory/` to implement summarization or selective retention
- Memory accessed via `build_text_obs()` in environment managers
- Template pattern: First step includes system prompt (`TEMPLATE_INIT`), subsequent steps do not (`TEMPLATE_STEP`)

### Ray Configuration
- Training requires Ray initialization
- Configure CPU/GPU resources per worker via `env.resources_per_worker`
- `num_cpus_per_env_worker` typically set to 0.1 to minimize CPU usage
- HTTP workers are Ray actors, one per environment server connection

### Projection Functions
- **Purpose**: Parse LLM free-form text outputs into structured actions
- **Pattern**: Use regex to extract action IDs/formats (e.g., `\d+` for numeric IDs)
- **Validity flags**: Return 1 for valid actions, 0 for invalid (enables invalid action penalties)
- **Fallback**: If parsing fails, return original text with validity=0

### Model Support
Supported models: Qwen3, Qwen3-VL, Qwen2.5, Qwen2.5-VL, LLaMA3.2
- Text-only and vision-language (multi-modal) agents
- LoRA support for reduced GPU requirements (7B models on 2x H100)

### Algorithm Selection
- **GiGPO**: Best for long-horizon tasks with fine-grained credit assignment (recommended)
- **GRPO**: Simpler critic-free baseline, good for shorter tasks
- **PPO**: Classic actor-critic, requires separate value network
- **RLOO**: Leave-one-out with PPO-clip (similar to LOOP)
- **DAPO**: GRPO + dynamic sampling + clip-higher

## Recipes & Advanced Features

The `recipe/` directory contains additional algorithm implementations:
- `dapo/`: DAPO algorithm
- `spin/`: SPIN algorithm
- `sppo/`: SPPO algorithm
- `r1/`: R1-style reasoning experiments

## Legacy Environments (Being Phased Out)

The codebase includes several legacy environments that do **not** use the Docker-based architecture. These environments require complex environment-specific setup and are being phased out in favor of Docker-based environments:

**Legacy Environments**:
- ALFWorld - Embodied AI text-based tasks
- WebShop - E-commerce interaction tasks (requires Python <=3.10, separate conda env)
- Search - Tool-calling/retrieval tasks (requires separate `retriever` conda env with faiss-gpu)
- Sokoban - Visual puzzle game
- Gym Cards - Visual card games
- AppWorld - Digital interface control (experimental, requires separate conda env)

**Key Differences from Docker-Based Architecture**:
- Environments run **locally in Python** rather than in Docker containers
- Require **environment-specific package installations** and conda environments
- **Version conflicts** between environments (e.g., WebShop needs Python 3.10, Search needs faiss-gpu)
- **Tightly coupled** to verl-agent codebase rather than cleanly separated

**Installation**: See `agent_system/environments/README.md` for detailed setup instructions for legacy environments.

**Training Scripts**: Examples in `examples/<algorithm>_trainer/run_<env_name>.sh` still work for legacy environments.

**Future Direction**: New environments should use the Docker-based architecture. Legacy environments are maintained for backward compatibility but will eventually be migrated or deprecated.

## Known Issues

- **Legacy environments**: Require complex setup, potential version conflicts (see "Legacy Environments" section)
- WebShop: Requires Google Drive cookie for gdown downloads
- AppWorld: Integration is experimental
- Some dependency warnings (typer version conflicts) can be safely ignored
- Ray may require explicit shutdown between runs: `ray stop`
