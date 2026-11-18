# isaac-rl

A Reinforcement Learning environment and training framework for **The Binding of Isaac: Repentance**, built around a low-latency, lockstep Lua-to-Python TCP IPC bridge and a custom `gymnasium.Env` interface.

Rather than relying on slow pixel-scraping and convolutional encoders, `isaac-rl` hooks directly into the game's Lua modding API (`MC_POST_UPDATE` and `MC_INPUT_ACTION`) to extract exact entity coordinates, velocity vectors, room boundaries, and projectile states at 15 Hz (4-tick frame skip), enabling rapid policy iteration on consumer hardware.

---

## Architecture

```text
┌─────────────────────────────────────────┐          ┌──────────────────────────────────────────┐
│   The Binding of Isaac (Lua Mod)        │          │   Python RL Environment (Gymnasium)      │
│                                         │          │                                          │
│  MC_POST_UPDATE (every 4 ticks / 15 Hz) │          │  IsaacCombatEnv.step(action)             │
│   ├─ Serialize player, room, enemies,   │── TCP ──▶│   ├─ Parse pipe-delimited state packet   │
│   │  and enemy projectiles              │ (127.0.0.1│   ├─ Build 56-dim normalized obs vector  │
│   └─ Block awaiting action reply        │  :5555)  │   ├─ Compute dense shaped reward         │
│                                         │◀── TCP ──│   └─ Sample policy & send action string  │
│  MC_INPUT_ACTION                        │          │                                          │
│   └─ Override GET_ACTION_VALUE &        │          │  PyTorch / PPO Agent                     │
│      IS_ACTION_PRESSED hooks            │          └──────────────────────────────────────────┘
└─────────────────────────────────────────┘

```
## Key Technical Details

    * Lockstep Synchronization: The Lua mod blocks on socket read every FRAME_SKIP = 4 simulation ticks until Python returns an action, preventing simulation drift during neural network forward/backward passes. TCP_NODELAY is enabled to eliminate Nagle buffering latency.

    * Dual-Hook Input Injection: Movement is injected via InputHook.GET_ACTION_VALUE (analog floats), while tear firing is injected via InputHook.IS_ACTION_PRESSED and IS_ACTION_TRIGGERED (digital booleans) to match the engine's internal polling behavior.

    * Player-Relative Observation Space (Box(56,)): All coordinates are normalized relative to Isaac and the active room's playable inner bounds ([-1, 1]):

        Self State (6 floats): Normalized room position (x, y), velocity (vx, vy), HP ratio, and room clear flag.

        Nearest 5 Enemies (25 floats): Relative offset (dx, dy), velocity (vx, vy), and normalized HP, sorted by Euclidean distance and zero-padded.

        Nearest 5 Projectiles (25 floats): Relative offset (dx, dy), velocity (vx, vy), and Euclidean distance, sorted by distance and zero-padded.

    * Action Space (MultiDiscrete([9, 5])): 9 movement directions (8-way + idle) × 5 shooting directions (4 cardinal + idle).

## Setup & Installation
1. Prerequisites

    The Binding of Isaac: Rebirth (with Afterbirth+ or Repentance DLC for Lua mod support)

    Python 3.10+

    PyTorch, Gymnasium, and NumPy:
    Bash

    pip install gymnasium numpy torch

2. Enable --luadebug in Steam

By default, Isaac sandboxes Lua mods and blocks the socket library. To enable local TCP communication:

    Open Steam → Right-click The Binding of Isaac: Rebirth → Properties.

    Under General → Launch Options, add:
    Plaintext

    --luadebug

(Tip: Use a dedicated save slot such as Save Slot 3 for RL training, as active mods disable achievements.)

3. Install the Lua Bridge Mod

Copy the mod folder into your local Isaac mods directory:

    Repentance:
    Plaintext

C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth\mods\isaac_rl_bridge\main.lua

Repentance+:
Plaintext

    %USERPROFILE%\Documents\My Games\Binding of Isaac Repentance+\mods\isaac_rl_bridge\main.lua

Launch the game, open the Mods menu, and verify Isaac RL Bridge is enabled.
Usage
Verify the Gymnasium Environment

Start the Python environment server first, then start a run in Isaac:
Python

from gymnasium.utils.env_checker import check_env
from isaac_env import IsaacCombatEnv

env = IsaacCombatEnv(host="127.0.0.1", port=5555)
check_env(env)
print("Environment passed all Gymnasium checks!")

To hot-reload changes to main.lua without restarting the game, open the in-game debug console (~) and run:
Plaintext

luamod isaac_rl_bridge

## Project Roadmap

    [x] Stage 1 — IPC Bridge: Synchronous TCP socket bridge with MC_POST_UPDATE state serialization and MC_INPUT_ACTION dual-hook injection.

    [ ] Stage 2a — Custom Gymnasium Environment: Player-relative 56-dim observation space and dense combat reward shaping (isaac_env.py).

    [ ] Stage 2b — Single-Room Combat Agent: Auto-combat room teleport on reset + PPO baseline training.

    [ ] Stage 3 — Vectorized Multi-Instance Scaling: Running multiple concurrent Isaac instances on multi-core CPUs via SubprocVecEnv / port offsets.

    [ ] Stage 4 — Hierarchical Floor Navigation: High-level planner for procedural dungeon routing, resource management, and item selection paired with the low-level combat policy.
