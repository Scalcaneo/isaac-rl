import socket
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IsaacCombatEnv(gym.Env):
    """
    Custom Gymnasium environment for room-to-room combat and exploration in The Binding of Isaac.
    Action Space: MultiDiscrete([9, 5]) -> 9 move directions x 5 shoot directions.
    Observation Space: Box(90,) -> player (6) + room_type (6) + enemies (25) + projectiles (25) + doors (16) + pickups (12).
    """

    metadata = {"render_modes": ["human"]}

    MOVE_MAP = [
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-0.707, -0.707),
        (0.707, -0.707),
        (-0.707, 0.707),
        (0.707, 0.707),
    ]

    SHOOT_MAP = [
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (1.0, 0.0),
    ]

    # Isaac RoomType enum mapping to 6 one-hot buckets:
    # [Normal, Shop, Treasure, Boss, Secret/SuperSecret, Curse/Other]
    ROOM_TYPE_BUCKETS = {
        1: 0,   # ROOM_DEFAULT
        2: 1,   # ROOM_SHOP
        4: 2,   # ROOM_TREASURE
        5: 3,   # ROOM_BOSS
        7: 4,   # ROOM_SECRET
        8: 4,   # ROOM_SUPERSECRET
        10: 5,  # ROOM_CURSE
    }

    MAX_ENEMIES = 5
    MAX_PROJECTILES = 5
    MAX_DOORS = 4
    MAX_PICKUPS = 3

    def __init__(self, host: str = "0.0.0.0", port: int = 5555, max_steps: int = 1800):
        super().__init__()
        self.host = host
        self.port = port
        self.max_steps = max_steps  # 1800 steps * 4 ticks = 2 minutes per episode

        self.action_space = spaces.MultiDiscrete([len(self.MOVE_MAP), len(self.SHOOT_MAP)])

        # 6 (self) + 6 (room one-hot) + 25 (enemies) + 25 (projectiles) + 16 (doors) + 12 (pickups) = 90
        obs_dim = (
            6
            + 6
            + (self.MAX_ENEMIES * 5)
            + (self.MAX_PROJECTILES * 5)
            + (self.MAX_DOORS * 4)
            + (self.MAX_PICKUPS * 4)
        )
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        self.conn = None
        self.stream = None

        # Tracking state for reward shaping
        self.prev_hp = None
        self.prev_total_enemy_hp = None
        self.prev_room_sig = None
        self.visited_rooms = set()
        self.prev_collectible_count = 0
        self.step_count = 0

    def _ensure_connected(self):
        if self.conn is None:
            print(f"[IsaacEnv] Waiting for Isaac client on {self.host}:{self.port}...")
            self.conn, addr = self.server_sock.accept()
            self.stream = self.conn.makefile("r", encoding="utf-8")
            print(f"[IsaacEnv] Connected to {addr}!")

    def _read_state_packet(self):
        line = self.stream.readline()
        if not line:
            raise ConnectionError("Isaac disconnected unexpectedly.")
        parts = line.strip().split("|")

        tick = int(parts[0])
        hp = float(parts[1])
        max_hp = max(float(parts[2]), 1.0)
        px, py = float(parts[3]), float(parts[4])
        vx, vy = float(parts[5]), float(parts[6])
        tl_x, tl_y = float(parts[7]), float(parts[8])
        br_x, br_y = float(parts[9]), float(parts[10])
        is_clear = int(parts[11]) == 1
        room_type = int(parts[12])

        enemies = []
        if len(parts) > 13 and parts[13]:
            for item in parts[13].split(";"):
                ex, ey, evx, evy, ehp = map(float, item.split(","))
                enemies.append((ex, ey, evx, evy, ehp))

        projectiles = []
        if len(parts) > 14 and parts[14]:
            for item in parts[14].split(";"):
                bx, by, bvx, bvy = map(float, item.split(","))
                projectiles.append((bx, by, bvx, bvy))

        doors = []
        if len(parts) > 15 and parts[15]:
            for item in parts[15].split(";"):
                dx, dy, is_open, target_type = map(float, item.split(","))
                doors.append((dx, dy, int(is_open), int(target_type)))

        pickups = []
        if len(parts) > 16 and parts[16]:
            for item in parts[16].split(";"):
                ix, iy, variant, subtype = map(float, item.split(","))
                pickups.append((ix, iy, int(variant), int(subtype)))

        return {
            "tick": tick,
            "hp": hp,
            "max_hp": max_hp,
            "pos": (px, py),
            "vel": (vx, vy),
            "bounds": (tl_x, tl_y, br_x, br_y),
            "is_clear": is_clear,
            "room_type": room_type,
            "enemies": enemies,
            "projectiles": projectiles,
            "doors": doors,
            "pickups": pickups,
        }

    def _get_room_signature(self, state: dict) -> tuple:
        bx1, by1, bx2, by2 = state["bounds"]
        door_sig = tuple(sorted((round(d[0]), round(d[1]), d[3]) for d in state["doors"]))
        return (state["room_type"], round(bx1), round(by1), round(bx2), round(by2), door_sig)

    def _encode_room_type(self, room_type: int) -> list:
        vec = [0.0] * 6
        bucket = self.ROOM_TYPE_BUCKETS.get(room_type, 5)
        vec[bucket] = 1.0
        return vec

    def _build_obs(self, state: dict) -> np.ndarray:
        px, py = state["pos"]
        vx, vy = state["vel"]
        tl_x, tl_y, br_x, br_y = state["bounds"]

        room_w = max(br_x - tl_x, 1.0)
        room_h = max(br_y - tl_y, 1.0)
        cx, cy = (tl_x + br_x) * 0.5, (tl_y + br_y) * 0.5

        # 1. Self state (6 floats)
        obs = [
            (px - cx) / (room_w * 0.5),
            (py - cy) / (room_h * 0.5),
            vx / 10.0,
            vy / 10.0,
            state["hp"] / state["max_hp"],
            1.0 if state["is_clear"] else 0.0,
        ]

        # 2. Current room type one-hot (6 floats)
        obs.extend(self._encode_room_type(state["room_type"]))

        # 3. Nearest MAX_ENEMIES (25 floats)
        sorted_enemies = sorted(
            state["enemies"], key=lambda e: math.hypot(e[0] - px, e[1] - py)
        )[: self.MAX_ENEMIES]
        for ex, ey, evx, evy, ehp in sorted_enemies:
            obs.extend([
                (ex - px) / room_w,
                (ey - py) / room_h,
                evx / 10.0,
                evy / 10.0,
                min(ehp / 50.0, 2.0),
            ])
        obs.extend([0.0] * (5 * (self.MAX_ENEMIES - len(sorted_enemies))))

        # 4. Nearest MAX_PROJECTILES (25 floats)
        sorted_projs = sorted(
            state["projectiles"], key=lambda b: math.hypot(b[0] - px, b[1] - py)
        )[: self.MAX_PROJECTILES]
        for bx, by, bvx, bvy in sorted_projs:
            dx = (bx - px) / room_w
            dy = (by - py) / room_h
            obs.extend([dx, dy, bvx / 15.0, bvy / 15.0, math.hypot(dx, dy)])
        obs.extend([0.0] * (5 * (self.MAX_PROJECTILES - len(sorted_projs))))

        # 5. Doors sorted by distance (4 * 4 = 16 floats)
        sorted_doors = sorted(
            state["doors"], key=lambda d: math.hypot(d[0] - px, d[1] - py)
        )[: self.MAX_DOORS]
        for dx_pos, dy_pos, is_open, target_type in sorted_doors:
            target_bucket = self.ROOM_TYPE_BUCKETS.get(target_type, 5) / 5.0
            obs.extend([
                (dx_pos - px) / room_w,
                (dy_pos - py) / room_h,
                float(is_open),
                target_bucket,
            ])
        obs.extend([0.0] * (4 * (self.MAX_DOORS - len(sorted_doors))))

        # 6. Nearest Pickups / Pedestal Items (3 * 4 = 12 floats)
        sorted_pickups = sorted(
            state["pickups"], key=lambda p: math.hypot(p[0] - px, p[1] - py)
        )[: self.MAX_PICKUPS]
        for ix, iy, variant, subtype in sorted_pickups:
            is_collectible = 1.0 if variant == 100 else 0.0
            obs.extend([
                (ix - px) / room_w,
                (iy - py) / room_h,
                is_collectible,
                min(variant / 100.0, 2.0),
            ])
        obs.extend([0.0] * (4 * (self.MAX_PICKUPS - len(sorted_pickups))))

        return np.clip(np.array(obs, dtype=np.float32), -10.0, 10.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_connected()

        if seed is not None:
            self.conn.sendall(f"RESET {int(seed)}\n".encode("utf-8"))
        else:
            self.conn.sendall(b"RESET\n")

        state = self._read_state_packet()
        self.prev_hp = state["hp"]
        self.prev_total_enemy_hp = sum(e[4] for e in state["enemies"])
        self.prev_room_sig = self._get_room_signature(state)
        self.visited_rooms = {self.prev_room_sig}
        self.prev_collectible_count = sum(1 for p in state["pickups"] if p[2] == 100)
        self.step_count = 0

        return self._build_obs(state), {}

    def step(self, action):
        move_idx, shoot_idx = int(action[0]), int(action[1])
        mx, my = self.MOVE_MAP[move_idx]
        sx, sy = self.SHOOT_MAP[shoot_idx]

        self.conn.sendall(f"{mx},{my},{sx},{sy}\n".encode("utf-8"))

        state = self._read_state_packet()
        self.step_count += 1

        curr_hp = state["hp"]
        curr_enemy_hp = sum(e[4] for e in state["enemies"])
        curr_room_sig = self._get_room_signature(state)
        curr_collectibles = sum(1 for p in state["pickups"] if p[2] == 100)

        same_room = (curr_room_sig == self.prev_room_sig)

        # 1. Combat rewards (only compute enemy HP delta within the same room)
        if same_room:
            damage_dealt = max(0.0, self.prev_total_enemy_hp - curr_enemy_hp)
        else:
            damage_dealt = 0.0

        damage_taken = max(0.0, self.prev_hp - curr_hp)
        reward = (0.05 * damage_dealt) - (1.0 * damage_taken) - 0.005

        # 2. Exploration reward: entering a brand-new room
        if curr_room_sig not in self.visited_rooms:
            self.visited_rooms.add(curr_room_sig)
            reward += 0.5
            # Extra bonus for discovering a Treasure Room (4)
            if state["room_type"] == 4:
                reward += 0.75

        # 3. Item pickup reward: pedestal collectible disappeared while staying in the same room
        if same_room and curr_collectibles < self.prev_collectible_count:
            reward += 1.5

        terminated = False
        truncated = False

        if curr_hp <= 0:
            reward -= 2.0
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        self.prev_hp = curr_hp
        self.prev_total_enemy_hp = curr_enemy_hp
        self.prev_room_sig = curr_room_sig
        self.prev_collectible_count = curr_collectibles

        return self._build_obs(state), float(reward), terminated, truncated, {}

    def close(self):
        if self.conn:
            self.conn.close()
        self.server_sock.close()