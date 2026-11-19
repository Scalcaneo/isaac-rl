import numpy as np
from isaac_env import IsaacCombatEnv


def main():
    env = IsaacCombatEnv()

    print("1. Resetting environment...")
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), f"Obs out of bounds: min={obs.min()}, max={obs.max()}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    print(f"Reset OK! Obs shape: {obs.shape}, dtype: {obs.dtype}")

    print("2. Running 100-step live rollout...")
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs), f"Step {step} obs out of bounds!"
        assert isinstance(reward, float), f"Reward must be float, got {type(reward)}"
        total_reward += reward

        if step % 15 == 0:
            print(f"Step {step:3d} | Reward: {reward:+.3f} | Total: {total_reward:+.3f} | Done: {terminated or truncated}")

        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()
    print("\nAll environment checks and live rollout passed!")


if __name__ == "__main__":
    main()