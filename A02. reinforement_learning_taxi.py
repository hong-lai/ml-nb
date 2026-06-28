import os
import gymnasium as gym
import numpy as np

np.random.seed(42)


# default with action masking
def get_valid_actions(info):
    action_mask = info["action_mask"]
    valid_actions = np.nonzero(action_mask)[0]
    return valid_actions


def train_q_learning(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # hyerparameters
    epsilon = 0.1
    alpha = 0.1  # learning rate
    gamma = 0.95  # discount factor
    n_epsoides = 50000

    q_table = np.zeros((n_states, n_actions))

    for epsoide in range(n_epsoides):
        state, info = env.reset()
        valid_actions = get_valid_actions(info)

        truncated = terminated = False

        while not (truncated or terminated):
            if len(valid_actions) == 0:
                action = np.random.choice(range(n_actions))

            elif np.random.random() < epsilon:
                # exploration
                action = np.random.choice(valid_actions)

            else:
                # exploitation
                action = valid_actions[np.argmax(q_table[state, valid_actions])]

            # run next step
            next_state, reward, terminated, truncated, info = env.step(action)

            if not (truncated or terminated):
                next_valid_actions = get_valid_actions(info)
                next_max = np.max(q_table[next_state, next_valid_actions])

                # update q_table
                q_table[state, action] += alpha * (
                    float(reward) + gamma * next_max - q_table[state, action]
                )

                valid_actions = next_valid_actions

            state = next_state

        if (epsoide + 1) % 500 == 0:
            print(f"Epsoide {epsoide + 1} / {n_epsoides}")

    return q_table


def test(env, q_table, n_tests=20):
    success = 0
    for _ in range(n_tests):
        state, info = env.reset()

        truncated = terminated = False
        while not (truncated or terminated):
            valid_actions = get_valid_actions(info)
            action = valid_actions[np.argmax(q_table[state, valid_actions])]
            next_state, _, terminated, truncated, info = env.step(action)
            state = next_state

        if terminated:
            success += 1
            print(f"Successful: {success} / {n_tests}", end='\r')


id = "Taxi-v4"

if os.path.exists(f"{id}.npy"):
    q_table = np.load(f"{id}.npy")
else:
    env = gym.make(id)
    q_table = train_q_learning(env)
    env.close()
    np.save(f"{id}.npy", q_table)


env = gym.make(id, render_mode="human")
test(env, q_table)
env.close()
