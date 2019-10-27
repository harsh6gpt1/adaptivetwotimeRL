import numpy as np
import time
import gym
import itertools
import math
import random
from collections import deque
from scipy import stats


def fourier_basis(statef, inner):
    state_repf = np.zeros(len(inner))
    for ind in range(len(inner)):
        state_repf[ind] = math.cos(math.pi * np.dot(inner[ind], statef))
    return state_repf


def project_state(statef, dimf, maxf, minf):
    temp = np.zeros(dimf)
    for ind in range(dimf):
        temp[ind] = (statef[ind] - minf[ind]) / (maxf[ind] - minf[ind])
    return temp


def compute_neu(num_ep, thetaf, per_ep, envir, dimf, maxf, minf, inner, disc, policy_in):
    # type: (object, object, object, object, object, object, object, object, object) -> object
    total_iterf = 0
    episode_neu = np.zeros(len(thetaf))
    for ind in range(num_ep):
        curr_s = envir.reset()
        ep_len = per_ep
        ep_rew = 0
        for t in range(per_ep):
            if policy_in == "random":
                action_f = random.randint(0, 1)
                if action_f == 1:
                    action_f += 1
            if policy_in == "optimal":
                action_f = int(target_policy_mountain_car(curr_state))
                action_f += 1
            new_s, rew, donev, infov = envir.step(action_f)
            statev = project_state(curr_s, dimf, maxf, minf)
            state_repv = fourier_basis(statev, inner)
            state_nextv = project_state(new_s, dimf, maxf, minf)
            state_rep_nextv = fourier_basis(state_nextv, inner)
            td_diffv = np.dot(thetaf, state_repv) - rew - disc * np.dot(thetaf, state_rep_nextv)
            curr_s = new_s
            episode_neu += td_diffv * state_repv
            ep_rew += rew
            if donev:
                ep_len = t + 1
                break
        total_iterf += ep_len
    episode_neu = episode_neu / total_iterf
    neu = np.linalg.norm(episode_neu)
    return neu


def target_policy_mountain_car(statef):
    return np.sign(statef[1])


start_time = time.time()
fourier_order = 3
dim = 2
num_parameters = (fourier_order + 1) ** dim
fourier_list = []
fourier_ini = np.asarray(range(fourier_order + 1))
for i in range(dim):
    fourier_list.append(fourier_ini)
fourier_inner = np.asarray([x for x in itertools.product(*fourier_list)])
fourier_scale = np.ones(num_parameters)
for i in range(num_parameters):
    if np.linalg.norm(fourier_inner[i]) != 0:
        fourier_scale[i] = 1. / np.linalg.norm(fourier_inner[i])
num_iter = 5
num_episodes = 10000
per_episode = 200
discount = 0.95
theta_ini = np.zeros(num_parameters)
omega_ini = np.zeros(num_parameters)
print_after = 100
gym.envs.register(
    id='MountainCarTD-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=per_episode,  # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make('MountainCarTD-v0')
max_state = env.observation_space.high
min_state = env.observation_space.low
test_after = 1000
test_episodes = 1000
err_arr = np.zeros((int(num_episodes / test_after), num_iter))
err_arr_from_ini = np.zeros(num_episodes)
replay = 200
policy = "random"
indices = range(replay)
indices = np.array(indices)
indices += 1
total_iter = 0
step_rule = "gupta"
algorithm = "tdc"
step_ini = 0.1
gamma_ini = 1.5
step_num_mannor = 0.08
update_threshold = 0.001
update_factor = 1.2
replay = 200
indices = range(replay)
indices = np.array(indices)
indices += 1
alpha = 1.0
beta = 0.9


for j in range(num_iter):
    theta = theta_ini
    omega = omega_ini
    theta_gupta_ini = theta_ini
    omega_gupta_ini = omega_ini
    if step_rule == "gupta":
        gamma = gamma_ini
        step_base = step_ini
        step_1 = step_base ** gamma
        step_2 = step_base
        replay_err = deque()
    print("iteration num: ", j)
    step_change_gupta = []
    for i in range(num_episodes):
        if step_rule == "gupta":
            if len(replay_err) >= replay and abs(slope) <= math.sqrt(step_2 ** (2 - gamma)) * update_threshold / replay:
                step_base = step_base / update_factor
                step_1 = step_base ** gamma
                step_2 = step_base
                replay_err = deque()
                theta_gupta_ini = theta
                omega_gupta_ini = omega
                step_change_gupta.append(i)
        if step_rule == "mannor":
            step_1 = step_num_mannor / ((i + 1) ** alpha)
            step_2 = step_num_mannor / ((i + 1) ** beta)
        curr_state = env.reset()
        episode_reward = 0
        if (i + 1) % print_after == 0:
            print("episode num:", i)
        for t in range(per_episode):
            if policy == "random":
                action = random.randint(0, 1)
                if action == 1:
                    action += 1
            if policy == "optimal":
                action = int(target_policy_mountain_car(curr_state))
                action += 1
            new_state, reward, done, info = env.step(action)
            state = project_state(curr_state, dim, max_state, min_state)
            state_rep = fourier_basis(state, fourier_inner)
            state_next = project_state(new_state, dim, max_state, min_state)
            state_rep_next = fourier_basis(state_next, fourier_inner)
            td_diff = np.dot(theta, state_rep) - reward - discount * np.dot(theta, state_rep_next)
            if algorithm == "gtd2":
                theta = theta + step_1 * np.dot(state_rep, omega) * (state_rep - discount * state_rep_next)
            if algorithm == "tdc":
                theta = theta + step_1 * (-td_diff) * state_rep - step_1 * discount * state_rep_next * np.dot(state_rep,
                                                                                                              omega)
            omega = omega + step_2 * (-td_diff - np.dot(omega, state_rep)) * state_rep
            curr_state = new_state
            episode_reward += reward
            total_iter += 1
            if done:
                break
        if (i + 1) % test_after == 0:
            error = compute_neu(test_episodes, theta, per_episode, env, dim, max_state, min_state, fourier_inner,
                                discount, policy)
            err_arr[int((i + 1) / test_after) - 1][j] = error
        error_from_ini = math.sqrt(np.linalg.norm(theta - theta_gupta_ini)**2 + np.linalg.norm(omega - omega_gupta_ini)
                                   ** 2)
        if step_rule == "gupta":
            if len(replay_err) >= replay:
                replay_err.popleft()
            replay_err.append(error_from_ini)
            y = np.array(replay_err)
            if len(replay_err) >= replay:
                slope, intercept, r_value, p_value, std_err = stats.linregress(indices, y)
        err_arr_from_ini[i] += error_from_ini
err_arr_mean = np.mean(err_arr, 1)
err_arr_from_ini = err_arr_from_ini / num_iter
total_time = time.time() - start_time
print("Time taken - %f seconds" % total_time)
