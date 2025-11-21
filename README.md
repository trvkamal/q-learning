# Q Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
→ Initialize Q-table and hyperparameters.<br>
→ Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.<br>
→ After training, derive the optimal policy from the Q-table.<br>
→ Implement the Monte Carlo method to estimate state values.<br>
→ Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.<br>

## Q LEARNING FUNCTION
#### Name: KAMALESH V
#### Register Number: 212222240042
```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
          action = select_action(state, Q, epsilon[e])
          next_state, reward, done, _ = env.step(action)
          td_target = reward + gamma * Q[next_state].max() * (not done)
          td_error = td_target - Q[state][action]
          Q[state][action] = Q[state][action] + alphas[e] * td_error
          state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```





## OUTPUT:
### Optimal State Value Functions:
<img width="731" height="134" alt="k71" src="https://github.com/user-attachments/assets/f60f0eb4-bb9f-40e5-8c2f-3c85527c64ac" />


### Optimal Action Value Functions:
<img width="836" height="565" alt="k72" src="https://github.com/user-attachments/assets/837129b6-bfde-42e0-818f-8035774ae552" />



### State value functions of Monte Carlo method:
<img width="873" height="406" alt="k73" src="https://github.com/user-attachments/assets/5a2899f6-b963-4dbe-9e1b-2a473b18c128" />



### State value functions of Qlearning method:
<img width="871" height="399" alt="k74" src="https://github.com/user-attachments/assets/bad826c9-7660-4501-92e5-d3fb0c992cc6" />



## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
