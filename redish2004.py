##################################preparation
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')
np.random.seed(0) 

critic_log = []
actor_log = []
Q = np.zeros(2) #action value
V = np.zeros(5) #state value
actions = [1,2]  # set multi acitons, food = 1 (S0 -> S1 -> S3), drug = 2(S0 -> S2 -> S4)
num_actions = np.array([0] * len(actions)) 

gamma = 0.9
lr = 0.08

################################ actor & critics
def actor():
    action = np.random.choice(a=actions, size=1, p=np.exp(Q) / np.sum(np.exp(Q), axis=0))
    num_actions[action - 1] += 1
    actor_log.append(copy.copy(num_actions))  
    return action


def critic(state): #when agent gets food
    global V
    global Q
    if state == 3 or state ==4:
        gain = reward
    else:
        gain = reward + gamma * V[next_state]
    estimated = V[state]
    td = gain - estimated
    V[state] += lr * td
    Q[action - 1] += lr * td

def d_critic(state): #when agent gets drug
    global V
    global Q
    dopamine=0.2
    if state == 3 or state ==4:
        gain = reward
    else:
        gain = reward + gamma * V[next_state]
    estimated = V[state]    
    td = max(gain - estimated + dopamine, dopamine)
    V[state] += lr * td
    Q[action - 1] += lr * td

############################## episodes
for e in range (1000):
    state = 0
    reward = 0

    action = actor()
    next_state = action
    critic(state)

    #S1 or S2
    state = next_state
    next_state = state + 2
    critic(state)
    
    #S3 or S4
    state = next_state
    next_state = 0
    if state == 3:
        reward = 1.0
        critic(state)

    elif state == 4:
        reward = 0.8
        d_critic(state)
    
    critic_log.append(copy.copy(V))

############################## draw figures
x = np.arange(1000)
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, [a[0] for a in actor_log], color='y', label='food')
ax1.plot(x, [a[1] for a in actor_log], color='b', label='drug')
ax1.legend()
ax1.set_xlabel("Episode")
ax1.set_ylabel("Cumulative number of actions")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x, [c[0] for c in critic_log], color='g', label='S0')
ax2.plot(x, [c[1] for c in critic_log], color='b', label='S1')
ax2.plot(x, [c[2] for c in critic_log], color='r', label='S2')
ax2.plot(x, [c[3] for c in critic_log], color='y', label='S3')
ax2.plot(x, [c[4] for c in critic_log], color='k', label='S4')
ax2.legend()
ax2.set_xlabel("Episode")
ax2.set_ylabel("Value of each state")

plt.savefig("demo.png")



                