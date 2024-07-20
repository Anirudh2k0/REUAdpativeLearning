from newEnv import CyberDefense
import numpy as np
#import matplotlib.pyplot as plt
from flask import Flask, request

alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

all_epochs = []
all_penalties = []
scores = []

def calculation(env):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    for i in range(1, 10001):
        state = env.reset()
        epochs, reward = 0, 0
        done = False
        total_reward = 0
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values
            
            next_state, reward, done, _ = env.step(action)
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            
            total_reward += reward
            
            state = next_state
            epochs += 1
        
        scores.append(total_reward)
        #if i % 100 == 0:
        #    print(f"Episode: {i}")
    return(q_table)


def test_agent(env, q_table):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        print(f"State: {state}, Action: {action}, Reward: {reward}")

#test_agent(env, q_table)

def difficulties(state,q_table):
    action = np.argmax(q_table[state])
    return action

app = Flask(__name__)
@app.route('/api', methods=['GET']) 

def api_data():
    if request.method == 'GET':
        current_level = int(request.args.get('current_level'))
        success = int(request.args.get('success'))
        currTask = {"level": current_level, "success": success}
        env = CyberDefense(currTask)
        currState = env.reset()
        q_table = calculation(env)
        next_level = difficulties(currState,q_table)

        if next_level == 0:
             #print("Change level to 0")
             return {'level': 0 ,
                'status_code': 200}
        elif next_level == 1:
             #print("Change level to 1")
             return {'level': 1 ,
                'status_code': 200}
        elif next_level == 2:
             #print("Change level to 2")
             return {'level': 2 ,
                'status_code': 200}

    else:
        return 'Method Not Allowed', 405
 
if __name__ == '__main__':
    app.run(debug=True)
