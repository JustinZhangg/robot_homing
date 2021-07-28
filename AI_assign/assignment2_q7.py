# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def homing_nn(n_trials, n_steps, learning_rate, eps, gamma, decayparameter, l=0.0):
    ## Definition of the environment
    N = 10                               #height of the gridworld ---> number of rows
    M = 10                              #length of the gridworld ---> number of columns
    N_states = N * M                    #total number of states
    states_matrix = np.eye(N_states)
    N_actions = 4                                           #number of possible actions in each state: 1->N 2->E 3->S 4->W
    action_row_change = np.array([-1,0,+1,0])               #number of cell shifted in vertical as a function of the action
    action_col_change = np.array([0,+1,0,-1])               #number of cell shifted in horizontal as a function of the action
    End = np.array([7, 9])                                  #terminal state--->reward
    #End = np.array([np.random.randint(N),np.random.randint(M)]) 
    s_end = np.ravel_multi_index(End,dims=(N,M),order='F')  #terminal state. Conversion in single index

    ## Rewards
    R = 1                              #only when the robot reaches the charger, sited in End state

    ## Variables
    weights = np.random.rand(N_actions,N_states)
    etracelist = np.zeros((N_actions,N_states))
    learning_curve = np.zeros((n_trials))
    
    #elig_old = np.zeros((N_actions, N_states))

    ## SARSA

    # Start trials
    for trial in range(n_trials):

        # Initialization
        Start = np.array([np.random.randint(N),np.random.randint(M)])   #random start
        s_start = np.ravel_multi_index(Start,dims=(N,M),order='F')      #conversion in single index
        state = Start                                                   #set current state
        s_index = s_start                                               #conversion in single index
        step = 0
        minmumstep = abs(Start[0] - End[0]) + abs(Start[1] - End[1])
        
        # Start steps
        while s_index != s_end and step <= n_steps:

            step += 1
            learning_curve[trial] = step - minmumstep

            input_vector = states_matrix[:,s_index].reshape(N_states,1)         #convert the state into an input vector

            #compute Qvalues. Qvalue=logsig(weights*input). Qvalue is 2x1, one value for each output neuron
            Q = 1 / ( 1 + np.exp( - weights.dot(input_vector)))    #Qvalue is 2x1 implementation of logsig
            
            # Note it is possible to remove the activation function and have a linear layer
            #Q = weights.dot(input_vector)

            #eps-greedy policy implementation
            greedy = (np.random.rand() > eps)               #1--->greedy action 0--->non-greedy action
            if greedy:
                action = np.argmax(Q)                           #pick best action
            else:
                action = np.random.randint(N_actions)           #pick random action


            state_new = np.array([0,0])
            #move into a new state
            state_new[0] = state[0] + action_row_change[action]
            state_new[1] = state[1] + action_col_change[action]

            #put the robot back in grid if it goes out. Consider also the option to give a negative reward
            if state_new[0] < 0:
                state_new[0] = 0
            if state_new[0] >= N:
                state_new[0] = N-1
            if state_new[1] < 0:
                state_new[1] = 0
            if state_new[1] >= M:
                state_new[1] = M-1
            
             # put the robot back to the position if it hits the wall, apply a penalty to the robot
            hitwall = False
            if state_new[0] == 3:
                if state_new[1] > 4:
                    state_new[0] = state[0]
                    state_new[1] = state[1]
                    hitwall = True
            if state_new[0] == 5:
                if state_new[1] < 4:
                    state_new[0] = state[0]
                    state_new[1] = state[1]
                    hitwall = True
            if state_new[1] == 6:
                if state_new[1] > 6:
                    state_new[0] = state[0]
                    state_new[1] = state[1]
                    hitwall = True
            
            s_index_new = np.ravel_multi_index(state_new,dims=(N,M),order='F')  #conversion in a single index
            
            
            if step > 1:                   
                    # use etrace method to update the weights
                    etracelist = gamma * decayparameter * etracelist + output_old.dot(input_old.T)
                    weights += learning_rate * (r_old - Q_old + gamma * Q[action]) * etracelist
           
            

            #store variables for sarsa computation in the next step
            output = np.zeros((N_actions,1))
            output[action] = 1

            #update variables
            input_old = input_vector
            output_old = output
            Q_old = Q[action]
            r_old = 0
            # To do: calculate eligibility trace
            # elig_old = 

            state[0] = state_new[0]
            state[1] = state_new[1]
            s_index = s_index_new

            ## check if state is terminal or if its over the step limite and update the weights consequently
          
            if step > n_steps:
                weights += learning_rate * (-1 - Q_old + gamma * Q[action]) * output_old.dot(input_old.T)
            elif s_index == s_end:
                weights += learning_rate * ((R - Q[action]) * output.dot(input_vector.T))
                pass
            elif hitwall == True:
                weights += learning_rate * (-0.1 - Q_old + gamma * Q[action]) * output_old.dot(input_old.T)
                
                


    return learning_curve, weights

# Parameter setup
nrepetitions = 10;  # number of runs for the algorithm
nTrials = 100000     # should be integer >0
nSteps = 50;        # maximum number of allowed steps
learningRate = 0.3; # should be real, Greater than 0
#learningRate = [0.1,0.2,0.3,0.4,0.5]
epsilon = 0.1;     # should be real, Greater or Equal to 0; epsion=0 Greedy, otherwise epsilon-Greedy
#epsilon = [0.1,0.2,0.3,0.4,0.5]
gamma = 0.9;        # should be real, positive, smaller than 1
#gamma = [0.5,0.6,0.7,0.8,0.9]
decayparameter = 0.3
#decayparameter = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

learning_curve = np.zeros(( nTrials))
#weight = np.zeros((4,100))
#averaged_extra_step = np.zeros((len(gamma)))
learning_curve, weight = homing_nn(nTrials, nSteps, learningRate, epsilon, gamma, decayparameter)

move_top = np.array([0,1])
move_down = np.array([0,-1])
move_left = np.array([-1,0])
move_right = np.array([1,0])

direction_top = np.zeros((100,2))
direction_left = np.zeros((100,2))
direction_right = np.zeros((100,2))
direction_down = np.zeros((100,2))
for each_weight in range(100):
    direction_top[each_weight] = weight[0][each_weight] * move_top
    direction_down[each_weight] = weight[1][each_weight] * move_down
    direction_left[each_weight] = weight[2][each_weight] * move_left
    direction_right[each_weight] = weight[3][each_weight] * move_right

total_direction = direction_top + direction_left + direction_right + direction_down 
print(total_direction)
#number of possible actions in each state: 1->N 2->E 3->S 4->W

x = np.arange(0, 10, 1)
y = np.arange(0, 10, 1)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
q = ax.quiver(X,Y, total_direction[:,0], total_direction[:,1])

plt.show()

learning_curve = np.zeros((nrepetitions, nTrials))


for i in range(nrepetitions):
    learning_curve[i] = homing_nn(nTrials, nSteps, learningRate, epsilon, gamma)

means = np.mean( learning_curve, axis=0)
errors = np.std(learning_curve, axis = 0) / np.sqrt(nrepetitions) # errorbars are equal to twice standard error i.e. std/sqrt(samples)

smooth_means = gaussian_filter1d(means, 2)
smooth_errors = gaussian_filter1d(errors, 2)

plt.errorbar(np.arange(nTrials), smooth_means, smooth_errors, 0, elinewidth = 0.1, capsize = 1, alpha =0.2)
plt.plot(smooth_means, 'tab:blue') # Plot the mean on top to standout

plt.xlabel('Trial',fontsize = 16)
plt.ylabel('Average Reward',fontsize = 16)
plt.tick_params(axis = 'both', which='major', labelsize = 14)
plt.savefig('Sarsa.png', dpi=300)
plt.show()