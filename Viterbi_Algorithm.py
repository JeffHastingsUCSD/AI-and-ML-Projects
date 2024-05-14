import numpy as np

def viterbi(obs_seq, states, start_prob, trans_prob, emis_prob):
    # Number of states
    num_states = len(states)
    
    # Number of observations
    num_obs = len(obs_seq)
    
    # Initialize the Viterbi probability matrix and the backpointer matrix
    viterbi_prob = np.zeros((num_obs, num_states))
    backpointer = np.zeros((num_obs, num_states), dtype=int)
    
    # Initialization step
    for s in range(num_states):
        viterbi_prob[0, s] = start_prob[s] * emis_prob[s, obs_seq[0]]
        backpointer[0, s] = 0
    
    # Iteration step
    for t in range(1, num_obs):
        for s in range(num_states):
            viterbi_prob[t, s] = np.max(viterbi_prob[t-1, :] * trans_prob[:, s]) * emis_prob[s, obs_seq[t]]
            backpointer[t, s] = np.argmax(viterbi_prob[t-1, :] * trans_prob[:, s])
    
    # Termination step
    best_path_prob = np.max(viterbi_prob[num_obs - 1, :])
    best_last_state = np.argmax(viterbi_prob[num_obs - 1, :])
    
    # Path backtracking
    best_path = np.zeros(num_obs, dtype=int)
    best_path[-1] = best_last_state
    for t in range(num_obs - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]
    
    return best_path, best_path_prob

# Example usage
obs_seq = [0, 1, 0]  # Example observation sequence (indexing observations)
states = [0, 1]      # Example states (indexing states)
start_prob = np.array([0.6, 0.4])  # Initial state probabilities
trans_prob = np.array([[0.7, 0.3],  # Transition probabilities
                       [0.4, 0.6]])
emis_prob = np.array([[0.5, 0.5],   # Emission probabilities
                      [0.4, 0.6]])

best_path, best_path_prob = viterbi(obs_seq, states, start_prob, trans_prob, emis_prob)
print("Most probable state sequence:", best_path)
print("Probability of the best path:", best_path_prob)
