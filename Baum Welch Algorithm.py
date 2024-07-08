import numpy as np

def forward(obs_seq, states, start_prob, trans_prob, emis_prob):
    num_states = len(states)
    num_obs = len(obs_seq)
    fwd = np.zeros((num_obs, num_states))
    fwd[0, :] = start_prob * emis_prob[:, obs_seq[0]]
    
    for t in range(1, num_obs):
        for s in range(num_states):
            fwd[t, s] = np.sum(fwd[t-1, :] * trans_prob[:, s]) * emis_prob[s, obs_seq[t]]
    
    total_prob = np.sum(fwd[-1, :])
    return fwd, total_prob

def backward(obs_seq, states, start_prob, trans_prob, emis_prob):
    num_states = len(states)
    num_obs = len(obs_seq)
    bwd = np.zeros((num_obs, num_states))
    bwd[-1, :] = 1
    
    for t in range(num_obs - 2, -1, -1):
        for s in range(num_states):
            bwd[t, s] = np.sum(trans_prob[s, :] * emis_prob[:, obs_seq[t + 1]] * bwd[t + 1, :])
    
    total_prob = np.sum(bwd[0, :] * start_prob * emis_prob[:, obs_seq[0]])
    return bwd, total_prob

def baum_welch(obs_seq, states, start_prob, trans_prob, emis_prob, n_iter=100):
    num_states = len(states)
    num_obs = len(obs_seq)
    for _ in range(n_iter):
        fwd, fwd_prob = forward(obs_seq, states, start_prob, trans_prob, emis_prob)
        bwd, bwd_prob = backward(obs_seq, states, start_prob, trans_prob, emis_prob)
        
        xi = np.zeros((num_obs - 1, num_states, num_states))
        for t in range(num_obs - 1):
            denom = np.sum(fwd[t, :] * trans_prob * emis_prob[:, obs_seq[t + 1]] * bwd[t + 1, :])
            if denom == 0:
                denom = np.finfo(float).eps  # Handle zero denominators
            for i in range(num_states):
                numer = fwd[t, i] * trans_prob[i, :] * emis_prob[:, obs_seq[t + 1]] * bwd[t + 1, :]
                xi[t, i, :] = numer / denom
        
        gamma = np.sum(xi, axis=2)
        gamma = np.vstack((gamma, np.sum(xi[-1, :, :], axis=0)))
        
        start_prob = gamma[0] / np.sum(gamma[0])  # Normalize start_prob
        trans_prob = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True)
        trans_prob = np.nan_to_num(trans_prob, nan=0.0)  # Replace NaNs with 0

        for s in range(num_states):
            for o in range(emis_prob.shape[1]):
                obs_indices = [t for t in range(num_obs) if obs_seq[t] == o]
                emis_prob[s, o] = np.sum(gamma[obs_indices, s]) / np.sum(gamma[:, s])
        emis_prob = np.nan_to_num(emis_prob, nan=0.0)  # Replace NaNs with 0
    
    return start_prob, trans_prob, emis_prob

# Example usage
obs_seq = [1, 1, 1]  # Example observation sequence (indexing observations)
states = [0, 1]      # Example states (indexing states)
start_prob = np.array([0.6, 0.4])  # Initial state probabilities
trans_prob = np.array([[0.7, 0.3],  # Transition probabilities
                       [0.4, 0.6]])
emis_prob = np.array([[0.5, 0.5],   # Emission probabilities
                      [0.4, 0.6]])

new_start_prob, new_trans_prob, new_emis_prob = baum_welch(obs_seq, states, start_prob, trans_prob, emis_prob)
print("Updated initial probabilities:\n", new_start_prob)
print("Updated transition probabilities:\n", new_trans_prob)
print("Updated emission probabilities:\n", new_emis_prob)
