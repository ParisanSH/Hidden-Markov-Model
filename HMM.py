import numpy as np 
import pandas as pd 
#--------------------------------------------------
# compute A_matris for Forward Algorithm
def compute_A(states_seq, unique_states, n):
    a_matris = np.zeros((n,n))
    for i in range(0,n):
        indexs = np.where(states_seq == unique_states[i])
        num_current_state = len(indexs[0])
        for j in range(0,n):
            count_match = 1
            for index in indexs[0][:]:
                if index+1 < len(states_seq):
                    if states_seq[index+1] == unique_states[j]:
                        count_match += 1
            prob = count_match/(num_current_state + n)
            a_matris[i][j] = prob
    print(a_matris.sum(axis=1))
    return(a_matris)
#--------------------------------------------------
# compute B_matris for Forward Algorithm
def compute_B(obs_seq, unique_obs, unique_states, states, n):
    vocb_len = len(np.unique(obs_seq))
    b_matris = np.zeros((vocb_len, n))
    for i in range(0,vocb_len):
        indexs = np.where(obs_seq == unique_obs[i])
        for j in range(0,n):
            count_match = 1
            for index in indexs[0][:]:
                if index+1 < len(obs_seq):
                    if states[index] == unique_states[j]:
                        count_match += 1
            num_current_state = len(np.where(states == unique_states[j])[0])
            prob = count_match/(num_current_state + vocb_len)
            b_matris[i][j] = prob
    return b_matris.transpose()

#--------------------------------------------------
#Forward algorithm for Likelihood Computation
def Forward(obs, unique_obs, N, A_matrix, B_matrix):
    T = len(obs) #len of observations
    pi = 1/N
    forwardProbMatrix = np.zeros((N,T)).transpose()
    #initialization step
    num_obs_in_unique = np.where(unique_obs == obs[0])[0][0]
    forwardProbMatrix[0] = pi * B_matrix[:, num_obs_in_unique]
    for t in range(1,T):
        num_obs_in_unique = np.where(unique_obs == obs[t])[0][0]
        forwardProbMatrix[t] = forwardProbMatrix[t-1].dot(A_matrix) * B_matrix[:, num_obs_in_unique]
    return forwardProbMatrix[-1].sum()

#--------------------------------------------------
# Decoding: The Viterbi Algorithm
def viterbi_func(obs, N, A_matrix, B_matrix, unique_states, unique_obs):
    T = len(obs)
    pi = 1/N
    viterbi = np.zeros((N,T)).transpose() #path_prob_matrix
    backpointer = np.zeros((N,T)).transpose()
    #initialization step
    num_obs_in_unique = np.where(unique_obs == obs[0])[0][0]
    viterbi[0] = pi * B_matrix[:, num_obs_in_unique]
    #recursion step
    for t in range(1,T):
        for s in range(N):
            num_obs_in_unique = np.where(unique_obs == obs[t])[0][0]
            viterbi[t,s] = np.max(viterbi[t-1]*A_matrix[:,s]) * B_matrix[s, num_obs_in_unique]
            backpointer[t,s] = np.argmax(viterbi[t-1]*A_matrix[:,s])
    bestpathprob = np.max(viterbi[:,-1])

    states = np.zeros(T, dtype=np.int32)
    states[T-1] = np.argmax(viterbi[T-1])
    for t in range(T-2, -1, -1):
        states[t] = backpointer[t+1, states[t+1]]
    print(states)
    print(bestpathprob)
    return states, bestpathprob

#--------------------------------------------------
#HMM Training: The Forward-Backward Algorithm 
def Forward_backward(obs, vocab_obs, tags):
    #initialize A ,B
    A = np.ones((len(tags),len(tags)))/len(tags)
    B = np.ones((len(tags),len(vocab_obs)))/len(vocab_obs)
    for iter in range(2000):
        # E-step caclculating sigma and gamma
        alfa_prob = forward_probs(obs, vocab_obs, tags, A, B) #
        beta_prob = backward_probs(obs, vocab_obs, tags, A, B) #, beta_val
        gamma_prob = compute_gamma(alfa_prob, beta_prob, obs, vocab_obs, tags, A, B)
        sigma_prob = compute_sigma(alfa_prob, beta_prob, obs, vocab_obs, tags, A, B)
        # M-step caclculating A, B matrices
        a_model = np.zeros((len(tags), len(tags)))
        for j in range(len(tags)): # calculate A-model
            for i in range(len(tags)):
                for t in range(len(obs)-1):
                    a_model[j,i] = a_model[j,i] + sigma_prob[j,t,i]
                normalize_a = [sigma_prob[j, t_current, i_current] for t_current in range(len(obs) - 1) for i_current in range(len(tags))]
                normalize_a = sum(normalize_a)
                if (normalize_a == 0):
                    a_model[j,i] = 0
                else:
                    a_model[j,i] = a_model[j,i]/normalize_a
        b_model = np.zeros((len(tags), len(vocab_obs)))
        for j in range(len(tags)): 
            for i in range(len(vocab_obs)): 
                indices = [idx for idx, val in enumerate(obs) if val == vocab_obs[i]]
                numerator_b = sum( gamma_prob[j,indices] )
                denomenator_b = sum( gamma_prob[j,:] )
                if (denomenator_b == 0):
                    b_model[j,i] = 0
                else:
                    b_model[j, i] = numerator_b / denomenator_b
        A = a_model
        B = b_model
    return A, B

#--------------------------------------------------
#find forward probabilities for forward_backward algorithm
def forward_probs(obs, vocab_obs, tags, A_, B_):
    a_start = 1/len(tags)
    states_val = np.zeros((len(tags), len(obs)))
    s = len(tags)
    for i, value in enumerate(obs):
        for j in range(s):
            if(i==0): # ijust for first sequence value use start prob
                index_in_vocab = np.where(vocab_obs == value)[0][0]
                states_val[j,i] = a_start * B_[j, index_in_vocab]
            else:
                index_in_vocab = np.where(vocab_obs == value)[0][0]
                temp_val = [states_val[k,i-1] * B_[j, index_in_vocab] * A_[k, j] for k in range(s)]
                states_val[j,i] = sum(temp_val)
    return states_val

#--------------------------------------------------
#find backward probabilities for forward_backward algorithm
def backward_probs(obs, vocab_obs, tags, A_, B_):
    backward_val = np.zeros((len(tags), len(obs)))
    backward_val[:,-1:] = 1
    for t in reversed(range(len(obs)-1)):
        for s in range(len(tags)):
            index_in_vocab = np.where(vocab_obs == obs[t+1])[0][0]
            backward_val[s,t] = np.sum(backward_val[:,t+1] * A_[s,:] * B_[:, index_in_vocab])
    return backward_val

#--------------------------------------------------
#find sigma probabilities for forward_backward algorithm
def compute_sigma(alfa, beta, obs, vocab_obs, tags, A_, B_):
    sigma_prob = np.zeros((len(tags), len(obs)-1, len(tags)))
    denomenator = np.multiply(alfa,beta)
    for i in range(len(obs)-1):
        for j in range(len(tags)):
            for k in range(len(tags)):
                index_in_vocab = np.where(vocab_obs == obs[i+1])[0][0]
                sigma_prob[j,i,k] = (alfa[j,i] * beta[k,i+1] * A_[j,k] * B_[k, index_in_vocab])/sum(denomenator[:,i])            
    return sigma_prob

#--------------------------------------------------
#find gamma probabilities for forward_backward algorithm
def compute_gamma(alfa, beta, obs, vocab_obs, tags, A_, B_):
    gamma_prob = np.zeros((len(tags), len(obs)))
    gamma_prob = np.multiply(alfa,beta) / sum(np.multiply(alfa,beta))
    return gamma_prob

#------------------- Main -------------------------
df = pd.read_csv('D://machine learning_hw//ML_HW4//ML_HW4_source//hmm-train.txt',sep=' ', header=None, encoding='ansi')
df.columns = ['Obs', 'state']
obs_train = list()
states_train = list()

for obs in df.Obs:
    obs_train.append(obs[2:-2])
for state in df.state:
    states_train.append(state[1:-2])

states_train = np.array(states_train)
obs_train = np.array(obs_train)
N = len(np.unique(states_train)) # number of states
#compute a , matrixs 
unique_states = np.unique(states_train)
unique_obs = np.unique(obs_train)
A_matrix = compute_A(states_train, unique_states, N)
B_matrix = compute_B(obs_train, unique_obs, unique_states, states_train, N)

#call Forward Algorithm to compute Likekihood
obs_test = ['why', 'should', 'you' ,'think', 'so']
obs_test2 = ['why', 'should', 'you' ,'think', 'like','that']
obs_test = np.array(obs_test)
obs_test2 = np.array(obs_test2)
Likelihood_comp = Forward(obs_test, unique_obs, N, A_matrix, B_matrix)
Likelihood_comp2 = Forward(obs_test2, unique_obs, N, A_matrix, B_matrix)
print('Prob of sentence1 : \t',Likelihood_comp)
print('Prob of sentence2 : \t',Likelihood_comp2)

#call Viterbi Algorithm 
best_path , best_path_prob = viterbi_func(obs_test, N, A_matrix, B_matrix, unique_states, unique_obs)
path_seq = list()
for index in best_path:
    path_seq.append(unique_states[index])
print(path_seq)
print(unique_states)
#Forward-Backward Algorithm datas
df = pd.read_csv('D://machine learning_hw//ML_HW4//ML_HW4_source//hmm-unsupervised.txt',sep=' ', header=None, encoding='ansi')
df.columns = ['Obs', 'state']
obs_unsupervised = list()
for obs in df.Obs:
    obs_unsupervised.append(obs[2:-2])
obs_unsupervised = np.array(obs_unsupervised)
df = pd.read_csv('D://machine learning_hw//ML_HW4//ML_HW4_source//tags_.txt',sep=' ', header=None, encoding='ansi')
tags = np.array(df)

vocab_obs = np.unique(obs_unsupervised)
print(vocab_obs.shape , obs_unsupervised.shape)
#call Forward-Backward
A_HMM , B_HMM = Forward_backward(obs_unsupervised, vocab_obs, tags)
print('-------------------------------------------------------------------------\n\n\n')
print('-------------------------        A        -------------------------------\n\n')
for elm in A_HMM:
    print(elm)
print('-------------------------------------------------------------------------\n\n\n')
print('-------------------------        B        -------------------------------\n\n')
for elm in B_HMM:
    print(elm)

#call Forward Algorithm for new model prameters
Likelihood_comp = Forward(obs_test, vocab_obs, len(tags), A_HMM, B_HMM)
Likelihood_comp2 = Forward(obs_test2, vocab_obs, len(tags), A_HMM, B_HMM)
print('Prob of sentence1 : \t',Likelihood_comp)
print('Prob of sentence2 : \t',Likelihood_comp2)

#call Viterbi Algorithm for new model prameters
best_path , best_path_prob = viterbi_func(obs_test, len(tags), A_HMM, B_HMM, tags, vocab_obs)
path_seq = list()
for index in best_path:
    path_seq.append(tags[index])
print(path_seq)
print(tags)