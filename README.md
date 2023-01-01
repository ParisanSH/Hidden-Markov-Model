# Hidden-Markov-Model
In this project, have tried to implement solutions to the three problems associated with HMMs: Forward Algorithm (likelihood computation), Decoding (The Viterbi Algorithm), and HMM training (The Forward-Backward Algorithm).

![image](https://user-images.githubusercontent.com/14861041/210172317-74641842-5ae4-43c7-bd36-8679734c2dea.png)

A hidden Markov model (HMM) allows us to talk about both observed events (e.g. words) and hidden events (e.g. part-of-speech tags). An HMM is specified by the following components:
![image](https://user-images.githubusercontent.com/14861041/210172354-4e8aef0d-a7fc-4ed5-984a-217e063a76d1.png)
![image](https://user-images.githubusercontent.com/14861041/210172377-dd104bfa-cea0-496e-ae4c-03ea58674a2f.png)

# Problem1) Likelihood Computation: The Forward Algorithm
First problem is to compute the likelihood of a particular observation sequence. More formally:
![image](https://user-images.githubusercontent.com/14861041/210172418-89560763-6053-41b3-bc27-ada52839281c.png)
 
The forward algorithm is a kind of dynamic programming algorithm, that is, an algorithm that uses a table to store intermediate values as it builds up the probability of the observation sequence. Forward the algorithm computes the observation probability by summing over the probabilities of all possible hidden state paths that could generate the observation sequence.

Each cell of the forward algorithm at 𝜶𝒕(𝒋) represents the probability of being in state 𝒋 after seeing the first 𝒕 observations, given the automaton λ. The value of each cell at 𝜶𝒕(𝒋) is computed by summing over the probabilities of every path that could lead to this cell. Formally, each cell expresses the following probability:
![image](https://user-images.githubusercontent.com/14861041/210172510-e943f7ca-945e-4310-92ed-6a6602498250.png)
Here, 𝒒𝒕 = 𝒋 means “the 𝒕^𝒕𝒉 state in the sequence of states is state 𝒋”. We compute this probability at 𝜶𝒕(𝒋) by summing over the extensions of all the paths that lead to the current cell. For a given state 𝒒𝒋 at time t,the value at 𝜶𝒕(𝒋) is computed as:

![image](https://user-images.githubusercontent.com/14861041/210172583-28853a5e-e2c1-4bb6-821c-a406cb0c8dc7.png)

The three factors of the forward model are:
![image](https://user-images.githubusercontent.com/14861041/210172597-fbd87492-effa-434b-aa6d-a11bd2035ad3.png)

Two formal definitions of the forward algorithm have been given: the pseudocode and a statement of the definitional recursion below.
![image](https://user-images.githubusercontent.com/14861041/210172803-afcefb9a-0992-4e2f-810c-c298e0637d8c.png)
![image](https://user-images.githubusercontent.com/14861041/210172810-f108e063-e096-44f8-8e0b-38aa1c001faa.png)

# Problem2) Decoding: The Viterbi Algorithm
For any model, such as an HMM, that contains hidden variables, the task of determining which sequence of variables is the underlying source of some sequence of observations is called the decoding task. The task of the decoder is to find the best hidden variable sequence (𝑞1𝑞2𝑞3 … 𝑞𝑛). More formally,
![image](https://user-images.githubusercontent.com/14861041/210172850-2e49f9fe-6e6c-4e71-8e28-a158f26b4847.png)

The most common decoding algorithms for HMMs is the Viterbi algorithm. Like the forward algorithm, Viterbi is a kind of dynamic programming.
Each cell 𝒗𝒕( 𝒋), represents the probability that the HMM is in state 𝒋 after seeing the first 𝒕 observationsand passing through the most probable state sequence 𝑞1 … 𝑞𝑡−1, given the automaton λ. The value of each cell 𝒗𝒕( 𝒋) is computed by recursively taking the most probable path. Like other dynamic programming algorithms, Viterbi fills each cell recursively. Given that we had already computed the probability of being in every state at time 𝑡 − 1, we compute the Viterbi probability by taking the most probable of the extensions of the paths that lead to the current cell. For a given state 𝒒𝒋 at time t, the value 𝒗𝒕( 𝒋) is computed as
![image](https://user-images.githubusercontent.com/14861041/210172894-c2bc81ac-245e-4d60-b20b-7362d07eb1f5.png)

The three factors of the Viterbi algorithm are:
![image](https://user-images.githubusercontent.com/14861041/210172907-64498161-707a-4f54-b077-9f4747cd19ac.png)

Pseudocode for the Viterbi algorithm is given in the following.

![image](https://user-images.githubusercontent.com/14861041/210172926-a6ec5956-bbbe-413b-8f37-e07219eea06f.png)

Note that the Viterbi algorithm is identical to the forward algorithm except that it takes the max over theprevious path probabilities whereas the forward algorithm takes the sum. Note also that the Viterbi algorithm has one component that the forward algorithm doesn’t have: backpointers. The reason is that while the forward algorithm needs to produce an observation likelihood, the Viterbi algorithm must produce a probability and also the most likely state sequence.

Finally, we can give a formal definition of the Viterbi recursion as follows
![image](https://user-images.githubusercontent.com/14861041/210172957-976c9e49-e5c1-49c4-aeb3-f9576b844e4c.png)
 
# Problem3) HMM Training: The Forward-Backward Algorithm

We turn to the third problem for HMMs: learning the parameters of an HMM, that is, the A and B matrices. Formally, 
![image](https://user-images.githubusercontent.com/14861041/210172978-5f2f3307-284d-4212-add0-dd335bd54f40.png)

The input to such a learning algorithm would be an unlabeled sequence of observations O and a vocabulary of potential hidden states Q.

The standard algorithm for HMM training is the forward-backward, or Baum-Welch algorithm, a special case of the Expectation-Maximization or EM algorithm. The algorithm trains both the transition probabilities A and the emission probabilities B of the HMM. EM is an iterative algorithm, computing an initial estimate for the probabilities, then using those estimates to computing a better estimate, and so on, iteratively improving the probabilities that it learns.

The Baum-Welch algorithm solves this problem by iteratively estimating the counts. The Baum-Welch algorithm starts with an estimate for the transition and observation probabilities and then uses these estimated probabilities to derive better and better probabilities.

To understand the algorithm, we need to define the backward probability. The backward probability 𝜷 is the probability of seeing the observations from time 𝒕 + 𝟏 to the end, given in state 𝑖 at time 𝑡 (and given the automaton 𝜆):
![image](https://user-images.githubusercontent.com/14861041/210173025-c43af892-aa7d-48d4-b8d3-4ef1b196cd4a.png)
It is computed inductively in a similar manner to the forward algorithm.
![image](https://user-images.githubusercontent.com/14861041/210173040-bfc4d190-33b0-4261-9254-26c61f418d46.png)

We are now ready to see how the forward and backward probabilities can help compute the transition probability 𝒂𝒊𝒋 and observation probability 𝒃𝒊(𝒐𝒕) from an observation sequence.
Let’s define the probability 𝝃𝒕 as the probability of being in state 𝒊 at time 𝒕 and state 𝒋 at time 𝒕 + 𝟏, given the observation sequence and of course the model:
![image](https://user-images.githubusercontent.com/14861041/210173079-b7800857-8377-446f-ace8-f8e821e845ab.png)

The following equation is used to compute 𝝃𝒕
![image](https://user-images.githubusercontent.com/14861041/210173087-2b980ced-4fea-474e-8996-b20162c41c42.png)

The transition probability 𝒂𝒊𝒋 is computed by the equation below:
![image](https://user-images.githubusercontent.com/14861041/210173098-23e61aac-da8d-422e-b15f-e6fdd10250e9.png)

We also need a formula for recomputing the observation probability 𝒃𝒊𝒋. This is the probability of a given symbol 𝒗𝒌 from the observation vocabulary V, given a state j: 𝒃̂ 𝒊(𝒗𝒌). For this, we need to know the probability of being in state 𝒋 at time 𝒕, called 𝜸𝒕(𝒋):

![image](https://user-images.githubusercontent.com/14861041/210173119-a98b815a-169a-4488-8f80-c8cd90aaad31.png)

The following equation is used to compute 𝜸𝒕(𝒋):
![image](https://user-images.githubusercontent.com/14861041/210173131-87437c3b-fcef-45b8-a371-96f54403cc20.png)

The observation probability 𝒃𝒊(𝒗𝒌) is re-estimated by the equation below:
![image](https://user-images.githubusercontent.com/14861041/210173145-458a5827-b778-48b2-a8fc-3b703e0abcf8.png)

The re-estimations of the transition A and observation B probabilities form the core of the iterative forwardbackward algorithm. The forward-backward algorithm starts with some initial estimate of the HMM parameters 𝜆 = (𝐴, 𝐵). The algorithm then iteratively runs two steps. Like other cases of the EM algorithm, the forward-backward algorithm has two steps: the expectation step (E-step), and the maximization step (M-step). In the E-step, the expected state occupancy count γ and the expected state transition count 𝜉 is computed from the earlier A and B probabilities. In the M-step, γ and 𝜉 are used to recompute new A and B probabilities.

Pseudocode for the forward-backward algorithm is given in the following.

![image](https://user-images.githubusercontent.com/14861041/210173224-189f16a6-7be7-4fd0-91f1-3c5ec9bc26f1.png)

Note1: Although in principle the Forward-Backward algorithm can do completely unsupervised learning of the A and B parameters, in practice the initial conditions are very important. For this reason, the algorithm is often given extra information. For example, for HMM-based speech recognition, the HMM structure is often set by hand, and only the emission (B) and (non-zero) A transition probabilities are trained from a set of observation sequences O.



