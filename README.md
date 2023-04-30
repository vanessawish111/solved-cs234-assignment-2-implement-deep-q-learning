Download Link: https://assignmentchef.com/product/solved-cs234-assignment-2-implement-deep-q-learning
<br>
In this assignment we will implement deep Q learning, following DeepMind’s paper ([<strong>mnih2015human</strong>] and [<strong>mnih-atari-2013</strong>]) that learns to play Atari from raw pixels. The purpose is to understand the effectiveness of deep neural network as well as some of the techniques used in practice to stabilize training and achieve better performance. You’ll also have to get comfortable with Tensorflow. We will train our networks on the Pong-v0 environment from OpenAI gym, but the code can easily be applied to any other environment.

In Pong, one player wins if the ball passes by the other player. Winning a game gives a reward of 1, while losing gives a negative reward of -1. An episode is over when one of the two players reaches 21 wins. Thus, the final score is between -21 (lost episode) or +21 (won episode). Our agent plays against a decent hardcoded AI player. Average human performance is −3 (reported in [<strong>mnih-atari-2013</strong>]). If you go to the end of the homework successfully, you will train an AI agent with super-human performance, reaching at least +10 (hopefully more!).

<h1>1           Test Environment (5 pts)</h1>

Before running our code on Pong, it is crucial to test our code on a test environment. You should be able to run your models on CPU in no more than a few minutes on the following environment:

<ul>

 <li>4 states: 0<em>,</em>1<em>,</em>2<em>,</em>3</li>

 <li>5 actions: 0<em>,</em>1<em>,</em>2<em>,</em>3<em>,</em> Action 0 ≤ <em>i </em>≤ 3 goes to state <em>i</em>, while action 4 makes the agent stay in the same state.</li>

 <li>Rewards: Going to state <em>i </em>from states 0, 1, and 3 gives a reward <em>R</em>(<em>i</em>), where <em>R</em>(0) = 0<em>.</em>1<em>,R</em>(1) = −0<em>.</em>2<em>,R</em>(2) = 0<em>,R</em>(3) = −0<em>.</em> If we start in state 2, then the rewards defind above are multiplied by −10. See Table 1 for the full transition and reward structure.</li>

</ul>

1

<ul>

 <li>One episode lasts 5 time steps (for a total of 5 actions) and always starts in state 0 (no rewards at the initial state).</li>

</ul>

<table width="0">

 <tbody>

  <tr>

   <td width="66">State (s)</td>

   <td width="76">Action (a)</td>

   <td width="102">Next State (s’)</td>

   <td width="85">Reward (R)</td>

  </tr>

  <tr>

   <td width="66">0</td>

   <td width="76">0</td>

   <td width="102">0</td>

   <td width="85">0.1</td>

  </tr>

  <tr>

   <td width="66">0</td>

   <td width="76">1</td>

   <td width="102">1</td>

   <td width="85">-0.2</td>

  </tr>

  <tr>

   <td width="66">0</td>

   <td width="76">2</td>

   <td width="102">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="66">0</td>

   <td width="76">3</td>

   <td width="102">3</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="66">0</td>

   <td width="76">4</td>

   <td width="102">0</td>

   <td width="85">0.1</td>

  </tr>

  <tr>

   <td width="66">1</td>

   <td width="76">0</td>

   <td width="102">0</td>

   <td width="85">0.1</td>

  </tr>

  <tr>

   <td width="66">1</td>

   <td width="76">1</td>

   <td width="102">1</td>

   <td width="85">-0.2</td>

  </tr>

  <tr>

   <td width="66">1</td>

   <td width="76">2</td>

   <td width="102">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="66">1</td>

   <td width="76">3</td>

   <td width="102">3</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="66">1</td>

   <td width="76">4</td>

   <td width="102">1</td>

   <td width="85">-0.2</td>

  </tr>

  <tr>

   <td width="66">2</td>

   <td width="76">0</td>

   <td width="102">0</td>

   <td width="85">-1.0</td>

  </tr>

  <tr>

   <td width="66">2</td>

   <td width="76">1</td>

   <td width="102">1</td>

   <td width="85">2.0</td>

  </tr>

  <tr>

   <td width="66">2</td>

   <td width="76">2</td>

   <td width="102">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="66">2</td>

   <td width="76">3</td>

   <td width="102">3</td>

   <td width="85">1.0</td>

  </tr>

  <tr>

   <td width="66">2</td>

   <td width="76">4</td>

   <td width="102">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="66">3</td>

   <td width="76">0</td>

   <td width="102">0</td>

   <td width="85">0.1</td>

  </tr>

  <tr>

   <td width="66">3</td>

   <td width="76">1</td>

   <td width="102">1</td>

   <td width="85">-0.2</td>

  </tr>

  <tr>

   <td width="66">3</td>

   <td width="76">2</td>

   <td width="102">2</td>

   <td width="85">0.0</td>

  </tr>

  <tr>

   <td width="66">3</td>

   <td width="76">3</td>

   <td width="102">3</td>

   <td width="85">-0.1</td>

  </tr>

  <tr>

   <td width="66">3</td>

   <td width="76">4</td>

   <td width="102">3</td>

   <td width="85">-0.1</td>

  </tr>

 </tbody>

</table>

Table 1: Transition table for the Test Environment

An example of a path (or an episode) in the test environment is shown in Figure 1, and the trajectory can be represented in terms of <em>s<sub>t</sub>,a<sub>t</sub>,R<sub>t </sub></em>as: <em>s</em><sub>0 </sub>= 0<em>,a</em><sub>0 </sub>= 1<em>,R</em><sub>0 </sub>= −0<em>.</em>2<em>,s</em><sub>1 </sub>= 1<em>,a</em><sub>1 </sub>= 2<em>,R</em><sub>1 </sub>= 0<em>,s</em><sub>2 </sub>= 2<em>,a</em><sub>2 </sub>= 4<em>,R</em><sub>2 </sub>= 0<em>,s</em><sub>3 </sub>= 2<em>,a</em><sub>3 </sub>= 3<em>,R</em><sub>3 </sub>= (−0<em>.</em>1) ∗ (−10) = 1<em>,s</em><sub>4 </sub>= 3<em>,a</em><sub>4 </sub>= 0<em>,R</em><sub>4 </sub>= 0<em>.</em>1<em>,s</em><sub>5 </sub>= 0.

Figure 1: Example of a path in the Test Environment

<ol>

 <li>(<strong>written </strong>5pts) What is the maximum sum of rewards that can be achieved in a single episode in the test environment, assuming <em>γ </em>= 1?</li>

</ol>

<h1>2           Q-learning (12 pts)</h1>

<strong>Tabular setting </strong>In the <em>tabular setting</em>, we maintain a table <em>Q</em>(<em>s,a</em>) for each tuple state-action. Given an experience sample (<em>s,a,r,s</em><sup>0</sup>), our update rule is

<em> ,                                               </em>(1)

where <em>α </em>∈ R is the learning rate, <em>γ </em>the discount factor.

<strong>Approximation setting </strong>Due to the scale of Atari environments, we cannot reasonably learn and store a Q value for each state-action tuple. We will instead represent our Q values as a function ˆ<em>q</em>(<em>s,a,</em><strong>w</strong>) where <strong>w </strong>are parameters of the function (typically a neural network’s weights and bias parameters). In this <em>approximation setting</em>, our update rule becomes

<strong>w </strong><em>.                                        </em>(2)

In other words, we are try to minimize

(3)

<strong>Target Network </strong>DeepMind’s paper [<strong>mnih2015human</strong>] [<strong>mnih-atari-2013</strong>] maintains two sets of parameters, <strong>w </strong>(to compute ˆ<em>q</em>(<em>s,a</em>)) and <strong>w</strong><sup>− </sup>(target network, to compute ˆ<em>q</em>(<em>s</em><sup>0</sup><em>,a</em><sup>0</sup>)) such that our update rule becomes

<strong>w </strong><em>.                                      </em>(4)

The target network’s parameters are updated with the Q-network’s parameters occasionally and are kept fixed between individual updates. Note that when computing the update, we don’t compute gradients with respect to <strong>w</strong><sup>− </sup>(these are considered fixed weights).

<strong>Replay Memory </strong>As we play, we store our transitions (<em>s,a,r,s</em><sup>0</sup>) in a buffer. Old examples are deleted as we store new transitions. To update our parameters, we <em>sample </em>a minibatch from the buffer and perform a stochastic gradient descent update.

<strong>-Greedy Exploration Strategy </strong>During training, we use an -greedy strategy. DeepMind’s paper [<strong>mnih2015human</strong>] [<strong>mnih-atari-2013</strong>] decreases <em> </em>from 1 to 0<em>.</em>1 during the first million steps. At test time, the agent choses a random action with probability

There are several things to be noted:

<ul>

 <li>In this assignment, we will update <strong>w </strong>every learningfreq steps by using a minibatch of experiences sampled from the replay buffer.</li>

 <li>DeepMind’s deep Q network takes as input the state <em>s </em>and outputs a vector of size = number of actions.</li>

</ul>

In the Pong environment, we have 6 actions, thus ˆ<em>q</em>(<em>s,</em><strong>w</strong>) ∈ R<sup>6</sup>.

<ul>

 <li>The input of the deep Q network is the concatenation 4 consecutive steps, which results in an input after preprocessing of shape (80 × 80 × 4).</li>

</ul>

We will now examine these assumptions and implement the epsilon-greedy strategy.

<ol>

 <li>(<strong>written </strong>3pts) What is one benefit of using experience replay?</li>

 <li>(<strong>written </strong>3pts) What is one benefit of the target network?</li>

 <li>(<strong>written </strong>3pts) What is one benefit of representing the <em>Q </em>function as ˆ<em>q</em>(<em>s,</em><strong>w</strong>) ∈ R<em><sup>K</sup></em></li>

 <li>(<strong>coding </strong>3pts) Implement the getaction and update functions in py. Test your implementation by running python q1schedule.py.</li>

</ol>

<h1>3           Linear Approximation (26 pts)</h1>

<ol>

 <li>(<strong>written </strong>3pts) Show that Equations (1) and (2) from section 2 above are exactly the same when <em>q</em>ˆ(<em>s,a,</em><strong>w</strong>) = <strong>w</strong><em><sup>T</sup>x</em>(<em>s,a</em>), where <strong>w </strong>∈ R<sup>|<em>S</em>||<em>A</em>| </sup>and <em>x </em>: <em>S </em>× <em>A </em>→ R<sup>|<em>S</em>||<em>A</em>| </sup>such that</li>

</ol>

1         if <em>s</em><sup>0 </sup>= <em>s,a</em><sup>0 </sup>= <em>a</em>

<em>s,a </em><em>s</em>0<em>,a</em>0

0    otherwise

for all (<em>s,a</em>) ∈ <em>S</em>×<em>A</em>, <em>x</em>(<em>s,a</em>) is a vector of length |<em>S</em>||<em>A</em>| where the element corresponding to <em>s</em><sup>0 </sup>∈ <em>S,a</em><sup>0 </sup>∈ <em>A </em>is 1 when <em>s</em><sup>0 </sup>= <em>s,a</em><sup>0 </sup>= <em>a </em>and is 0 otherwise.

<ol start="2">

 <li>(<strong>written </strong>3pts) Derive the gradient with regard to the value function parameter <strong>w </strong>∈ R<em><sup>n </sup></em>given <em>q</em>ˆ(<em>s,a,</em><strong>w</strong>) = <strong>w</strong><em><sup>T</sup>x</em>(<em>s,a</em>) for any function <em>x</em>(<em>s,a</em>) 7→ <em>x </em>∈ R<em><sup>n </sup></em>and write the update rule for <strong>w</strong>.</li>

 <li>(<strong>coding </strong>15pts) We will now implement linear approximation in Tensorflow. This question will setup the whole pipeline for the remiander of the assignment. You’ll need to implement the following functions in py (pleasd read throughq2linear.py) :

  <ul>

   <li>addplaceholdersop</li>

   <li>getqvaluesop</li>

   <li>addupdatetargetop</li>

   <li>addlossop</li>

   <li>addoptimizerop</li>

  </ul></li>

</ol>

Test your code by running python q2linear.py <strong>locally on CPU</strong>. This will run linear approximation with Tensorflow on the test environment. Running this implementation should only take a minute or two.

<ol start="4">

 <li>(<strong>written </strong>5pts) Do you reach the optimal achievable reward on the test environment? Attach the plot png from the directory results/q2linear to your writeup.</li>

</ol>

<h1>4           Implementing DeepMind’s DQN (15 pts)</h1>

<ol>

 <li>(<strong>coding </strong>10pts) Implement the deep Q-network as described in [<strong>mnih2015human</strong>] by implementing getq valuesop in py. The rest of the code inherits from what you wrote for linear approximation. Test your implementation <strong>locally on CPU </strong>on the test environment by running python q3nature.py. Running this implementation should only take a minute or two.</li>

 <li>(<strong>written </strong>5pts) Attach the plot of scores, png, from the directory results/q3nature to your writeup. Compare this model with linear approximation. How do the final performances compare?</li>

</ol>

How about the training time?

<h1>5           DQN on Atari (27 pts)</h1>

The Atari environment from OpenAI gym returns observations (or original frames) of size (210×160×3), the last dimension corresponds to the RGB channels filled with values between 0 and 255 (uint8). Following DeepMind’s paper [<strong>mnih2015human</strong>], we will apply some preprocessing to the observations:

<ul>

 <li>Single frame encoding: To encode a single frame, we take the maximum value for each pixel color value over the frame being encoded and the previous frame. In other words, we return a pixel-wise max-pooling of the last 2 observations.</li>

 <li>Dimensionality reduction: Convert the encoded frame to grey scale, and rescale it to (80 × 80 × 1). (See Figure 2)</li>

</ul>

The above preprocessing is applied to the 4 most recent observations and these encoded frames are stacked together to produce the input (of shape (80×80×4)) to the Q-function. Also, for each time we decide on an action, we perform that action for 4 time steps. This reduces the frequency of decisions without impacting the performance too much and enables us to play 4 times as many games while training. You can refer to the <em>Methods Section </em>of [<strong>mnih2015human</strong>] for more details.

(a) Original input (210 × 160 × 3) with RGB colors (b) After preprocessing in grey scale of shape (80×80×1) Figure 2: Pong-v0 environment

<ol>

 <li>(<strong>written </strong>2pts) Why do we use the last 4 time steps as input to the network for playing Atari games?</li>

 <li>(<strong>written </strong>5pts) What’s the number of parameters of the DQN model (for Pong) if the input to the Q-network is a tensor of shape (80<em>,</em>80<em>,</em>4) and we use ”SAME” padding? How many parameters are required for the linear Q-network, assuming the input is still of shape (80<em>,</em>80<em>,</em>4)? How do the number of parameters compare between the two models?</li>

 <li>(<strong>coding and written </strong>5pts). Now, we’re ready to train on the Atari Pong-v0 First, launch linear approximation on pong with python q4trainatarilinear.py<strong>on Azure’s GPU</strong>. This will train the model for 500,000 steps and should take approximately an hour. What do you notice about the performance?</li>

 <li>(<strong>coding and written </strong>10 pts). In this question, we’ll train the agent with DeepMind’s architecture on the Atari Pong-v0 Run python q5trainatarinature.py <strong>on Azure’s GPU</strong>. This will train the model for 5 million steps and should take around <strong>12 hours</strong>. Attach the plot scores.png from the directory results/q5trainatarinature to your writeup. You should get a score of around 13-15 after 5 million time steps. As stated previously, the Deepmind paper claims average human performance is −3.</li>

</ol>

As the training time is roughly 12 hours, you may want to check after a few epochs that your network is making progress. The following are some training tips: • If you terminate your terminal session, the training will stop. In order to avoid this, you should use screen to run your training in the background.

<ul>

 <li>The evaluation score printed on terminal should start at -21 and increase.</li>

 <li>The max of the q values should also be increasing</li>

 <li>The standard deviation of q shouldn’t be too small. Otherwise it means that all states have similar q values</li>

 <li>You may want to use Tensorboard to track the history of the printed metrics. You can monitor your training with Tensorboard by typing the command tensorboard –logdir=results and then connecting to ip-of-you-machine:6006. Below are our Tensorboard graphs from one training session:</li>

</ul>

<ol start="5">

 <li>(<strong>written </strong>5pts) Compare the performance of the DeepMind architecure with the linear Q-network approximation. How can you explain the gap in performance?</li>

</ol>

<h1>6           Real world RL with neural networks (10 pts)</h1>

Given a stream of batches of <em>n </em>environment interactions (<em>s<sub>i</sub>,a<sub>i</sub>,r<sub>i</sub>,s</em><sup>0</sup><em><sub>i</sub></em>) we want to learn the optimal value function using a neural network. The underlying MDP has a finite sized action space.

<ol>

 <li>(<strong>written </strong>4pts) Your friend first suggests the following approach

  <ul>

   <li>Initialize parameters <em>φ </em>of neural network <em>V<sub>φ</sub></em></li>

   <li>For each batch of <em>k </em>tuples ( ) do Stochastic Gradient Descent with loss function <em>V<sub>φ</sub></em>(<em>s<sub>i</sub></em>)|<sup>2 </sup>where <em>y<sub>i </sub></em>= max<em><sub>a</sub></em><em><sub>i</sub></em>[<em>r<sub>i </sub></em>+ <em>γV<sub>φ</sub></em>(<em>s</em><sup>0</sup><em><sub>i</sub></em>)]</li>

  </ul></li>

</ol>

What is the problem with this approach? (Hint: Think about the type of data we have)

<ol start="2">

 <li>(<strong>written </strong>3pts) Your friend now suggests the following

  <ul>

   <li>Initialize parameters <em>φ </em>of neural network for state-action value function <em>Q<sub>φ</sub></em>(<em>s,a</em>)</li>

   <li>For each batch of <em>k </em>tuples (<em>s<sub>i</sub>,a<sub>i</sub>,r<sub>i</sub>,s</em><sup>0</sup><em><sub>i</sub></em>) do Stochastic Gradient Descent with loss function <em>Q<sub>φ</sub></em>(<em>s<sub>i</sub>,a<sub>i</sub></em>)|<sup>2 </sup>where</li>

  </ul></li>

</ol>

Now as we just have the network <em>Q<sub>φ</sub></em>(<em>s,a</em>) how would you determine <em>V </em>(<em>s</em>) needed for the above training procedure?

<ol start="3">

 <li>(<strong>written </strong>3pts) Is the above method of learning the <em>Q </em>network guaranteed to give us an approximation of the optimal state action value function?</li>

</ol>