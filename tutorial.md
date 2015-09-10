## Clockwork recurrent neural network

Clockwork recurrent neural network **CWRNN** is a specific version of recurrent neural network (RNN). The algorithm is a based upon a simple adjustment of RNN.

### <a name="qlearningNode"></a>Recurrent neural network (RNN)

The RNN is implemented in the `RNN` node. 
[RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) are neural networks with the ability to classify and predict **temporarily dependent data**. To the typical problems with complex temporal dependencies belong sequence prediction (such as handwriting and spoken word classification), sequence generation and similar. Recurrent networks can solve this problems by keeping **short-term memory** which is implemented by **using recurrent connections** (unit receives its own activation from the past step as an input).

RNN is partitioned into 3 layers:

 * input layer
 * hidden layer
 * output layer.

Input layer only provides new data to every unit in the hidden layer. Hidden layer is fully connected graph with recurrent connections of units to itself. Every unit from hidden layer is connected to the output unit, which provides the output of the network.

### <a name="qlearningNode"></a>Clockwork recurrent neural network (CWRNN)

The RNN is implemented in the `CWRNN` node. It is basically only a **small modification of the RNN**, the hidden layer of CWRNN is divided into separate blocks and each block processes input at different time periods. That means that **not all units at every time step change their activation upon the actual input**. The not updated units retain this way some **information about the past** which they provide to the newly updated units, they provide **context** from more deep history.

ADD SOME MORE MATH!!!

At each time step $\t$ output $\mathbf{y}_O^t$ is calculated by equation:

$$ \mathbf{y}_H^t = f_H (\mathbf{W}_H * \mathbf{y}^(t-1) + \mathbf{W}_I * \mathbf{x}^(t)) $$

$$ \mathbf{y}_O^t = f_O (\mathbf{W}_O * \mathbf{y}_H^(t)) $$

describe matrixes

add drawing??


### <a name="qlearningNode"></a>How to use the Node

any input

has its own Real time recurrent learner

output corresponding to the size of the target

TADY NEJAKY OBRAZEK ZAPOJENI

![Q-Learning Node](discreteqlearning/discrete-Qlearning.PNG)


#### <a name="qlearningHowToUse"></a>When to Use the Node

It is suitable to use this Node if the problem:

 * **Has time dependency:** delay the input from `MnistWorld` (or any other repeating serie) and use the same as target to the network. This way you make the network predict future steps.  
 * **You want to generate series** If your network is already trained on some series (text, music etc.), you can try to generate your own. Just use the output of the network as the input to the same.
