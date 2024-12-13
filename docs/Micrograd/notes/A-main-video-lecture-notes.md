
#### **NEURAL NETWORKS AND BACKPROPAGATION**
------------
-------------

##### **CHAPTER** [00:00:00](https://www.youtube.com/watch?v=VMj-3S1tku0&t=0s) intro
What training a Neural Network looks like under the hood.
By the end of this, we will be able to define and train a Neural Net, to see how it works in an intuitive level.

&nbsp;

##### **CHAPTER** [00:00:25](https://www.youtube.com/watch?v=VMj-3S1tku0&t=25s) micrograd overview 
Micrograd - is basically a tiny Autograd engine.

Autograd engine - short for Automatic Gradient, it implements backpropagation.

Backpropagation - is an algorithm which allows you to efficiently evaluate a gradient of some kind of a loss function with respect to the weights of a neural network.
What that allows us to do is that we can iteratively tune the weights of that neural network to -> minimize the loss function, and therefore -> improve the accuracy of the neural network.

*So, Backpropagation would be at the **Mathematical core** of any modern deep neural network library like PyTorch or Jax.*

**Functionality of Micrograd explained - [micrograd-functionality](micrograd-functionality.md)**

So ultimately, Micrograd is all you need to train a Neural Network. Everything else is just for efficiency.

&nbsp;

##### **CHAPTER** [00:08:08](https://www.youtube.com/watch?v=VMj-3S1tku0&t=488s) derivative of a simple function with one input
A Simple working example was done to explain the working/calculation of a Derivative of a function.
Added to [Jupyter notebook on GitHub](../1-derivative-simple-function.ipynb)

&nbsp;

##### **CHAPTER** [00:14:12](https://www.youtube.com/watch?v=VMj-3S1tku0&t=852s) derivative of a function with multiple inputs
Added three more scalar inputs a, b, c. To the function d = a*b + c

The derivative of the final function d was seen wrt to each of them, and the behavior was observed.

Added to [Jupyter notebook on GitHub](../2-derivative-function-with-multiple-inputs.ipynb)

(Although I understood how the math of how this worked, I'm still not fully aware of how this explains the "behavior of the derivative" as he mentions in the end 🤔)

&nbsp;

##### **CHAPTER** [00:19:09](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1149s) starting the core Value object of micrograd and its visualization
Now, Neural Networks are these massive mathematical expressions.
So, we will be needing some data structures to maintain these expressions. Which is what we will be building.

- **Initial explanation from [value-object-creation](value-object-creation.md)**
- **From 22:45 to 24:54 - Visualization:** Explained in [value-object-creation (Visualization of the expression)](value-object-creation.md#visualization-of-the-expression)
- **From 24:55 to 29:01 - Generating the visual graphs** in [value-object-creation (Visualization of the expression continued)](value-object-creation.md#visualization-of-the-expression-continued)
- [value-object-creation (SUMMARY & WHAT TO DO NEXT)](value-object-creation.md#summary--what-to-do-next)

&nbsp;

##### **CHAPTER** [00:32:10](https://www.youtube.com/watch?v=VMj-3S1tku0&t=1930s) manual backpropagation example #1: simple expression
A grad variable has been declared in the Value object. Which will contain the derivative of L w.r.t each of those leaf nodes. 

Manual Backpropagation for L, f and g in the expression: [Notebook](../4_1_manual_backpropagation_simpleExpression.ipynb)
We have basically done 'gradient checks' here. Gradient check is essentially where we are deriving the backpropagation of an expression, by checking all of it's intermediate nodes.

**(VERY IMPORTANT PART) From 38:06** - This step will be the crux of backpropagation. This will be THE MOST IMPORTANT NODE TO UNDERSTAND. If we understand the gradient of this node, then we understand all of backpropagation and all of training of NN!! -> [crux-node-backpropagation](crux-node-backpropagation.md)

&nbsp;

##### **CHAPTER** [00:51:10](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3070s) preview of a single optimization step
Here, we are just trying to nudge the inputs to make our L value go up.

Now, we modify the weights of the leaf nodes (because that is what we usually have control over) slightly towards more positive direction and see how that affects L (in a more positive direction). 

So if we want L to increase, we should nudge the nodes slightly towards the gradient (eg, a should increase in the direction of the gradient, in step size)
[Notebook](../5_optimization_single_step_preview.ipynb)

This is basically one step of the optimization that we ended up running. And really these gradient values calculated, give us some power, because we know how to influence the final outcome. And this will be extremely useful in training neural nets (which we will see soon :) )

&nbsp;

##### **CHAPTER** [00:52:52](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3172s) manual backpropagation example #2: a neuron
Here, we are going to backpropagate through a neuron.

So now, we want to eventually build out neural networks. In the simpler stage, these are multi-level perceptrons.
For example: We have a two layer neural net, which contains two hidden (inner) layers which is made up of neurons which are interconnected to each other.

Now, biologically neurons are obviously complicated devices, but there are simple mathematical models of them. [neurons-explaination](neurons-explaination.md)
[Notebook](../6_0_backpropagation_neuron.ipynb)

From 1:00:38, we'll see the tanh function in action and then the implementation of backpropagation (Manual backpropagation method)

**If you want to influence the final output, then you should increase the bias. Only then the tanh will squash the final output and flat out to the value 1 (As seen in the graph.**
[Notebook](../6_1_backpropagation_neuron_manual_calculation.ipynb)

&nbsp;

##### **CHAPTER** [01:09:02](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4142s) implementing the backward function for each operation
We will be creating functions which would calculate the backpropagation i.e. the gradient values by itself! As the name of the chapter suggests, we'll be implementing it in each of the operations, like for '+', ' * ', 'tanh'	\
[Notebook](../7_backward_func_each_operation.ipynb)	\
\
Note on the '_ backward' function created:	\
	In the operation functions, we had created 'out' values which are an addition to/combination of the 'self' and 'other' values.	\
	Therefore we set the 'out._ backward' to be the function that backpropagates the gradient.	\
	Therefore we define what should happen when that particular operation function (Eg, add, mul) is called, inside the 'def backward()'

&nbsp;

##### **CHAPTER** [01:17:32](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4652s) implementing the backward function for a whole expression graph
Instead of calling the '_ backward' function each time, we are creating it as a function in the Value object itself.	

Now, we also need to make sure that all the nodes have been accessed and forward pass through. So, we have used a concept called 'topological sort' where all the nodes are accessed/traversed in the same/one direction (either left-to-right or vice-versa) [See the image here](https://miro.medium.com/v2/resize:fit:828/format:webp/1*uMg_ojFXts2WZSjcZe4oRQ.png)	

Therefore, we are adding the code for it, where it ensures that all the nodes are accessed at least once (and only stored once) and the node is only stored after/when all of it's child nodes are accessed and stored. This way we know we have traversed through the entire graph.	

Once all the nodes have been topologically sorted, we then reverse the nodes order (Since we are traversing it from left to right i.e. input to output, we are reversing it, so that the gradients are calculated. As we have done previously in our examples) call the '_ backward' function to perform backpropagation from the output.
[Notebook](../7_1_backward_func_entire_graph.ipynb)	

And that was it, that was backpropagation! (Atleast for one neuron :P)

&nbsp;

##### **CHAPTER** [01:22:28](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4948s) fixing a backprop bug when one node is used multiple times
Resolving a bug, where if there are multiple same nodes, then the calculation of the gradient isn't happening correctly as it considers both those separate nodes as a single node.	
[Notebook](../8_handling_onenode_used_multiple_times.ipynb)	

Solution was to append the gradient values.

&nbsp;

##### **CHAPTER** [01:27:05](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5225s) breaking up a tanh, exercising with more operations
So far we had directly made a function for tanh, because we knew what it's derivative be.	

So now we'll be trying to expand it into its other derivative form which contains exponents.	

Therefore we are able to perform more operations such as exponents, division and subtraction (Therefore making it a good exercise)	

**Entire detailed explanation in: [expanding-tanh-and-adding-more-operations](expanding-tanh-and-adding-more-operations.md)**

Apart from showing that we can do different operations. We also want to show that the level up to which we want to implement the operations is up to us.	

To explain, in our example- It can directly be 'tanh' or break it down into the expressions of exp, divide and subtract.	

As long as the know the backward pass of that operation, it can be implemented in anyway.

&nbsp;

##### **CHAPTER** [01:39:31](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5971s) doing the same thing but in PyTorch: comparison
Now we are going to see how we can convert our code into PyTorch (syntax?). Normally PyTorch is used during production.
**Comparison and Explanation: [pytorch-comparision](pytorch-comparision.md)**

&nbsp;

##### **CHAPTER** [01:43:55](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6235s) building out a neural net library (multi-layer perceptron) in micrograd 
We'll be building everything we have learnt till now, into a proper Neural Network type in code, using PyTorch.
**Everything will be broken down properly in - [multi-layer-perceptron](multi-layer-perceptron.md)**
So, forward pass implementation has been done. Next we will also implement the backpropagation part.

&nbsp;

##### **CHAPTER** [01:51:04](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6664s) creating a tiny dataset, writing the loss function
We made like a list of input values and then a list of values we want as an output.
Saw how the pred values turn out.
Calculated the loss function for individual as well as entire, to see how it is affecting.

Also note: I had added `__radd__()` to the Value object to handle the case of int adding with a Value.
Detailed step by step process is in the [notebook](../x12_creating_a_loss_function.ipynb)

&nbsp;

##### **CHAPTER** [01:57:56](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7076s) collecting all of the parameters of the neural net
Here we are going to write some convenience code, so that we can gather all the parameters of the neural net and work on all of them simultaneously. 
We'll be nudging them by a small amount based on the gradient differentiation.

That's where we are adding parameters functions. Now another reason why we are doing this, is that even the n function (Neuron function) in PyTorch also provides us with parameters which we can use. Therefore, we are declaring one for our MLP too. So there is the parameters of tensors and for us it's parameters of scalars.
[Notebook](../x13_collecting_all_parameters_in_NN.ipynb)

&nbsp;

##### **CHAPTER** [02:01:12](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7272s) doing gradient descent optimization manually, training the network
We are basically performing 'Gradient Descent' here, by slightly nudging the values of the inputs to see how it can help to reduce the loss function.
[Notebook](../x14_manual_gradient_descent_optimization.ipynb)	

(I didn't exactly get the predicted output as excepted, but hey this is just the beginning. We'll see where we get from here :) )

&nbsp;

**SUMMARY** [02:14:03](https://www.youtube.com/watch?v=VMj-3S1tku0&t=8043s) 

---------
----------
*And that was the end of the first lecture! See you in the next one ;)*
