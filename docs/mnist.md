# Genetic Algorithms on MNIST

This post explores training a simple MLP on MNIST, using genetic algorithms.

## The SGD baseline

A simple training of a MLP with 128 hidden neurons on MNIST using SGD is implemented in code/gradient.py. 
Without tuning the hyperparameters all that much, we can get to 95% valset acc in 4 epochs.

TODO add a graph for the number of network updates it takes to reach 95%.


## A simple asexual genetic algorithm

We begin with a simple genetic algorithm, very similar to that implemented in the paper https://arxiv.org/abs/1712.06567.
Chromosomes consist of strings of integers. Each integer specifies a unique mutation to the network. 
The network is initialized using He initialization, using some fixed seed, and then the final network is constructed as follows:

For each integer in the chromosome, in order, we set the seed of the torch random number generator to that integer, and then 
we generate a perturbation to the weights of the network using that seed. The perturbation is multiplied by some learning rate 
and then added to the weights of the network. The way I set (and evolve) the learning rate is explored in the "Learning rates" section below.

Once the network has been constructed, the fitness of an individual is the cross entropy loss over the entire MNIST training dataset.
In the section "Batch sizes" below we will investigate what happens when we only evaluate individuals on subsets of the MNIST dataset.
And in the section "Fitness types" below we will investigate using other types of fitnesses.

### What kind of perturbation should we use? 

The paper linked above generated the perturbations by sampling from a
normal distribution with mean 0 and standard deviation sigma. 
However it is not clear that this is the distribtuion from which to sample parameter mutations.
So this section will investigate if any other distributions perform better.

Initially, I decided to inspect the distribution of the partial derivatives over all parameters during normal SGD, and use
this (approximate) distribution in genetic algorithm mutations as well. Investigating the partial derivatives during
gradient descent, we see that their distribution is much sharper than a normal distribution, closer to an exponential distribution:

![plot](./images/partials_distribution.pdf)

So we can see that the actual updates during gradient descent will look more like the exponential distribution than the normal
one. (Note that something like a squared normal distribution or a squared exponential distribution could be even better,
but I just tested exponential in this post).

However, when I try mutating the parameters with an exponential distribution rather than a normal distribution, performance
decreases. What if we went the other extreme and used a uniform distribution? This too is worse than a normal distribution.
So it seems that making the distribution of random parameter updates have a heavier tail (exponential distribution) and a 
lighter tail (uniform distribution) both result in worse performance. 

TODO add graph

The normal distribution is radially symmetric, which is an appealing quality for mutations. 
But how are we to reconcile this result with the gradient descent distribution looking more exponential? 
If the true gradient distribution (as a random variable) is not radially symmetric, why would a radially symmetric
mutation perform the best?

Well, if we consider the true gradient to be a random vector with each component drawn from an exponential distribution,
then we can compute its dot product with normal mutations and compare with exponential mutations to see if one is likely to
perform better. The below graph shows this computation, and shows that exponential mutations are not more likely to 
align with the true gradient than normal mutations are (the plot also shows the uniform distribution).

![plot](./images/random_dots.png)

I am not sure why emperically the normal distribution seems to perform better than these other two. TODO investigate 

From now on in the following sections, I always use the normal distribution to mutate the parameters.

### Learning Rates
The symmetric normal distribution is controlled by a single parameter - sigma. Futhermore, the normal distribution family is 
closed under multiplication by a scalar:

TODO insert formula

So we can think of the sigmas for each layer as determining the learning rate of the algorithm - increasing sigma will result in 
larger magnitude updates after each mutation. Like gradient descent, genetic algorithms perform best when this learning rate changes over time,
decreasing as training converges. This can be accomplished easily by simply allowing these parameters to mutate along with all other parameters
of the model. During each mutation, sigma is either multiplied by some scaling factor
gamma, divided by gamma, or kept constant, each with equal probability. 

The below tensorboards shows training MNIST to ~95% acc using three different values of gamma: 1.05, 1.1, and 1.3.  

![plot](./images/val_acc_sigmamut_hyperparam.png)
![plot](./images/fitness_sigmamut_hyperparam.png)

We can see the algorithm is relatively robust to the value of gamma in this range. I use 1.05 in the following experiments since it performed the best 
of these three. When training this way, we can see how the values of sigma for each matrix in the MLP change over training:

![plot](./images/sigma1_curve.png)
![plot](./images/sigma2_curve.png)

As expected, the sigmas decrease over training. Additionally, notice how the light blue curve has much higher swings than the other curves.
That is because this curve uses gamma=1.3. Most of the curves on this graph use gamma=1.05. The gray curve uses gamma=1.1.

Now, we need to make sure that letting sigma mutate like this is actually better than just training with a constant sigmas. We compare
training with gamma=1.05 to training with gamma=1 (no sigma mutation) and constant values of sigma taken from the beginning of these
curves, the middle, and the end.


TODO insert graph showing mutating lr vs constant lr

Note I have not tested whether a tuned learning rate schedule will outperform this self-scheduling mutation.

### Batch Sizes
In all of the above experiments, to evaluate the fitness of the model, I compute its loss over the entire MNIST training dataset.
In this section I will investigate what happens when we use minibatches to evaluate the model instead of the entire training dataset.
