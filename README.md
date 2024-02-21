Reproduction of the paper "Deep Neuroevolution: Genetic Algorithms are a Competitive 
Alternative for Training Deep Neural Networks for Reinforcement Learning."
https://arxiv.org/pdf/1712.06567.pdf

Also using as a way to learn how to run distributed computing on clusters on aws spot
instances.

Currently distributed.py has options for to use local synchronous computation,
python multiprocessing, or RabbitMQ.

scripts/build.sh builds the image for kubernetes master/worker pods
scripts/start.sh deploys master, worker, rabbitmq, and tensorboard pods/services
to the cluster.



Current status 2/20/24: 
Spot instance training is working, currently running Atari FrostBite-v5
on 10 spot instance workers with a population size of 128 and parent pop size of 32.
(Note this is different from the paper's 1000x20). Currently have ran ~350 generations
and the best score is 2430 (averaged over 3 runs, note this also differs from the paper's
10).

In this graph the x-axis is generation, y-axis is average population score
![graph1](images/average.png "Average fitness")


In this graph the x-axis is generation, y-axis is best reward over all time (averaged over
3 runs for each agent)
![graph2](images/best.png "Best fitness")


This graph is just to track the total number of frames we've trained on.
In this graph the x-axis generation and the y-axis is the total environment frames divided by 4
(I'm off by a factor of 4 because I forgot atari env rolls out 4 frames per training timestep).
![graph3](images/frames.png "Training frames")


This is costing me about $15 a day, I probably should work on bringing that down more, it adds up
fast!
