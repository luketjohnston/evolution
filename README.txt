Reproduction of the paper "Deep Neuroevolution: Genetic Algorithms are a Competitive 
Alternative for Training Deep Neural Networks for Reinforcement Learning."
https://arxiv.org/pdf/1712.06567.pdf


Currently distributed.py has options for to use local synchronous computation,
python multiprocessing, or RabbitMQ.

scripts/build.sh builds the image for kubernetes master/worker nodes
scripts/start.sh deploys master, worker, rabbitmq, and tensorboard nodes/services
to the cluster.

Current status 2/17/24: everything is working on Cartpole on amazon eks, next step
is to get karpenter working so we can use spot instances. Atari is implemented too
but I haven't ran it long enough locally to reproduce the paper results, waiting until
I get the spot instance cluster setup to try that.
