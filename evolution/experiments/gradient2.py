#!python
import numpy as np
import pickle
from collections import defaultdict
from evolution.src.policies import LinearPolicy
import torch
import numpy as np
from evolution.src.config import *
from evolution.other.unpickler import renamed_load

from sklearn.preprocessing import normalize
import torchvision

from scipy.stats import shapiro

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

minibatch_size = 500
num_samples = 1000

make_pdf = True
do_train = False # if not do_train, will just load data from pickle and remake pdf
pdfname = 'images/batch_gradient.pdf'

device = torch.device("mps")

transform = torchvision.transforms.Compose([
    #torchvision.transforms.CenterCrop(10),
    torchvision.transforms.ToTensor(),
])
    
training_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=True, transform = transform)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=minibatch_size, shuffle=True)
    
val_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=False, transform = transform)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=minibatch_size, shuffle=False)

#individual_path1 = 'saves/june6/mnist-cross_entropy-asexual-mlp-parent1-child256-elites0-dsall-t0-pis2701387-sigma0.02-hidden_dim128-initialization_seed2701387-mutationnormal-sigma_mutation1.05-sigma_only_generations100_(-0.6634310483932495, 0.0)_gen1000.pkl'

individual_path1 = 'saves/june11/mnist-cross_entropy-asex-mlp-p1-c256-e0-dsall-t0-pis7244424-s10.0001-s20.0001-hidden_dim128-initialization_seed7244424-mutationnormal-sigma_mutation1.05-sigma_only_generations-1_-0.43067601323127747.pkl'
    
dna, config = pickle.load(open(individual_path1, 'rb'))
policy_args = config['eval_args']['policy_args']
eval_args = config['eval_args']
# Note, policy must be loaded with trainable=False so it can be reconstructed by 
# mutating from its dna.
    
policy = factory(dna, **policy_args)
# After loading, we set the parameters to trainable:
policy.l1.requires_grad = True
policy.l2.requires_grad = True


evaler = MNIST(**eval_args)
evaler.eval(dna, val=False, cached_policy=policy)

grad_dict = defaultdict(lambda: [])
full_grad_dict = {}

# Just need an optimizer so we can zero out the gradients, we aren't actually taking any steps here
optimizer = torch.optim.SGD(policy.parameters(), lr=0.1) 


# First, compute grad with respect to entire trainiing data
for i,(x,y) in enumerate(train_dataloader):
    x = x.to(device)
    y = y.to(device)
    logits,intrinsic_fitness = policy(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

grads = []
for p in policy.named_parameters():
    full_grad_dict[p[0]] = p[1].grad.detach().cpu().clone().numpy()
    grads.append(p[1].grad.detach().cpu().clone().flatten())

traingrad = torch.cat(grads).flatten()


samples = 0
popsize=256

# Now, compute grad with respect to smaller minibatches
done = False
minibatch_vs_training_sims = []
minibatch_bestsims = []
training_bestsims = []
sexsims = []
orgsims = []
while not done:
    for i,(x,y) in enumerate(train_dataloader):
        print(samples)
        #if samples % 100 == 0: print(samples)
        
        optimizer.zero_grad() 
        x = x.to(device)
        y = y.to(device)
        logits,intrinsic_fitness = policy(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
    
        grads = []
        for p in policy.named_parameters():
            grads.append(p[1].grad.detach().cpu().clone().flatten())
        grad = torch.cat(grads).reshape([-1])
        minibatch_vs_training_sims.append(torch.nn.functional.cosine_similarity(grad, traingrad, dim=0))


        # Now, randomly sample popsize vectors and take the one with highest similarity to the minibatch gradient
        bestsim = -99999999
        simlist = []
        for _ in range(popsize):
            x1 = torch.normal(mean=0, std=1, size=grad.shape)
            sim = torch.nn.functional.cosine_similarity(x1, grad, dim=0)
            simlist.append((sim,x1))

        simlist.sort(key=lambda x: x[0]) # make sure we never try to sort on the tensor

        bestsim = simlist[-1][0]
        p1 = simlist[-1][1]

        minibatch_bestsims.append(bestsim)
        training_sim = torch.nn.functional.cosine_similarity(p1, traingrad, dim=0)
        training_bestsims.append(training_sim)

        p2 = simlist[-2][1]

        sexsim = torch.nn.functional.cosine_similarity(p1 + p2, traingrad, dim=0)
        sexsims.append(sexsim)

        #org = simlist[-16][1]
        #for i in range(15):
        #  org += simlist[-i-1][1]

        #orgsim = torch.nn.functional.cosine_similarity(org, traingrad, dim=0)
        #orgsims.append(orgsim)


        samples += 1
        if samples >= num_samples: 
            done=True
            break


def make_histogram(ax, similarities, title, label=None, color=None):
    ax.hist(similarities, bins=100, label=label, color=color)
    #ax.set_yscale('log')
    ax.set_title(title, fontsize=10)

fig, ax = plt.subplots(1,1)

#for i,pname in enumerate(full_grad_dict.keys()):
#    similarities = []
#    for minibatch_gradient in grad_dict[pname]:
#        a = full_grad_dict[pname].flatten()
#        b = minibatch_gradient.flatten()
#        sim = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
#        similarities.append(sim)

#make_histogram(ax_lst[i], similarities, pname)


minibatch_std = torch.std(torch.tensor(minibatch_bestsims))
minibatch_mean = torch.mean(torch.tensor(minibatch_bestsims))
ax.hist(minibatch_bestsims, bins=100, label=f'pop256 vs minibatch. std: {minibatch_std:.3e}, mean: {minibatch_mean:.3e}', color='r', alpha=0.8)


training_std = torch.std(torch.tensor(training_bestsims))
training_mean = torch.mean(torch.tensor(training_bestsims))
ax.hist(training_bestsims, bins=100, label=f'pop256mini vs training. std: {training_std:.3e}, mean: {training_mean:.3e}', color='b', alpha=0.8)


sex_std = torch.std(torch.tensor(sexsims))
sex_mean = torch.mean(torch.tensor(sexsims))
ax.hist(sexsims, bins=100, label=f'sexual vs training. std: {sex_std:.3e}, mean: {sex_mean:.3e}', color='g', alpha=0.8)


ax.set_title('Minibatch similarities')

ax.legend(loc='best', fontsize=8)
fig.savefig(f'images/minibatch{minibatch_size}_vs_full_gradient.png')




