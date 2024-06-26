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

eval_also=True

minibatch_size = 500
#num_samples = 200
num_samples = 400
numparents_l = [1,64]
mults = [1/2,1]

update_as_we_go=False
sort_with = 'fitness' # | 'similarity'

make_pdf = True
do_train = False # if not do_train, will just load data from pickle and remake pdf
#pdfname = 'images/batch_gradient_avstuck1.pdf'
imname = f'images/gradient_eval{eval_also}_update{update_as_we_go}_sortwith{sort_with}_1.png'

device = torch.device("mps")

transform = torchvision.transforms.Compose([
    #torchvision.transforms.CenterCrop(10),
    torchvision.transforms.ToTensor(),
])
    
training_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=True, transform = transform)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=minibatch_size, shuffle=True)
    
#val_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=False, transform = transform)
#val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=minibatch_size, shuffle=False)

#individual_path1 = 'saves/june6/mnist-cross_entropy-asexual-mlp-parent1-child256-elites0-dsall-t0-pis2701387-sigma0.02-hidden_dim128-initialization_seed2701387-mutationnormal-sigma_mutation1.05-sigma_only_generations100_(-0.6634310483932495, 0.0)_gen1000.pkl'

individual_path1 = 'saves/june11/mnist-cross_entropy-asex-mlp-p1-c256-e0-dsall-t0-pis7244424-s10.0001-s20.0001-hidden_dim128-initialization_seed7244424-mutationnormal-sigma_mutation1.05-sigma_only_generations-1_-0.43067601323127747.pkl'

#individual,policy,config=torch.load('saves/avpop_b500_scale5_exp2/mnist-cross_entropy-ave-mlp-p64-c256-e0-ds500-t0-dnaordmultdna-pis9903378-s10.002-s20.001-hidden_dim128-initialization_seed9903378-mutationnormal-sigma_mutation1.0-sigma_only_generations-1_val0.8763612508773804_gen4603.pt')
#dna=individual.dna
#policy = policy.to(device)
    
dna, config = pickle.load(open(individual_path1, 'rb'))
policy_args = config['eval_args']['policy_args']
eval_args = config['eval_args']
# Note, policy must be loaded with trainable=False so it can be reconstructed by 
# mutating from its dna.
    
policy = factory(dna, **policy_args)
original_policy = policy.clone()

# After loading, we set the parameters to trainable:
policy.l1.requires_grad = True
policy.l2.requires_grad = True



eval_args['num_train_datapoints'] = 'all'
all_evaler = MNIST(**eval_args)
(_, metadata),_ = all_evaler.eval(dna, val=False, cached_policy=policy)
print(metadata)
original_fitness = - metadata['train_loss']

evaler = MNIST(**eval_args)
evaler.train_batch_i = 0
evaler.train_batch_size = minibatch_size
evaler.num_train_datapoints = 59000 # just to enforce shuffling

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
    grads.append(p[1].grad.detach().clone().flatten())

traingrad = -1 * torch.cat(grads).flatten()


samples = 0
popsize=256

# Now, compute grad with respect to smaller minibatches
done = False

minibatch_vs_training_sims = []
best_of_pop_wrt_minibatch_vs_training_sims = []

bestX_parents_of_pop_wrt_minibatch_vs_training_sims = defaultdict(lambda: [])
bestX_parents_of_pop_wrt_minibatch_fitness_vs_original = defaultdict(lambda: [])

cosine_sim_vs_fitness = []

best_of_pop_vs_minibatch = []

magnitudes = defaultdict(lambda: [])

losses = []



pop = []
while True:
    x,y = evaler.get_batch_helper()
    #if samples % 100 == 0: print(samples)
    
    optimizer.zero_grad() 
    x = x.to(device)
    y = y.to(device)
    logits,intrinsic_fitness = policy(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    losses.append(loss.item())
    
    grads = []
    for p in policy.named_parameters():
        grads.append(p[1].grad.detach().clone().flatten())

    minibatch_grad = -1 * torch.cat(grads).reshape([-1])

    minibatch_vs_training_sims.append(torch.nn.functional.cosine_similarity(minibatch_grad, traingrad, dim=0).cpu())


    # take a random update to the parameters
    x1 = torch.normal(mean=0, std=1, size=minibatch_grad.shape, device=device)

    
    # NOTE we are still doing something different here,
    # we are ordering by similarity to minibatch gradient and then averaging, whereas in prod we 
    # instead order by fitness and then average. 
    # 
    # This is fundamentally different and results in oversampling easy minibatches

    if sort_with == 'fitness':
        fitness = -1 * loss
        pop.append((fitness, x1))
    else:
        sim = torch.nn.functional.cosine_similarity(x1, minibatch_grad, dim=0)
        pop.append((sim, x1))

    if len(pop) == popsize:
        print(samples)
    
        pop.sort(key=lambda x: x[0]) # make sure we never try to sort on the tensor

        bestsim = pop[-1][0]

        best_of_pop_vs_minibatch.append(bestsim)


        for numparents in numparents_l:

            org = pop[-numparents][-1] / numparents
            if numparents > 1:
              for (sim,p) in pop[-numparents+1:]:
                org += p / numparents

            magnitudes[numparents].append(torch.linalg.vector_norm(org.view([-1])))

            sim = torch.nn.functional.cosine_similarity(org, traingrad, dim=0).item()
            bestX_parents_of_pop_wrt_minibatch_vs_training_sims[numparents].append(sim)

            if eval_also:
                if update_as_we_go:
                    newpolicy = policy
                else:
                    newpolicy = original_policy.clone()

                dl1 = org[:newpolicy.l1.flatten().size()[0]]

                dl1 = torch.nn.functional.normalize(dl1, dim=0) * (dl1.shape[-1] ** 0.5) 
                dl1 = dl1.view(newpolicy.l1.shape).to(device) 

                dl2 = org[newpolicy.l1.flatten().size()[0]:]
                #print('dl2 norm before:', torch.norm(dl2))
                dl2 = torch.nn.functional.normalize(dl2, dim=0) * (dl2.shape[-1] ** 0.5) 
                dl2 = dl2.view(newpolicy.l2.shape).to(device) 

                #print('dl2 norm after:', torch.norm(dl2))

                for i,mult in enumerate(mults):

                    sigma1, sigma2 = (2e-3,1e-3)
                    #[X1, X2] with std 1
                    #[aX1, aX2] is the same as [X1, X2] with std a
                    with torch.no_grad():

                        newpolicy.l1 += dl1 * sigma1 * mult
                        newpolicy.l2 += dl2 * sigma2 * mult

                        (_,metadata),_ = all_evaler.eval(dna=None, val=False, cached_policy=newpolicy)
                        new_fitness = -metadata['train_loss']
                        newpolicy.l1 -= dl1 * sigma1 * mult
                        newpolicy.l2 -= dl2 * sigma2 * mult
                            

                            

                    fitdiff = new_fitness - original_fitness
                    cosine_sim_vs_fitness.append((
                        bestX_parents_of_pop_wrt_minibatch_vs_training_sims[numparents][-1],
                        fitdiff))

                    bestX_parents_of_pop_wrt_minibatch_fitness_vs_original[(numparents,mult)].append(fitdiff)
        if update_as_we_go:
          with torch.no_grad():
            newpolicy.l1 += dl1 * sigma1 * mult
            newpolicy.l2 += dl2 * sigma2 * mult
            original_fitness = new_fitness
            print(original_fitness)



        samples += 1

        pop = []

        if samples >= num_samples: 
            break



        fig, ax_lst = plt.subplots(2,2,figsize=(12,12))
        ax_lst = [ax_lst[0][0], ax_lst[0][1],ax_lst[1][0],ax_lst[1][1]]
        
        
        #for i,pname in enumerate(full_grad_dict.keys()):
        #    similarities = []
        #    for minibatch_gradient in grad_dict[pname]:
        #        a = full_grad_dict[pname].flatten()
        #        b = minibatch_gradient.flatten()
        #        sim = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
        #        similarities.append(sim)
        
        #make_histogram(ax_lst[i], similarities, pname)
        
        
        
        
        # Plot the minibatch gradient similarity to the training gradient 
        mb_vs_training_std = torch.std(torch.tensor(minibatch_vs_training_sims))
        mb_vs_training_mean = torch.mean(torch.tensor(minibatch_vs_training_sims))
        #ax_lst[1].hist(minibatch_vs_training_sims, bins=100, label=f'mb vs training. std: {mb_vs_training_std:.3e}, mean: {mb_vs_training_mean:.3e}', color='r', alpha=0.5, density=False)

        mean = torch.mean(torch.tensor(losses))
        std = torch.std(torch.tensor(losses))
        ax_lst[1].hist(losses, bins=100, label=f'losses. std: {std:.3e}, mean: {mean:.3e}', color='r', alpha=0.5, density=False)
        
        colors = [
            'red',
            'blue',
            'green',
            #'yellow',
            'cyan',
            'magenta',
            'sienna',
            'darkorange',
            'khaki',
            'olive',
            'darkolivegreen',
            'lawngreen',
            'mediumaquamarine',
            'teal',
            'deepskyblue',
            'steelblue',
            'slategray',
            'mediumpurple',
            'darkorchid',
            'pink',
            'k']

        
        for numparents,c in zip(numparents_l,colors):
        
            # Also interesting is the best similarity between 256 trials to just any random vector
            # (here, the interesting one is the training gradient). This is the target similarity I guess
            x = torch.tensor(bestX_parents_of_pop_wrt_minibatch_vs_training_sims[numparents])
            bop_vs_minibatch_std = torch.std(x)
            bop_vs_minibatch_mean = torch.mean(x)
            ax_lst[0].hist(x, bins=100, label=f'{numparents} parents, sim to train. std: {bop_vs_minibatch_std:.3e}, mean: {bop_vs_minibatch_mean:.3e}', color=c, alpha=0.5, density=False)

        i = 0
        for numparents in numparents_l:
          for mult in mults:
            c = colors[i]
            i += 1

            x = torch.tensor(bestX_parents_of_pop_wrt_minibatch_fitness_vs_original[(numparents,mult)])
            std = torch.std(x)
            mean = torch.mean(x)
            ax_lst[3].hist(x, bins=100, label=f'{numparents} parents, mult {mult}, fitness improvement. std: {std:.3e}, mean: {mean:.3e}', color=c, alpha=0.5, density=False)
        
            #x = torch.tensor(magnitudes[numparents])
            ##print('magnitudes:', x)
            #std = torch.std(x)
            #mean = torch.mean(x)
            #ax_lst[2].hist(x, bins=100, label=f'{numparents} parents, magnitude of sum. std: {std:.3e}, mean: {mean:.3e}', color=c, alpha=0.5, density=False)

        # plot cosine similarity vs fitness improvement
        x,y = zip(*cosine_sim_vs_fitness)
        ax_lst[2].scatter(x, y, label=f'cosine_sim_vs_fitness')

        
        ax_lst[0].set_title('Minibatch similarities')
        
        ax_lst[0].legend(loc='best', fontsize=8)
        ax_lst[1].legend(loc='best', fontsize=8)
        ax_lst[2].legend(loc='best', fontsize=8)
        ax_lst[3].legend(loc='best', fontsize=8)
        fig.savefig(imname)
        plt.close(fig)




