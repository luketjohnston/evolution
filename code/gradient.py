#!python
import numpy as np
import pickle
from evaluations import RandomDataloader, DumbDataloader
from collections import defaultdict
from policies import ConvPolicy, ConvModule, LinearPolicy
from codes import BasicDNA
import torch
from config import input_dims, kernel_dims, channels, strides, hidden_size, num_classes
import torchvision

from scipy.stats import shapiro

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

make_pdf = True
do_train = False # if not do_train, will just load data from pickle and remake pdf
#pdfname = 'images/test.pdf'
#pdfname = 'images/all_grads_together.pdf'
pdfname = 'images/tmp.pdf'

trainings = 100
    

target_val_acc = 0.95 # for mlp with 128 hidden size
#target_val_acc = 0.99 # for conv net


device = torch.device("mps")

dna = BasicDNA([])

input_dims=[28,28,1] # mnist
trials = 5
kernel_dims = [3,3]
channels = [1,32,64]
strides = [1,1,1]
hidden_size = 128
num_classes = 10

get_grad_stats = True

save_every = 1000 # used to save the gradients during training.


#policy_network = ConvModule(dna, input_dims, kernel_dims, channels, strides, num_classes, hidden_size).to(device)

if do_train:

    batch_size=64
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    training_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=True, transform = transform)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    
    val_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=False, transform = transform)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    
    grad_dict = defaultdict(lambda : [])
    epochs = 10
    
    for training in range(trainings):
        policy_network = LinearPolicy(dna, 28*28, hidden_size, num_classes, initialization_seed=training, sigma=None, trainable=True).to(device)
        optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(policy_network.parameters(), lr=0.1) # 6 epochs to 99
    
        counter = 0
    
        for epoch in range(epochs):
            print(f'Training {training} Epoch {epoch}')
        
            train_loss = 0
            val_loss = 0
        
            for i,(x,y) in enumerate(train_dataloader):
                counter += 1
                
                optimizer.zero_grad() 
                x = x.to(device)
                y = y.to(device)
                logits = policy_network(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                train_loss += loss.item()
                loss.backward()
        
                if counter % save_every == 1:
                    for p in policy_network.named_parameters():
                        grad_dict[(p[0], counter)].append(p[1].grad.detach().cpu().clone().numpy())
        
                optimizer.step()
        
        
        
            train_loss /= i
            print(f'  Ave train loss throughout epoch {epoch}: {train_loss}')
            print('  total training batches: ',i)
        
            correct = 0
            total = 0
            for i,(x,y) in enumerate(val_dataloader):
                total += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                logits = policy_network(x)
                correct += torch.sum(torch.argmax(logits, dim=1) == y)
                loss = torch.nn.functional.cross_entropy(logits, y)
                val_loss += loss.item()
            val_loss /= i
            print('  val acc:', correct / (total))
            print('  val loss:', val_loss)
            if correct / total > target_val_acc:
                break

    pickle.dump(dict(grad_dict), open(f'saves/mnist_sgd_param_updates.pkl', 'wb'))
else:
    grad_dict = pickle.load(open(f'saves/mnist_sgd_param_updates.pkl', 'rb'))


def plot_grads(ax, grads, title):
    ax.hist(grads, bins=100)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=10)

if make_pdf:
    with PdfPages(pdfname) as pdf:

        # compile all 

        fig, ax_lst = plt.subplots(3,2)
        ax_lst[2][0].set_xlabel('Gradient magnitude', fontsize=8)
        ax_lst[2][1].set_xlabel('Gradient magnitude', fontsize=8)

        ax_lst[0][0].set_ylabel('Count', fontsize=8)
        ax_lst[1][0].set_ylabel('Count', fontsize=8)
        ax_lst[2][0].set_ylabel('Count', fontsize=8)

        fig.suptitle("Histograms of gradient update distributions")

        for (param_name, counter),param_updates in grad_dict.items():
            print(param_name)
            lambda_estimates = []
            n = np.array(param_updates).size
            print("Lambda biased estimator for all updates, from wikipedia: ", (n - 2) / (n) * 1 / np.mean(np.abs(param_updates)))

            all_updates = np.concatenate(param_updates).flatten()
            if param_name == 'l1':
              j = 0
            elif param_name == 'l2':
              j = 1
            else: 
              print('PARAM NAME assert false:', param_name)
              assert False
            i = counter // save_every

            if i >= 3: continue

            title = f'Parameter {param_name}\nUpdate number {counter}'
            plot_grads(ax_lst[i][j], all_updates, title)



            #for update in param_updates:
            #    lambda_est = 1 / np.mean(np.abs(update))
            #    #print("biased estimator for lambda:", lambda_est)
            #    lambda_estimates.append(lambda_est)
            #    #print(update.shape)
            #    #print(shapiro(update))
            #    #print('max grad: ', np.max(update))
            #    #print('min grad: ', np.min(update))
            #    plt.hist(update.flatten(), bins=100)
            #    plt.title(param_name + ' update num: ' + str(update_num))
            #    plt.set_yscale('log')
            #    #axes = plt.gca()
            #    #axes.set_ylim([0,10])
            #    update_num += save_every
            #    pdf.savefig()
            #    plt.clf()



            #plt.plot(lambda_estimates)
            #plt.title('lambda estimate over time')
            #pdf.savefig()
            #plt.clf()

        plt.tight_layout()
        pdf.savefig(fig)

        fig, [normal_ax, exp_ax] = plt.subplots(1,2)
        fig.suptitle("Sampling from other distributions")
    
        # test a normal distribution and see what it looks like in log space
        x = torch.normal(mean=0.0, std=100.0, size=(9999999,)).cpu().numpy()
        plot_grads(normal_ax, x, 'Normal')
        normal_ax.set_xlabel('Gradient magnitude', fontsize=8)
        normal_ax.set_ylabel('Count', fontsize=8)
    
        # test an exponential distribution and see what it looks like in log space
        x = torch.zeros((999999,)).exponential_()
        x *= (torch.randint(2, size=x.shape) * 2 - 1)
        plot_grads(exp_ax, x, 'Double exponential')
        exp_ax.set_xlabel('Gradient magnitude', fontsize=8)

        plt.tight_layout()

        pdf.savefig(fig)
    
    
                
        
        
        
