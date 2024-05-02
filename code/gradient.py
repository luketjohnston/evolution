#!python
import numpy as np
from evaluations import RandomDataloader, DumbDataloader
from collections import defaultdict
from policies import ConvPolicy, ConvModule
from codes import BasicDNA
import torch
from config import input_dims, kernel_dims, channels, strides, hidden_size, num_classes
import torchvision

from scipy.stats import shapiro

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

device = torch.device("mps")


dna = BasicDNA(1)

input_dims=[28,28,1] # mnist
trials = 5
kernel_dims = [3,3]
channels = [1,32,64]
strides = [1,1,1]
hidden_size = 128
num_classes = 10

get_grad_stats = True


#policy_network = ConvPolicy(dna, input_dims, kernels, channels, strides, num_classes, hidden_size) 
policy_network = ConvModule(dna, input_dims, kernel_dims, channels, strides, num_classes, hidden_size).to(device)
 
#batch_size = 32
#num_train_datapoints = 32 * batch_size
#num_val_datapoints = 2 * batch_size
#train_dataloader  = RandomDataloader(input_dims, num_classes, num_train_datapoints, batch_size, seed=0, shuffle=True)
#val_dataloader    = RandomDataloader(input_dims, num_classes, num_val_datapoints, batch_size, seed=1, shuffle=False)


batch_size=64

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
])

training_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=True, transform = transform)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

val_data = torchvision.datasets.MNIST('./data/mnist', download=True, train=False, transform = transform)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)



#optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.001)
optimizer = torch.optim.SGD(policy_network.parameters(), lr=0.1) # 6 epochs to 99


#num_train_batches = num_train_datapoints // batch_size
#num_val_batches = num_val_datapoints // batch_size

save_every = 10
counter = 0

grad_dict = defaultdict(lambda : [])
epochs = 1


for epoch in range(epochs):
    print(f'Epoch {epoch}')

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
                grad_dict[p[0]].append(p[1].grad.detach().cpu().clone().numpy())


        #print(loss.item())
        optimizer.step()
        #print(loss.item())



    train_loss /= i
    print(f'Ave train loss throughout epoch {epoch}: {train_loss}')
    print('total training batches: ',i)

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
    print('val acc:', correct / (total))
    print('val loss:', val_loss)

    #print(f"Train loss: {train_loss}")
    #print(f"Val loss: {val_loss}")


if True:
    with PdfPages('images/test.pdf') as pdf:
        for param_name,param_updates in grad_dict.items():
            update_num = 0
            print(param_name)
            lambda_estimates = []
            n = np.array(param_updates).size
            print("Lambda biased estimator for all updates, from wikipedia: ", (n - 2) / (n) * 1 / np.mean(np.abs(param_updates)))
            for update in param_updates:
                lambda_est = 1 / np.mean(np.abs(update))
                #print("biased estimator for lambda:", lambda_est)
                lambda_estimates.append(lambda_est)
                #print(update.shape)
                #print(shapiro(update))
                #print('max grad: ', np.max(update))
                #print('min grad: ', np.min(update))
                plt.hist(update.flatten())
                plt.title(param_name + ' update num: ' + str(update_num))
                plt.yscale('log')
                #axes = plt.gca()
                #axes.set_ylim([0,10])
                update_num += save_every
                pdf.savefig()
                plt.clf()

            plt.plot(lambda_estimates)
            plt.title('lambda estimate over time')
            pdf.savefig()
            plt.clf()
    
        # test a normal distribution and see what it looks like in log space
        x = torch.normal(mean=0.0, std=100.0, size=(9999999,)).cpu().numpy()
        plt.hist(x)
        plt.yscale('log')
        plt.title('normal dist')
        pdf.savefig()
    
        plt.clf()
    
        # test an exponential distribution and see what it looks like in log space
        x = torch.zeros((999999,)).exponential_()
        plt.hist(x)
        plt.yscale('log')
        plt.title('exponential dist')
        pdf.savefig()
    
    
                
        
        
        
