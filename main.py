import json, torch
import torch.optim as optim
from nn_gen import VAE
from data_gen import Data
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('-o', type=str, default='result_dir', metavar='N',
                    help='input result directory')
parser.add_argument('-n', type=int, default=100, metavar='N',
                    help='number of images to generate (default: 100)')
parser.add_argument('-param', type=str, default='param/param.json', metavar='N',
                    help='parameter json filepath (default: param/param.json)')

args = parser.parse_args()

if __name__ == '__main__':

    # Command line arguments
    #arg = sys.argv
    param_file = args.param

    # Hyperparameters from json file
    with open(param_file) as paramfile:
        param = json.load(paramfile)

    # Construct a model and dataset
    model= VAE()
    data= Data(param['data_file'])

    # Define an optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=param['learning_rate'])

    obj_vals= []
    cross_vals= []
    num_epochs= int(param['num_epochs'])

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0

        i = 0
        while i<len(data.x_train):
            x = data.x_train[i:(i+param['batch_size']),:,:,:]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss_function(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            i += param['batch_size']

        obj_vals.append(train_loss)

        # report progress in output stream
        if not ((epoch + 1) % param['display_epochs']):
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                  '\tTraining Loss: {:.4f}'.format(train_loss))

    # final report
    print('Final training loss: {:.4f}'.format(obj_vals[-1]))

    # Plot saved in results folder
    plt.plot(range(num_epochs), obj_vals, color="blue")
    plt.title('Training loss vs. epoch')
    plt.savefig('result_dir/loss.pdf')
    plt.close()

    # Generate n sample images from the model
    n = args.n
    with torch.no_grad():
        sample = torch.randn(n, 20)
        sample = model.decoder(sample)
        for i in range(n):
            sample_i = sample[i,:,:,:]    
            save_image(sample_i.view(14, 14),
                    'result_dir/%i.pdf'%(i+1))
    


