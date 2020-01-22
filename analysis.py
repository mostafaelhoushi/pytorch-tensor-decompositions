import torch
import torchvision

decomp_type = 'tucker'
from_epochs = range(10,200,10)
dataset = 'cifar10'
arch = 'resnet56'

for from_epoch in from_epochs:
    model_dir = str('./models/' + dataset + '/' + arch + '/from_epoch_' + str(from_epoch) + '_' + decomp_type + '_decompose')

    model_file = str(model_dir + '/' + 'model.pth')
    checkpoint_file = str(model_dir + '/' + 'checkpoint.pth.tar')


    #model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']

    num_params = sum(p.numel() for p in state_dict.values()) #len(model.parameters())
    print(num_params)

    
