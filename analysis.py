import torch
import torchvision

decomp_type = 'tucker'
from_epochs = range(10,210,10)
dataset = 'cifar10'
arch = 'resnet56'

for from_epoch in from_epochs:
    from_epoch_label = str('from_epoch_' + str(from_epoch) + '_') if from_epoch < 200 else ''

    model_dir = str('./models/' + dataset + '/' + arch + '/' + from_epoch_label + decomp_type + '_decompose')

    model_file = str(model_dir + '/' + 'model.pth')
    checkpoint_file = str(model_dir + '/' + 'checkpoint.pth.tar')


    #model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    num_params = sum(p.numel() for p in state_dict.values()) #len(model.parameters())
    print('#params: ', num_params, ' best_acc: ', best_acc)

    
