import torch
import torchvision
import torch.nn as nn

def main():
    decomp_type = 'tucker'
    from_epochs = range(10,210,10)
    dataset = 'cifar10'
    arch = 'vgg19'

    for from_epoch in from_epochs:
        # before decomposing
        before_decomp_record = get_stats_before_decompose(dataset, arch, decomp_type, from_epoch)

        # just after decomposing
        after_decomp_record = get_stats_after_decompose(dataset, arch, decomp_type, from_epoch)

        # after training decomposed
        after_training_decomp_record = get_stats_after_training_decomposed(dataset, arch, decomp_type, from_epoch)

        print("Epoch: ", from_epoch, "\n#params before: ", before_decomp_record['num_params'], "\tafter: ", after_training_decomp_record['num_params'])

def get_stats_before_decompose(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = '_' + str(from_epoch) if from_epoch < 200 else ''

    model_dir = str('./models/' + dataset + '/' + arch + '/' + 'no_decompose')

    model_file = str(model_dir + '/' + 'model.pth')
    checkpoint_file = str(model_dir + '/' + 'checkpoint' + from_epoch_label + '.pth.tar')
 
    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    #num_params1 = sum(p.numel() for p in state_dict.values()) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights = get_weights(model)

    return {'num_params': num_params, 'best_acc': best_acc, 'weights': weights}

def get_stats_after_decompose(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = str('from_epoch_' + str(from_epoch) + '_') if from_epoch < 200 else ''

    model_dir = str('./models/' + dataset + '/' + arch + '/' + from_epoch_label + decomp_type + '_decompose')

    model_file = str(model_dir + '/' + 'model.pth')
    #TODO: save checkpoint right after decomposing
    first_epoch = from_epoch + 10
    first_epoch_label = '_' + str(first_epoch) if first_epoch < 200 else ''
    checkpoint_file = str(model_dir + '/' + 'checkpoint' + first_epoch_label + '.pth.tar')

    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    #num_params1 = sum(p.numel() for p in state_dict.values()) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights = get_weights(model)

    return {'num_params': num_params, 'best_acc': best_acc, 'weights': weights}

def get_stats_after_training_decomposed(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = str('from_epoch_' + str(from_epoch) + '_') if from_epoch < 200 else ''

    model_dir = str('./models/' + dataset + '/' + arch + '/' + from_epoch_label + decomp_type + '_decompose')

    model_file = str(model_dir + '/' + 'model.pth')
    checkpoint_file = str(model_dir + '/' + 'checkpoint.pth.tar')
 
    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    #num_params1 = sum(p.numel() for p in state_dict.values()) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights = get_weights(model)

    return {'num_params': num_params, 'best_acc': best_acc, 'weights': weights}

def get_weights(model):
    weights = []
    for name, module in model._modules.items():
        if type(module) in [nn.Conv2d, nn.Linear]:
            weights.append(module.weight)

    return weights

if __name__ == '__main__':
    main()