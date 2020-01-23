import torch
import torchvision

def main():
    decomp_type = 'tucker'
    from_epochs = range(10,210,10)
    dataset = 'cifar10'
    arch = 'vgg19'

    for from_epoch in from_epochs:
        # before decomposing
        num_params_before_decomp, best_acc_before_decomp = get_stats_before_decompose(dataset, arch, decomp_type, from_epoch)

        # just after decomposing

        # after training decomposed
        num_params_after_train_decomp, best_acc_after_train_decomp = get_stats_after_training_decomposed(dataset, arch, decomp_type, from_epoch)

        print("Epoch: ", from_epoch, "\n#params before: ", num_params_before_decomp, "\tafter: ", num_params_after_train_decomp)

def get_stats_before_decompose(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = "_" + str(from_epoch) if from_epoch < 200 else ""

    model_dir = str('./models/' + dataset + '/' + arch + '/' + 'no_decompose')

    model_file = str(model_dir + '/' + 'model.pth')
    checkpoint_file = str(model_dir + '/' + 'checkpoint' + from_epoch_label + '.pth.tar')

 
    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    #num_params1 = sum(p.numel() for p in state_dict.values()) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_params, best_acc

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

    return num_params, best_acc

if __name__ == '__main__':
    main()