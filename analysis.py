import torch
import torchvision
import torch.nn as nn
import os
from ptflops import get_model_complexity_info

base_dir = '.'
training_epochs = 200

def main():
    decomp_type = 'channel'
    from_epochs = range(10,training_epochs + 10,10)
    dataset = 'cifar10'
    arch = 'vgg19'

    for from_epoch in from_epochs:
        print("Epoch: ", from_epoch)

        before_decomp_record = get_stats_before_decompose(dataset, arch, decomp_type, from_epoch)
        after_decomp_record = get_stats_after_decompose(dataset, arch, decomp_type, from_epoch)
        after_training_decomp_record = get_stats_after_training_decomposed(dataset, arch, decomp_type, from_epoch)

        '''
        correlations = [pearsonr(wf.flatten(), wl.flatten()).item() for wf, wl in zip(after_decomp_record['weights'], after_training_decomp_record['weights'])]
        #cosine_similarity = [torch.nn.functional.cosine_similarity(wf.flatten(), wl.flatten()) for wf, wl in zip(after_decomp_record['weights'], after_training_decomp_record['weights'])]
        for wf, wl in zip(after_decomp_record['weights'], after_training_decomp_record['weights']):
            wf.flatten()
            wl.flatten()
            print(torch.nn.functional.cosine_similarity(wf.flatten(), wl.flatten()))
        '''
        
        if (before_decomp_record is not None):
            print("\tbefore decomp: ", "#params: ", "{:.2e}".format(before_decomp_record['num_params']), " flops: ", "{:.2e}".format(before_decomp_record['flops']), " training flops: ", "{:.2e}".format(before_decomp_record['training_flops']))
        if (after_decomp_record is not None):
            print("\tafter decomp: ", "#params: ", "{:.2e}".format(after_decomp_record['num_params']), " flops: ", "{:.2e}".format(after_decomp_record['flops']), " training flops: ", "{:.2e}".format(after_decomp_record['training_flops']))
        if (after_training_decomp_record is not None):
            print("\tafter training: ", "#params: ", "{:.2e}".format(after_training_decomp_record['num_params']), " flops: ", "{:.2e}".format(after_training_decomp_record['flops']), " training flops: ", "{:.2e}".format(after_training_decomp_record['training_flops']))
        if (after_decomp_record is not None and after_training_decomp_record is not None):
            print("\ttotal training flops: ", " flops: ", "{:.2e}".format(after_training_decomp_record['training_flops'] + before_decomp_record['training_flops']))

def get_params_flops(model, dataset, epochs):
    if dataset == 'cifar10':
        input_size = (3, 32, 32)
        num_examples = 50e3
    elif dataset == 'cifar100':
        input_size = (3, 32, 32)
        num_examples = 50e3
    elif dataset == 'imagenet':
        input_size = (3, 224, 224)
        num_examples = 1.2e6
    else:
        raise Exception('Unhandled dataset: ', dataset)
    
    flops, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)

    inference_flops = flops # two FLOPs per multiply and add
    # using equation from https://openai.com/blog/ai-and-compute/
    training_flops = inference_flops * 3 * num_examples * epochs

    return params, inference_flops, training_flops
    

def get_stats_before_decompose(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = '_' + str(from_epoch) if from_epoch < training_epochs else ''

    model_dir = os.path.join(base_dir, 'models', dataset, arch, 'no_decompose')

    if (os.path.exists(model_dir) is False):
        #print("\tUnable to find: ", model_dir)
        return None

    model_file = os.path.join(model_dir, 'model.pth')
    checkpoint_file = os.path.join(model_dir, 'checkpoint' + from_epoch_label + '.pth.tar')
 
    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    num_params, inference_flops, training_flops = get_params_flops(model, dataset, from_epoch - 0)
    weights = get_weights(model)

    return {'num_params': num_params, 'best_acc': best_acc, 'weights': weights, 'flops': inference_flops, 'training_flops': training_flops}

def get_stats_after_decompose(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = str('from_epoch_' + str(from_epoch) + '_') if from_epoch < training_epochs else ''

    model_dir = os.path.join(base_dir, 'models', dataset, arch, from_epoch_label + decomp_type + '_decompose')

    if (os.path.exists(model_dir) is False):
        #print("\tUnable to find: ", model_dir)
        return None

    model_file = os.path.join(model_dir, 'model.pth')
    #TODO: save checkpoint right after decomposing
    first_epoch = from_epoch + 10
    first_epoch_label = '_' + str(first_epoch) if first_epoch < training_epochs else ''
    checkpoint_file = os.path.join(model_dir, 'checkpoint' + first_epoch_label + '.pth.tar')

    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']

    num_params, inference_flops, training_flops = get_params_flops(model, dataset, from_epoch - from_epoch)
    weights = get_weights(model)

    return {'num_params': num_params, 'best_acc': best_acc, 'weights': weights, 'flops': inference_flops, 'training_flops': training_flops}

def get_stats_after_training_decomposed(dataset, arch, decomp_type, from_epoch):
    from_epoch_label = str('from_epoch_' + str(from_epoch) + '_') if from_epoch < training_epochs else ''

    model_dir = os.path.join(base_dir, 'models', dataset, arch, from_epoch_label + decomp_type + '_decompose')

    if (os.path.exists(model_dir) is False):
        #print("\tUnable to find: ", model_dir)
        return None

    model_file = os.path.join(model_dir, 'model.pth')
    checkpoint_file = os.path.join(model_dir, 'checkpoint.pth.tar')
 
    model = torch.load(model_file)
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['state_dict']
    best_acc = checkpoint['best_acc1']
    epochs = checkpoint['epoch']

    num_params, inference_flops, training_flops = get_params_flops(model, dataset, epochs - from_epoch if epochs - from_epoch > 0 else epochs)
    weights = get_weights(model)

    return {'num_params': num_params, 'best_acc': best_acc, 'weights': weights, 'flops': inference_flops, 'training_flops': training_flops}

def get_weights(model, weights=[]):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            weights = get_weights(module, weights)
        elif type(module) == nn.Conv2d or type(module) == nn.Linear:
            weights.append(module.weight)

    return weights

def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

if __name__ == '__main__':
    main()
