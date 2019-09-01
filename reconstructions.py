import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn

def reconstruct_model(model, cp=False):
    # TODO: Find a better way to avoid having to convert model from CPU to CUDA and back
    model.cpu()
    iterator = iter(model._modules.items())
    item = next(iterator, None)

    while item is not None:
        name, module = item

        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = reconstruct_model(model=module, cp=cp)
            item = next(iterator, None)
        elif type(module) == nn.Conv2d:
            conv_layers_list = [module]
            conv_names_list = [name]

            # add all consecutive conv layers to list
            item = next(iterator, None)
            while item is not None:
                name, module = item
                if type(module) == nn.Conv2d:
                    conv_layers_list.append(module)
                    conv_names_list.append(name)
                    item = next(iterator, None)
                else:
                    break
            
            # reconstruct
            if len(conv_layers_list) > 1:
                if cp:
                    raise("cp reconstruction not yet supported")
                else: # tucker reconstruction
                    assert(len(conv_layers_list)) == 3
                    [last_layer, core_layer, first_layer] = conv_layers_list
                    [last_name, core_name, first_name] = conv_names_list 

                    first_weight = first_layer.weight.data.squeeze(-1).squeeze(-1)
                    core_weight = core_layer.weight.data
                    last_weight = torch.transpose(last_layer.weight.data, 1, 0).squeeze(-1).squeeze(-1) 
                    print("first_weight: ", first_weight.shape, " last_weight: ", last_weight.shape)
                    reconstructed_weight = tl.tucker_to_tensor(core_weight, [first_weight, last_weight])

                    assert(first_layer.bias is not None)
                    assert(core_layer.bias is None)
                    assert(last_layer.bias is None)

                    reconstructed_bias = first_layer.bias.data

                    reconstructed_layer = torch.nn.Conv2d(in_channels=first_layer.in_channels, \
                                    out_channels=last_layer.out_channels, kernel_size=core_layer.kernel_size, stride=core_layer.stride,
                                    padding=core_layer.padding, dilation=core_layer.dilation, bias=True)
                    reconstructed_layer.weight.data = reconstructed_weight
                    reconstructed_layer.bias.data = reconstructed_bias

                    model._modules[first_name] = reconstructed_layer
                    model._modules[core_name] = torch.nn.Identity()
                    model._modules[last_name] = torch.nn.Identity()
        else:
            item = next(iterator, None)

    model.cuda()

    return model