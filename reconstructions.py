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
        elif type(module) == nn.Linear:
            linear_layers_list = [module]
            linear_names_list = [name]

            # add all consecutive conv layers to list
            item = next(iterator, None)
            while item is not None:
                name, module = item
                if type(module) == nn.Linear:
                    linear_layers_list.append(module)
                    linear_names_list.append(name)
                    item = next(iterator, None)
                else:
                    break
            
            # reconstruct
            if len(linear_layers_list) > 1:
                combined_weight = None
                for i, (layer, name) in enumerate(zip(linear_layers_list, linear_names_list)):
                    if i == 0:
                        combined_weight = layer.weight.data
                    else:
                        combined_weight = torch.matmul(layer.weight.data, combined_weight)

                    if i < len(linear_layers_list) - 1:
                        assert(layer.bias is None)
                        model._modules[name] = torch.nn.Identity() 
                    else:
                        assert(layer.bias is not None)
                        model._modules[name] = torch.nn.Linear(in_features = linear_layers_list[0].in_features, out_features = linear_layers_list[-1].out_features, bias = True)
                        model._modules[name].weight.data = combined_weight
                        model._modules[name].bias.data = linear_layers_list[-1].bias.data

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
                    if(len(conv_layers_list) == 3):
                        [last_layer, core_layer, first_layer] = conv_layers_list
                        [last_name, core_name, first_name] = conv_names_list 
                        first_weight = first_layer.weight.data.squeeze(-1).squeeze(-1)
                        core_weight = core_layer.weight.data
                        last_weight = torch.transpose(last_layer.weight.data, 1, 0).squeeze(-1).squeeze(-1) 
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
                    elif(len(conv_layers_list) == 2):
                        [core_layer, last_layer] = conv_layers_list
                        [core_name, last_name] = conv_names_list 
                        core_weight = core_layer.weight.data
                        last_weight = last_layer.weight.data.squeeze(-1).squeeze(-1) 
                        reconstructed_weight = tl.tucker_to_tensor(core_weight, [last_weight])

                        assert(core_layer.bias is None)
                        assert(last_layer.bias is not None)

                        reconstructed_bias = last_layer.bias.data

                        reconstructed_layer = torch.nn.Conv2d(in_channels=core_layer.in_channels, \
                                        out_channels=last_layer.out_channels, kernel_size=core_layer.kernel_size, stride=core_layer.stride,
                                        padding=core_layer.padding, dilation=core_layer.dilation, bias=True)
                        reconstructed_layer.weight.data = reconstructed_weight
                        reconstructed_layer.bias.data = reconstructed_bias

                        model._modules[core_name] = torch.nn.Identity()
                        model._modules[last_name] = reconstructed_layer
                        
        else:
            item = next(iterator, None)

    model.cuda()

    return model
