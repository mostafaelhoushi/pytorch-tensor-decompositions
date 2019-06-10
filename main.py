import torch
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
from model_utils import train, test

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--cp", dest="cp", action="store_true", \
        help="Use cp decomposition. uses tucker by default")
    parser.add_argument("-b", "--batch-size", default=256, type=int,
                        metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                            "batch size of all GPUs on the current node when "
                            "using Data Parallel or Distributed Data Parallel")

    parser.set_defaults(train=False)
    parser.set_defaults(decompose=False)
    parser.set_defaults(fine_tune=False)
    parser.set_defaults(cp=False)    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    tl.set_backend("pytorch")

    train_data_loader = dataset.loader(args.train_path, args.batch_size)
    test_data_loader = dataset.test_loader(args.test_path, args.batch_size)

    if args.train:
        model = models.vgg16(pretrained=True).cuda() # ModifiedVGG16Model().cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)

        train(model, train_data_loader, test_data_loader, optimizer, epochs=10)
        torch.save(model, "model.pth")

    if args.test:
        if args.decompose:
            model = torch.load("decomposed_model.pth")
        else:
            model = models.vgg16(pretrained=True).cuda() # ModifiedVGG16Model().cuda()
        optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.99)
        
        test(model, test_data_loader)
        torch.save(model, "model.pth")

    elif args.decompose:
        model = models.vgg16(pretrained=True).cuda() # torch.load("model").cuda()
        model.eval()
        model.cpu()
        N = len(model.features._modules.keys())
        for i, key in enumerate(model.features._modules.keys()):

            if i >= N - 2:
                break
            if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
                conv_layer = model.features._modules[key]
                if args.cp:
                    rank = max(conv_layer.weight.data.numpy().shape)//3
                    decomposed = cp_decomposition_conv_layer(conv_layer, rank)
                else:
                    decomposed = tucker_decomposition_conv_layer(conv_layer)

                model.features._modules[key] = decomposed

            torch.save(model, "decomposed_model.pth")


    elif args.fine_tune:
        base_model = torch.load("decomposed_model.pth")
        model = torch.nn.DataParallel(base_model)

        for param in model.parameters():
            param.requires_grad = True

        print(model)
        model.cuda()        

        if args.cp:
            optimizer = optim.SGD(model.parameters(), lr=0.000001)
        else:
            # optimizer = optim.SGD(chain(model.features.parameters(), \
            #     model.classifier.parameters()), lr=0.01)
            optimizer = optim.SGD(model.parameters(), lr=0.001)

        train(model, train_data_loader, test_data_loader, optimizer, epochs=10)
        torch.save(model, 'fine_tuned_model.pth')

        test(model, test_data_loader)
        model.cuda()
        model.train()
        train(model, train_data_loader, test_data_loader, optimizer, epochs=100)
        torch.save(model, 'fine_tuned_model1.pth')
        model.eval()
        test(model, test_data_loader)