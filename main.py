import torch
from torchvision import models
import sys
import os
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
from decompositions import decompose_model, cp_decomposition_conv_layer, tucker_decomposition_conv_layer
from model_utils import train, test

model_names = torch.hub.list('pytorch/vision', force_reload=True)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='DIR',
        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
        choices=model_names,
        help="model architecture: " + str(model_names) + " (default: resnet50)")
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--decompose", dest="decompose", action="store_true")
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
    parser.add_argument("--load_model", dest="load_model")
    parser.add_argument("--load_opt", dest="load_opt")
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
    
    if args.load_model:
        model = torch.load(args.load_model)
    else:
        model = torch.hub.load('pytorch/vision', args.arch, pretrained=True).cuda() 
    
    train_path = os.path.join(args.data, "train")
    test_path = os.path.join(args.data, "val")
    
    if args.decompose:
        
        model.eval()
        model.cpu()
        model = decompose_model(model, args.cp)
        
        torch.save(model, "decomposed_model.pth")

    if args.train:
        train_data_loader = dataset.loader(train_path, args.batch_size)
        test_data_loader = dataset.test_loader(test_path, args.batch_size)
    
        model = torch.nn.DataParallel(model)
        
        for param in model.parameters():
            param.requires_grad = True
        
        if args.load_opt:
            optimizer = torch.load(args.load_opt)
        else:
            if args.decompose:
                if args.cp:
                    optimizer = optim.SGD(model.parameters(), lr=0.000001)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=0.001)
            else:
                optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99)

        torch.save(optimizer, "optimizer.pth")
        train(model, train_data_loader, test_data_loader, optimizer, epochs=10)
        torch.save(model, "trained_model.pth")
               
    elif args.test:
        test_data_loader = dataset.test_loader(test_path, args.batch_size)
                
        test(model, test_data_loader)
        torch.save(model, "tested_model.pth")
