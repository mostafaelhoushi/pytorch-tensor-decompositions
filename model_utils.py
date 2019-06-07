import torch
import time

from torch.autograd import Variable

def train(model, train_data_loader, test_data_loader, optimizer, epochs=10, criterion=torch.nn.CrossEntropyLoss()):
    for i in range(epochs):
        print("Epoch: ", i)
        train_epoch(model, train_data_loader, optimizer, criterion)
        test(model, test_data_loader)
    print("Finished fine tuning.")

def train_batch(model, batch, label, optimizer, criterion=torch.nn.CrossEntropyLoss()):
    model.train()
    model.zero_grad()
    input = Variable(batch)
    criterion(model(input), Variable(label)).backward()
    optimizer.step()

def train_epoch(model, train_data_loader, optimizer, criterion=torch.nn.CrossEntropyLoss()):
    for i, (batch, label) in enumerate(train_data_loader):
        train_batch(model, batch.cuda(), label.cuda(), optimizer, criterion)

def test(model, test_data_loader):
    model.cuda()
    model.eval()
    correct = 0
    total = 0
    total_time = 0
    for i, (batch, label) in enumerate(test_data_loader):
        batch = batch.cuda()
        t0 = time.time()
        output = model(Variable(batch)).cpu()
        t1 = time.time()
        total_time = total_time + (t1 - t0)
        pred = output.data.max(1)[1]
        correct += pred.cpu().eq(label).sum()
        total += label.size(0)
    
    print("Accuracy :", float(correct) / total)
    print("Average prediction time", float(total_time) / (i + 1), i + 1)

    model.train()

