import torch
import time

from torch.autograd import Variable

def train(model, train_data_loader, test_data_loader, optimizer, epochs=10, criterion=torch.nn.CrossEntropyLoss()):
    model.cuda()

    for i in range(epochs):
        print("Epoch: ", i)
        for i, (batch, label) in enumerate(train_data_loader):
            model.train()
            model.zero_grad()
            batch = batch.cuda()
            label = label.cuda()
            input = Variable(batch)
            criterion(model(input), Variable(label)).backward()
            optimizer.step()
        test(model, test_data_loader)
    print("Finished fine tuning.")

def test(model, test_data_loader, criterion=torch.nn.CrossEntropyLoss(), print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_data_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    model.cuda()
    model.eval()
    correct = 0
    total = 0
    total_time = 0
    for i, (batch, label) in enumerate(test_data_loader):
        batch = batch.cuda()
        input = Variable(batch)

        t0 = time.time()
        output = model(input).cpu()
        t1 = time.time()

        total_time = total_time + (t1 - t0)
        pred = output.data.max(1)[1]
        correct += pred.cpu().eq(label).sum()
        total += label.size(0)

        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(t1 - t0)

        if i % print_freq == 0:
            progress.print(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    print("Average prediction time", float(total_time) / (i + 1), i + 1)

    return (losses.avg, top1.avg.cpu().numpy(), top5.avg.cpu().numpy(), batch_time.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
