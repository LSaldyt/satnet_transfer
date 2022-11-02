import torch
import csv
from rich.progress import track
from torch.utils.data import *

def split(dataset, s):
    ''' Split a single dataset into test/train dataloaders '''
    l = len(dataset)
    train_n = int(l * s.split)
    sets    = random_split(dataset, [train_n, l - train_n])
    loaders = (DataLoader(ds, batch_size=s.batch_size) for ds in sets)
    return loaders

def epoch_loop(model, dataloader, optimizer, epoch_n, s, train=True):
    loss_agg, err_agg = 0, 0
    for inp, inp_mask, label in track(dataloader, description=f'Epoch {epoch_n}'):
        if train:
            optimizer.zero_grad()
        predictions = model(inp.contiguous(), inp_mask.contiguous())
        loss = (torch.nn.functional.binary_cross_entropy(predictions, label)
                / inp.shape[-1])
        if train:
            loss.backward()
            optimizer.step()
        bit_error = (torch.sum(torch.abs(torch.where(predictions > 0.5, 1.0, 0.) - label))
                     / torch.sum(inp_mask))
        loss_agg += loss.item()
        err_agg  += bit_error
        # print(inp, inp_mask, label)
        print('loss', loss.item(), 'error', bit_error)
    return loss_agg / len(dataloader), err_agg / len(dataloader)

def loop(model, dataset, optimizer, s, train=True):
    train_loader, test_loader = split(dataset, s)
    with open(s.metrics_file, 'w') as outfile:
        outfile.write('train_loss,train_error,test_loss,test_error,epoch\n') # headers
        writer = csv.writer(outfile)
        for e in range(s.epochs):
            train_loss, train_error = epoch_loop(model, train_loader, optimizer, e, s, train=True)
            test_loss,   test_error = epoch_loop(model, test_loader , optimizer, e, s, train=False)
            writer.writerow([train_loss,train_error,test_loss,test_error,e])
            outfile.flush()
