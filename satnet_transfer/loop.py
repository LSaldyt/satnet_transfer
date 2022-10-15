import torch
from rich.progress import track

def loop(model, dataloader, optimizer, s, train=True):
    for e in range(s.epochs):
        for inp, inp_mask, label in track(dataloader, description=f'Epoch {e}'):
            optimizer.zero_grad()
            predictions = model(inp.contiguous(), inp_mask.contiguous())
            loss = (torch.nn.functional.binary_cross_entropy(predictions, label)
                    / inp.shape[-1])
            if train:
                loss.backward()
                optimizer.step()
            bit_error = (torch.sum(torch.abs(torch.where(predictions > 0.5, 1.0, 0.) - label))
                         / torch.sum(inp_mask))
            print(loss.item(), bit_error)

