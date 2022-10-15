import torch
from rich.progress import track

def loop(model, dataloader, optimizer, s):
    for e in range(s.epochs):
        for inp, inp_mask, label in track(dataloader, description=f'Epoch {e}'):
            optimizer.zero_grad()
            predictions = model(inp.contiguous(), inp_mask.contiguous())
            loss = torch.nn.functional.binary_cross_entropy(predictions, label)

            loss.backward()
            optimizer.step()
            print(loss.item())

