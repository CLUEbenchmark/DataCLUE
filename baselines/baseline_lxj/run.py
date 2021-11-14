
from utils import (
    MyDataset,
    BertForSequenceClassification,
    collate_fn,
    Stat,
    SeedEverything
)
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from time import time

# PRETRAIN = 'bert-base-chinese'
PRETRAIN = 'clue/roberta_chinese_clue_tiny'


def train(model, optimizer, loader, result, epoch, device, log_step, accumulation_steps):

    model.train()
    start_epoch = time()
    start_step = time()

    for step, batch in enumerate(loader):

        for key in batch:
            if key == 'idx': continue
            batch[key] = batch[key].to(device)

        loss, predict = model(**batch)

        if (step + 1) % log_step == 0:
            elapsed = (time() - start_step) / log_step
            msg = '| epoch {:1d} step {:1d} time {:.3f}s | acc: {:2.5f} |'.format(
                epoch + 1, step + 1, elapsed, sum(predict['acc']) / len(predict['acc']))
            print(msg)
            start_step = time()

        if (step + 1) % accumulation_steps == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
        
        for i, index in enumerate(batch['idx']):
            # NOTE, loss, acc, margin
            tmp = result.get(index, [[], [], []])
            tmp[0].append(predict['loss'][i])

            correct_logit = predict['logit'][i, batch['label'][i].item()]
            predict['logit'][i, :].sort()
            if predict['acc'][i]:
                tmp[1].append(1)
                margin = correct_logit - predict['logit'][i, -2]
            else:
                tmp[1].append(0)
                margin = correct_logit - predict['logit'][i, -1]

            tmp[2].append(margin)
            result[index] = tmp
    print('=' * 50)
    print(f'finished epoch {epoch} cost {time() - start_epoch:2.5f}s')


def test(model, loader, device):

    model.eval()
    acc = []
    for batch in loader:
        for key in batch:
            if key == 'idx': continue
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            _, predict = model(**batch)

        acc.extend(predict['acc'])
    
    print(f'test public acc: {sum(acc) / len(acc):2.4f}')
    print('=' * 50)


def main():

    filepath = '../datasets/cic'
    epochs = 100
    log_step = 50
    lr = 3e-5
    batch_size = 32
    accumulation_steps = 8
    save_dir = './' # TODO
    seed = 1

    SeedEverything(seed)
    
    traindataset = MyDataset(filepath = filepath, mode = 'train', tokenizer = PRETRAIN)
    testdataset = MyDataset(filepath = filepath, mode = 'test', tokenizer = PRETRAIN)

    trainloader = DataLoader(traindataset, batch_size = batch_size, collate_fn = collate_fn)
    testloader = DataLoader(testdataset, batch_size = batch_size, collate_fn = collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification(PRETRAIN, num_labels=118)
    model.to(device)


    optimizer = Adam(model.parameters(), lr = lr)
    # TODO no scheduler
    result = {}

    for epoch in range(epochs):
        train(model, optimizer, trainloader, result, epoch, device, log_step, accumulation_steps)
        test(model, testloader, device)


    # NOTE 统计完成后，去除数据重新训练，查看效果
    res = Stat(result)
    left_idx = res[-2][-1000:].tolist()
    traindataset = MyDataset(filepath = filepath, mode = 'train', tokenizer = PRETRAIN, left_idx = left_idx)
    trainloader = DataLoader(traindataset, batch_size = batch_size, collate_fn = collate_fn)

    model = BertForSequenceClassification(PRETRAIN, num_labels=118)
    model.to(device)
    optimizer = Adam(model.parameters(), lr = lr)
    
    for epoch in range(epochs):
        train(model, optimizer, trainloader, result, epoch, device, log_step, accumulation_steps)
        test(model, testloader, device)

    print('finished')


if __name__ == '__main__':
    main()