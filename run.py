import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DeblurNet


def main():
    EPOCH = 100
    BATCH_SIZE = 128
    INIT_LR = 0.001

    SavePeriod = 10
    IdxValid = 0

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    # ResultDir = "result/"
    # InputPath = "dataset/subject_to9_input_data.npy"
    # LabelPath = "dataset/subject_to9_label_data.npy"

    # DatasetInput, DatasetLabel = load_dataset(InputPath, LabelPath)
    # n_input = DatasetInput[0].shape[1]
    # n_label = DatasetLabel[0].shape[1]

    # train_input, train_label, valid_input, valid_label = unpack_dataset(DatasetInput, DatasetLabel, IdxValid)
    # print("Train data shape:", train_input.shape, train_label.shape)
    # print("Valid data shape:", valid_input.shape, valid_label.shape)

    # save_label(valid_label, ResultDir)

    # input_mean, input_std = extract_stat(train_input)
    # label_mean, label_std = extract_stat(train_label)

    # train_input = normalization(train_input, input_mean, input_std)
    # train_label = normalization(train_label, label_mean, label_std)
    # valid_input = normalization(valid_input, input_mean, input_std)
    # valid_label = normalization(valid_label, label_mean, label_std)

    # train_dataset = Dataset.CMU_Dataset(train_input, train_label, device=device)
    # valid_dataset = Dataset.CMU_Dataset(valid_input, valid_label, device=device)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_dataset,
    #                           batch_size=BATCH_SIZE, shuffle=False)
    x = torch.randn(1, 15, 720, 1280)
    model = DeblurNet()
    model(x)

    n_sample_train = train_dataset.n_sample
    lr_step_size = int(n_sample_train / BATCH_SIZE)

    loss_fn = torch.nn.MSELoss().to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=INIT_LR)
    #lr_sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=0.99)
    
    print("########## Start Train ##########")
    for idx_epoch in range(EPOCH+1):
        start_time = time.time()
        
        train_loss = 0.
        
        for idx_batch, (x, y) in enumerate(train_loader):
            model.zero_grad()

            x, y = x.to(device), y.to(device)
            output = model(x)

            loss = loss_fn(output, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            #lr_sch.step()
            
        train_loss /= idx_batch+1
        elapsed_time = time.time() - start_time
        
        print("\r %05d | Train Loss: %.7f | lr: %.7f | time: %.3f"%(idx_epoch+1, train_loss, optimizer.param_groups[0]['lr'], elapsed_time))
        
    
if __name__ == '__main__':
    main()