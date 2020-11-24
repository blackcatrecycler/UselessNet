import torch
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from model.sbnet import NMSLBlock,SBNet
import argparse

args = None

def main():
    model=SBNet(args.bnum,args.inpf).cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    t0=time.clock()
    while time.clock()-t0<args.time:
        inp = torch.rand(args.batch, 3, 100, 100).cuda()
        tuth = torch.randint(low=0,high=9,size=[args.batch]).cuda()
        optimizer.zero_grad()
        x=model(inp)
        loss = criterion(x, tuth)
        loss.backward()
        optimizer.step()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='gogogo')
    parser.add_argument('--bnum',default=10, type=int, help='模型的block数量')
    parser.add_argument('--inpf',default=1000, type=int, help='参与计算的feature通道数')
    parser.add_argument('--time',default=20, type=int, help='程序运行时间(s)')
    parser.add_argument('--batch', default=8, type=int, help='生成的input_batch')

    args = parser.parse_args()
    main()