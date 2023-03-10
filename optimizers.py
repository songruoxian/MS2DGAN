import itertools
import os
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def get_options():
    opt = Options().initialize()
    return opt

opt = get_options()

model_gnn = torch.nn.DataParallel(model_gnn(
    in_dim=opt.dim,
    hidden_dim=opt.dim,
    out_dim=opt.out_dim,
    dim=opt.dim,
)).cuda()

G_AB =  torch.nn.DataParallel(GeneratorAB(
    nfeat=opt.in_feature,
    nhid=opt.hidden1,
    nout=opt.hidden2,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE
)).cuda()

G_BA =  torch.nn.DataParallel(GeneratorBA(
    nfeat=opt.in_feature,
    nhid=opt.hidden1,
    nout=opt.hidden2,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE
)).cuda()

D_A = torch.nn.DataParallel(DiscriminatorA(
    input_size=opt.in_feature,
    output_size=1,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE,
    class_nums=opt.class_nums
)).cuda()

D_B = D = torch.nn.DataParallel(DiscriminatorB(
    input_size=opt.in_feature,
    output_size=1,
    dropout=opt.dropout,
    batch_size=opt.BATCH_SIZE,
    class_nums=opt.class_nums
)).cuda()


criterionIdt = torch.nn.L1Loss().cuda()
criterionCycle = torch.nn.L1Loss().cuda()
criterionGEN = torch.nn.L1Loss().cuda()

optimizer_G = torch.optim.SGD(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr_G,momentum=opt.momentum, weight_decay=opt.weight_decay)
optimizer_D = torch.optim.SGD(itertools.chain(D_A.parameters(), D_B.parameters()), lr=opt.lr_D,momentum=opt.momentum, weight_decay=opt.weight_decay)
optimizer_M = torch.optim.Adam(filter(lambda p: p.requires_grad, model_gnn.parameters()), lr=0.0001,
                                  weight_decay=0.001)





