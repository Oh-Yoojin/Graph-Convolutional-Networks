import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout1, dropout2):
        super(GCN, self).__init__()
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.gc1 = GraphConvolution(nfeat, self.nhid1)
        self.gc2 = GraphConvolution(self.nhid1, self.nhid2)
        self.gc3 = GraphConvolution(self.nhid2, self.nhid3)
        self.gc4 = GraphConvolution(self.nhid3, nclass)
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout1)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout2)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout2)
#        x = F.relu(self.gc4(x, adj))
#        x = F.dropout(x, self.dropout2)
        x = F.relu(self.gc4(x,adj))
        return F.log_softmax(x, dim=1)
