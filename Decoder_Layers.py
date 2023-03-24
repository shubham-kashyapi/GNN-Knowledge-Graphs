import torch
from torch import nn

class DistMult_Layer(nn.Module):
    def __init__(self, xdim, num_rel):
        '''
        xdim (int)- Dimension of node embeddings
        num_rel (int)- Number of relations
        '''
        super().__init__()
        self.R = nn.Parameter(torch.empty(num_rel, xdim))
        nn.init.xavier_uniform_(self.R)
        
    def forward(self, X_feat, edge_list, edge_type):
        '''
        Xfeat- (num_nodes, xdim) tensor
        edge_list: (2, |E|) tensor in format [source, target]
        edge_type: (1, |E|) tensor
        
        Returns-
        scores: (1, |E|) score for every edge

        Note- We assume that all the relation matrices are diagonal, as in the DistMult paper.
        '''
        source_embed = torch.index_select(X_feat, 0, edge_list[0, :])
        R_edges = torch.index_select(self.R, 0, edge_type[0, :])
        target_embed = torch.index_select(X_feat, 0, edge_list[1, :])
        scores = torch.sum(source_embed*R_edges*target_embed, 1)
        return scores
