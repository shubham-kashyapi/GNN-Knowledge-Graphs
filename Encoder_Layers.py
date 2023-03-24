import torch
from torch import nn
from torch_geometric.nn import MessagePassing

class RGCN_Layer(MessagePassing):
    
    def __init__(self, xdim_in, xdim_out, num_rel, basis_size):
        '''
        xdim_in: int (Dimension of input node features). Number of nodes if no input features and one-hot
                 encoding is to be used
        xdim_out: int (Dimension of output node features)
        num_rel: int (Number of relation types)
        basis_size: int (Size of basis for Wr matrices)
        '''
        super().__init__(aggr = 'add', flow = 'source_to_target')
        self.xdim_in = xdim_in
        self.xdim_out = xdim_out
        self.num_rel = num_rel
        self.basis_size = basis_size
        ###########################
        # Self weight matrix
        ###########################
        self.Wo = nn.Parameter(torch.empty(xdim_in, xdim_out))
        nn.init.xavier_uniform_(self.Wo)
        ######################################
        # Basis matrices and relation weights
        ######################################
        self.Vb = nn.Parameter(torch.empty(basis_size, xdim_in, xdim_out))
        nn.init.xavier_uniform_(self.Vb)
        self.Ar = nn.Parameter(torch.empty(num_rel, basis_size))
        nn.init.xavier_uniform_(self.Ar)
        ######################################
        self.sigmoid = nn.Sigmoid()
        return
    
    def forward(self, x_feat, e_list, e_type, normc):
        '''
        x_input: None if one-hot encoding, otherwise tensor of size (num_nodes, num_features)
        e_list: (2, |E|) tensor in format [source, target]
        e_type: (1, |E|) tensor
        normc: (1, |E|) float tensor. Normalization constant for each edge
        '''        
        return self.propagate(e_list, x_input = x_feat, e_list = e_list, e_type = e_type, normc = normc)
    
    def message(self, x_input_j, e_list, e_type, normc):
        '''
        x_input_j: None if one-hot encoding, otherwise lifted input features for the source node of each edge
        e_list: (2, |E|) long tensor in the format [source, target]
        e_type: (1, |E|) long tensor
        '''
        W = torch.matmul(self.Ar, self.Vb.view(self.basis_size, -1)).view(self.Ar.size(0), self.xdim_in, self.xdim_out)
        if x_input_j is None:
            msg = normc.unsqueeze(1)*W[e_type[0, :], e_list[0, :], :]
        else:
            msg = normc.unsqueeze(1)*torch.matmul(x_input_j.unsqueeze(1), W[e_type[0, :], :, :]).squeeze()

        return msg
    
    def update(self, aggr, x_input):
        '''
        aggr: (num_nodes, xdim_out) float tensor- aggregated neighborhood features for each node 
        x_input: None if one-hot encoding, otherwise float tensor of size (num_nodes, num_features)
        '''
        if x_input is None:
            x_output = self.sigmoid(aggr + self.Wo)
        else:
            x_output = self.sigmoid(aggr + torch.matmul(x_input, self.Wo))
        
        return x_output
         