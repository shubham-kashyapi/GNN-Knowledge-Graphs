import torch
from torch import nn
from Encoder_Layers import RGCN_Layer
from Decoder_Layers import DistMult_Layer, EBV_Layer

class Encoder_Model(nn.Module):
    
    def __init__(self, num_nodes, embed_dim, num_rel, basis_size):
        '''
        num_nodes: int (Number of nodes)
        embed_dim: int (Embedding dimension)
        num_rel: int (Number of relation types)
        basis_size: int (Size of basis for Wr relation matrices)      
        '''
        super().__init__()
        self.rgcn1 = RGCN_Layer(num_nodes, embed_dim, num_rel, basis_size)
        self.rgcn2 = RGCN_Layer(embed_dim, embed_dim, num_rel, basis_size)
        self.dropout = nn.Dropout()
    
    def forward(self, e_list_true, e_type_true, normc):
        '''
        e_list_true: (2, |E_true|) long tensor in format [source, target]. This includes true edges only (used for training).
        e_type_true: (1, |E_true|) long tensor. This includes true edges only. (used for training)
        normc: (1, |E_true|) float tensor. Normalization constant for each edge
        
        Note- We assume one-hot node features as input to the first RGCN layer
        
        Returns:
        X_embed- (num_nodes, embed_dim) float tensor
        '''
        X_embed = self.rgcn1(None, e_list_true, e_type_true, normc)
        X_embed = self.dropout(X_embed)
        X_embed = self.rgcn2(X_embed, e_list_true, e_type_true, normc)
        return X_embed
        
class Decoder_Model(nn.Module):
    
    def __init__(self, embed_dim, num_rel):
        '''
        embed_dim: int (Embedding dimension of nodes)
        num_rel: int (Number of relation types)
        '''
        super().__init__()
        self.decoder = DistMult_Layer(embed_dim, num_rel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_embed, edge_list_pred, edge_type_pred):
        '''
        Xfeat- (num_nodes, xdim) float tensor
        edge_list_pred: (2, |E|) long tensor in format [source, target]. May include true / corrupted edges.
        edge_type_pred: (1, |E|) long tensor. May include true / corrupted edges.
        
        Returns:
        scores- (1, |E|) float tensor. Decoder score for each edge after applying sigmoid activation. 
        '''
        scores_sig = self.sigmoid(self.decoder(X_embed, edge_list_pred, edge_type_pred))
        return scores_sig
    
    def get_l2_norm(self):
        '''
        Returns the l2 norm of the decoder parameters. Can be added as a regularization term to the cost function.
        '''
        return torch.sqrt(torch.sum(torch.square(self.decoder.R)))
    

class Decoder_Model_EBV(nn.Module):
    
    def __init__(self, embed_dim, num_rel, basis_dim, ebvecs):
        ''' 
        embed_dim: int (Embedding dimension of nodes)
        num_rel: int (Number of relation types)
        basis_dim: int (Dimension of Equiangular Basis Vectors)
        ebvecs: (num_cls, eb_dim) float tensor: Pre-initialized Equiangular Basis Vectors
        '''
        super().__init__()
        self.linear = nn.Linear(embed_dim, basis_dim, bias = False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.ebvlayer = EBV_Layer(num_rel, basis_dim, ebvecs)
        
    def forward(self, X_embed, edge_list_pred, edge_type_pred):
        '''
        Xfeat- (num_nodes, self.embed_dim) float tensor
        edge_list_pred: (2, |E|) long tensor in format [source, target]. May include true / corrupted edges.
        edge_type_pred: (1, |E|) long tensor. May include true / corrupted edges.
        
        Returns:
        scores- (1, |E|) float tensor. Probability score for each edge after applying EBV layer. 
        '''
        X_trans = self.linear(X_embed) # Transforming the node embeddings to the space of relation embeddings
        source_embed = torch.index_select(X_trans, 0, edge_list_pred[0, :])
        target_embed = torch.index_select(X_trans, 0, edge_list_pred[1, :])
        headtail_embed = source_embed-target_embed
        scores = self.ebvlayer(headtail_embed).gather(1, edge_type_pred.view(-1, 1)).view(1, -1)
        return scores  
    
    def get_l2_norm(self):
        '''
        Returns the l2 norm of the decoder parameters. Can be added as a regularization term to the cost function.
        '''
        return torch.sqrt(torch.sum(torch.square(self.linear.weight)))
    