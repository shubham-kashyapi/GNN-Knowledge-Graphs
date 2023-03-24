import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
from torch import nn

def get_corrupted_triples(triples_true, num_nodes, sample_factor):
    '''
    triples_true: (num_edges, 3) long tensor
    num_nodes: int
    sample_factor: int (number of corrupted triples to sample for every true triple)
    
    Returns:
    triples_corrupt: (sample_factor*num_edges, 3) long tensor
    '''
    num_triples = int(sample_factor)*triples_true.size(0)
    source_corrupt = torch.randint(num_nodes, size = (num_triples, 1))
    target_corrupt = torch.randint(num_nodes, size = (num_triples, 1))
    corrupt1 = torch.cat((source_corrupt, triples_true[:, 1:3].repeat(sample_factor, 1)), 1)
    corrupt2 = torch.cat((triples_true[:, 0:2].repeat(sample_factor, 1), target_corrupt), 1)
    triples_corrupt = torch.cat((corrupt1, corrupt2), 0)
    rand_indices = torch.randperm(triples_corrupt.size(0))
    triples_corrupt = triples_corrupt[rand_indices[0:num_triples], :]
    return triples_corrupt


def get_normalization_constant(e_list, e_type):
    '''
    e_list: (num_edges, 2) long tensor
    e_type: (num_edges, 1) long tensor
    
    Computes the normalization constant for each edge
    normc: (num_edges, 1) float tensor
    '''
    elist_types = torch.cat((e_list.T, e_type.reshape(-1, 1)), 1)
    # Degree of a target node corresponding to a particular relation type
    target_reltype = [(int(edge[1]), int(edge[2])) for edge in elist_types]
    counts = Counter(target_reltype)
    rel_degree = torch.tensor([counts[(int(edge[1]), int(edge[2]))] for edge in elist_types])
    normc = torch.pow(rel_degree.to(torch.float), -1)
    return normc
        
        
        