import os
import json
import numpy as np
from tqdm import tqdm
import pickle

import torch
from torch import nn
from Models import Encoder_Model, Decoder_Model
from utils import get_corrupted_triples, get_normalization_constant
from eval_functions import *

#################################################################
# Set arguments (modify later to command line arguments)
#################################################################
data_path = './fb15k-237.json'
embed_dim = 128
basis_size = 2 # Basis size for relation matrices in encoder
sample_factor = 1 # For sampling corrupted triples
use_cuda = False
saved_model_path = './Model_Checkpoints_Final/model_epoch_200.pth'
##########################################################
# Setting the device for pytorch
##########################################################
use_cuda = use_cuda and torch.cuda.is_available()
torch_device = torch.device('cuda') if use_cuda else torch.device('cpu')

##########################################################
# Data loading and preprocessing part
##########################################################
print('Preprocessing the data')
with open(data_path, 'r') as f:
    data_dict = json.loads(f.read())

num_nodes, num_rel = len(data_dict['node_index']), len(data_dict['rels_index'])
#train_triples = torch.tensor(data_dict['train_triples'], device = torch_device).to(torch.long) # Used for training
test_triples = torch.tensor(data_dict['test_triples'], device = torch_device).to(torch.long) # Used for inference
######################################################
# Using true and corrupted test triples
######################################################
print('Number of true test triples: {}'.format(test_triples.size()))
corrupt_triples = get_corrupted_triples(test_triples, num_nodes, sample_factor)
print('Number of corrupt training triples: {}'.format(corrupt_triples.size()))
e_list_true = torch.cat((test_triples[:, 0].view(-1, 1), test_triples[:, 2].view(-1, 1)), 1).T
e_type_true = test_triples[:, 1].view(1, -1)
e_list_corrupt = torch.cat((corrupt_triples[:, 0].view(-1, 1), corrupt_triples[:, 2].view(-1, 1)), 1).T
e_type_corrupt = corrupt_triples[:, 1].view(1, -1)
edge_list_pred  = torch.cat((e_list_true, e_list_corrupt), 1) # Includes true and corrupted triples
edge_type_pred = torch.cat((e_type_true, e_type_corrupt), 1)  # Includes true and corrupted triples
# Includes labels (1 or 0 respectively) for true and corrupted triples
test_labels = torch.cat((torch.ones(test_triples.size(0), dtype = torch.float), 
                          torch.zeros(corrupt_triples.size(0), dtype = torch.float)), 0) 
normc = get_normalization_constant(e_list_true, e_type_true) # Normalization constant for each edge
print('Completed preprocessing the data')

##########################################################
# Running inference
##########################################################
# Loading the saved model
enc_model = Encoder_Model(num_nodes, embed_dim, num_rel, basis_size)
dec_model = Decoder_Model(embed_dim, num_rel)
checkpoint = torch.load(saved_model_path)
enc_model.load_state_dict(checkpoint['enc_model_state_dict'])
dec_model.load_state_dict(checkpoint['dec_model_state_dict'])
    
if use_cuda:
    enc_model.cuda()
    dec_model.cuda()

# Running inference on the test set
enc_model.eval()
dec_model.eval()
with torch.no_grad():
    X_embeds = enc_model(e_list_true, e_type_true, normc) # Node embeddings using training data
    triple_scores = dec_model(X_embeds, edge_list_pred, edge_type_pred)
    pred = (triple_scores > 0.5).float()
    pred_acc = torch.mean((pred == test_labels).float())
#     link_pred_metrics = Link_Prediction_Metrics(X_embed, dec_model, true_triples)    
#     metrics_to_compute = [('mean_rank', 0), ('mean_reciprocal_rank', 0), ('hits_at_k', 1), 
#                           ('hits_at_k', 3), ('hits_at_k', 5), ('hits_at_k', 10)]
#     leftq_rel_metrics,rightq_rel_metrics,left_overall_metrics,right_overall_metrics = link_pred_metrics.compute_metrics(metrics_to_compute) 

# results_dict = {'left_rel': leftq_rel_metrics, 'right_rel': rightq_rel_metrics,
#                 'left_overall': left_overall_metrics, 'right_overall': right_overall_metrics}

# with open('./train_metrics_200.pkl', 'wb') as file:
#     pickle.dump(results_dict, file) 

print("True / false triple prediction accuracy on test set = {}".format(pred_acc))
    