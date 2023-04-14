import os
import json
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
from torch import nn
from Models import Encoder_Model, Decoder_Model, Decoder_Model_EBV
from utils import get_corrupted_triples, get_normalization_constant

#################################################################
# Set arguments (modify later to command line arguments)
#################################################################
data_path = './fb15k-237.json'
embed_dim = 32
basis_size = 2 # Basis size for relation matrices in encoder
sample_factor = 1 # For sampling corrupted triples
learning_rate = 1e-3
use_cuda = False
reg_param = 0.01 # for L2 regularization of the decoder
save_num_epochs = 10 # save the model at multiples of these many epochs
ebvecs_path = './EBV_Generation/eq_100_237.pkl'
save_path = './Model_Checkpoints_EBV/' # checkpoints during training
saved_model_path = './Model_Checkpoints_EBV/model_epoch_200.pth'
load_saved_model = True
start_epoch = 0
num_epochs = 200
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
true_triples = torch.tensor(data_dict['train_triples'], device = torch_device).to(torch.long)
print('Number of true training triples: {}'.format(true_triples.size()))
corrupt_triples = get_corrupted_triples(true_triples, num_nodes, sample_factor)
print('Number of corrupt training triples: {}'.format(corrupt_triples.size()))
e_list_true = torch.cat((true_triples[:, 0].view(-1, 1), true_triples[:, 2].view(-1, 1)), 1).T
e_type_true = true_triples[:, 1].view(1, -1)
e_list_corrupt = torch.cat((corrupt_triples[:, 0].view(-1, 1), corrupt_triples[:, 2].view(-1, 1)), 1).T
e_type_corrupt = corrupt_triples[:, 1].view(1, -1)
edge_list_pred  = torch.cat((e_list_true, e_list_corrupt), 1) # Includes true and corrupted triples
edge_type_pred = torch.cat((e_type_true, e_type_corrupt), 1)  # Includes true and corrupted triples
# Includes labels (1 or 0 respectively) for true and corrupted triples
train_labels = torch.cat((torch.ones(true_triples.size(0), dtype = torch.float), 
                          torch.zeros(corrupt_triples.size(0), dtype = torch.float)), 0) 
normc = get_normalization_constant(e_list_true, e_type_true) # Normalization constant for each edge
print('Completed preprocessing the data')

##########################################################
# Training part
##########################################################

def train_function(enc_model, dec_model, e_list_true, e_type_true, normc, e_list_pred, e_type_pred, train_labels, reg_param):
    print('Starting forward pass.')   
    X_embeds = enc_model(e_list_true, e_type_true, normc)
    triple_scores = dec_model(X_embeds, e_list_pred, e_type_pred)
    print('Completed forward pass')
    print('Computing loss')
    loss_func = nn.BCELoss()
    bce_loss = loss_func(triple_scores, train_labels)
    reg_loss = reg_param*dec_model.get_l2_norm()
    tot_loss = bce_loss + reg_loss
    print('BCE loss = {}, regularization loss = {}, total loss = {}'.format(bce_loss, reg_loss, tot_loss))
    pred = (triple_scores > 0.5).float()
    pred_acc = torch.mean((pred == train_labels).float())
    print('True / corrupted classification accuracy = {}'.format(pred_acc))
    return tot_loss, pred_acc

print('Setting up model and optimizer')
enc_model = Encoder_Model(num_nodes, embed_dim, num_rel, basis_size)
print('Done initializing encoder')
# dec_model = Decoder_Model(embed_dim, num_rel) # Use this for DistMult decoder
with open(ebvecs_path, 'rb') as file:
    ebvecs_prefined = pkl.load(file)
#dec_model = Decoder_Model_EBV(embed_dim, num_rel, ebvecs_predefined.size(1), ebvecs_predefined)
print('Encoder parameters:', list(enc_model.parameters()))
print('Decoder parameters:', list(dec_model.parameters()))
optimizer = torch.optim.Adam(list(enc_model.parameters()) + list(dec_model.parameters()), lr = learning_rate)

if load_saved_model:
    checkpoint = torch.load(saved_model_path)
    enc_model.load_state_dict(checkpoint['enc_model_state_dict'])
    dec_model.load_state_dict(checkpoint['dec_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if use_cuda:
    enc_model.cuda()
    dec_model.cuda()

print('Completed setting up model and optimizer')

for epoch in range(start_epoch+1, start_epoch+num_epochs+1):
    print('Epoch {} starts'.format(epoch))
    enc_model.train()
    dec_model.train()
    optimizer.zero_grad()
    
    loss, acc = train_function(enc_model, dec_model, e_list_true, e_type_true, normc,
                             edge_list_pred, edge_type_pred, train_labels, reg_param)
                             
    loss.backward()
    optimizer.step()
    
    print('Epoch = {}, Loss = {}'.format(epoch, loss))
    
    if epoch % save_num_epochs == 0:
        torch.save({
                    'epoch': epoch,
                    'enc_model_state_dict': enc_model.state_dict(),
                    'dec_model_state_dict': dec_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'acc': acc
                   }, os.path.join(save_path, 'model_epoch_{}.pth'.format(epoch)))
        
        