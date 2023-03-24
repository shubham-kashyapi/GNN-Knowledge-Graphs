import os
import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from Models import Encoder_Model, Decoder_Model

class Link_Prediction_Metrics:
    
    def __init__(self, X_embed, dec_model, test_triples):
        '''
        X_embed- (num_nodes, embed_dim) float tensor. Computed using the training data
        dec_model- Gives scores for candidate triples
        test_triples- (num_triples, 3) long tensor
        metrics_req- List of required metrics (e.g. [('mean_rank'), ('mean_reciprocal_rank'), ('hits_at_k', 5), ...])
        '''
        self.X_embed = X_embed
        self.dec_model = dec_model
        self.test_triples = test_triples
        self.get_query_dict(test_triples)
        self.metric_to_method_map = {'mean_rank': self.rank, 'mean_reciprocal_rank': self.reciprocal_rank,
                                     'hits_at_k': self.hits_at_k}
        
    def get_query_dict(self, test_triples):
        '''
        test_triples- (num_triples, 3) long tensor

        Returns:
        left_queries, 
        Converts the triples to left and right dictionaries. Each key is a query and the value 
        corresponding to a query is a list of valid answers to the query
        '''
        left_queries, right_queries = dict(), dict()
        for triple in test_triples:
            left_q_key, right_q_key = (int(triple[0]), int(triple[1]), 'x'), ('x', int(triple[1]), int(triple[2]))
            if left_q_key not in left_queries.keys():
                left_queries[left_q_key] = [int(triple[2])]
            else:
                left_queries[left_q_key].append(int(triple[2]))
            if right_q_key not in right_queries.keys():
                right_queries[right_q_key] = [int(triple[0])]
            else:
                right_queries[right_q_key].append(int(triple[0]))

        self.left_queries = left_queries
        self.right_queries = right_queries
        return
        
    def compute_metrics(self, metrics_req):
        num_nodes = self.X_embed.size(0)
        leftq_metrics, rightq_metrics = dict(), dict()

        for left_query, correct_ans in tqdm(self.left_queries.items()):
            left_entities = torch.empty(1, num_nodes).fill_(left_query[0]).to(torch.long)
            edge_types = torch.empty(1, num_nodes).fill_(left_query[1]).to(torch.long)
            right_entities = torch.arange(num_nodes).to(torch.long).view(1, -1) # Candidate entities for the query
            edge_list = torch.cat((left_entities, right_entities), 0) 
            cand_ranks = torch.argsort(-1*self.dec_model(self.X_embed, edge_list, edge_types)).view(-1) # Sorted in decreasing order of scores
            
            leftq_metrics[left_query] = []
            for metric_tup in metrics_req:
                metric_val = self.metric_to_method_map[metric_tup[0]](cand_ranks, correct_ans, *metric_tup[1:])
                leftq_metrics[left_query].append(metric_val)
            

        for right_query, correct_ans in tqdm(self.right_queries.items()):
            right_entities = torch.empty(1, num_nodes).fill_(right_query[2]).to(torch.long)
            edge_types = torch.empty(1, num_nodes).fill_(right_query[1]).to(torch.long)
            left_entities = torch.arange(num_nodes).to(torch.long).view(1, -1) # Candidate entities for the query
            edge_list = torch.cat((left_entities, right_entities), 0) 
            cand_ranks = torch.argsort(-1*self.dec_model(self.X_embed, edge_list, edge_types)).view(-1) # Sorted in decreasing order of scores
            
            rightq_metrics[right_query] = []
            for metric_tup in metrics_req:
                metric_val = self.metric_to_method_map[metric_tup[0]](cand_ranks, correct_ans, *metric_tup[1:])
                rightq_metrics[right_query].append(metric_val)
        
        leftq_rel_metrics, rightq_rel_metrics = dict(), dict()
        leftq_rel_count, rightq_rel_count = dict(), dict()
        for left_query in self.left_queries:
            if left_query[1] not in leftq_rel_metrics.keys():
                leftq_rel_metrics[left_query[1]] = len(self.left_queries[left_query])*np.array(leftq_metrics[left_query])
                leftq_rel_count[left_query[1]] = len(self.left_queries[left_query])
            else:
                leftq_rel_metrics[left_query[1]] += (len(self.left_queries[left_query])*np.array(leftq_metrics[left_query]))
                leftq_rel_count[left_query[1]] += len(self.left_queries[left_query])
                
        for right_query in self.right_queries:
            if right_query[1] not in rightq_rel_metrics.keys():
                rightq_rel_metrics[right_query[1]] = len(self.right_queries[right_query])*np.array(rightq_metrics[right_query])
                rightq_rel_count[right_query[1]] = len(self.right_queries[right_query])
            else:
                rightq_rel_metrics[right_query[1]] += (len(self.right_queries[right_query])*np.array(rightq_metrics[right_query]))
                rightq_rel_count[right_query[1]] += len(self.right_queries[right_query])
        
        left_overall_metrics = (1.0/len(self.test_triples))*np.sum(np.array(list(leftq_rel_metrics.values())), axis = 0)
        right_overall_metrics = (1.0/len(self.test_triples))*np.sum(np.array(list(rightq_rel_metrics.values())), axis = 0)
        
        for rel in leftq_rel_metrics.keys():
            leftq_rel_metrics [rel] = (1.0/leftq_rel_count[rel])*leftq_rel_metrics[rel]
            
        for rel in rightq_rel_metrics.keys():
            rightq_rel_metrics[rel] = (1.0/rightq_rel_count[rel])*rightq_rel_metrics[rel]
        
        return leftq_rel_metrics, rightq_rel_metrics, left_overall_metrics, right_overall_metrics
    
    
    def hits_at_k(self, cand_ranks, correct_ans, k):
        num_hits = np.sum([True for idx in cand_ranks[:k] if (idx in correct_ans)])
        return num_hits
    
    def rank(self, cand_ranks, correct_ans, k):
        idx = 0
        for idx in range(len(cand_ranks)):
            if int(cand_ranks[idx]) in correct_ans:
                break
        return (idx+1)
    
    def reciprocal_rank(self, cand_ranks, correct_ans, k):
        idx = 0
        for idx in range(len(cand_ranks)):
            if int(cand_ranks[idx]) in correct_ans:
                break
        return 1.0/(idx+1)
