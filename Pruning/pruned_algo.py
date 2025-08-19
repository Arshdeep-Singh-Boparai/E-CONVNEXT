#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:14:21 2025

@author: arshdeep
"""
# operator norm based pruning (under review in TASLP)
import numpy as np
import scipy
from scipy.stats.mstats import gmean
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def rank1_apporx(data):
    u,w,v= np.linalg.svd(data)
    M = np.matmul(np.reshape(u[:,0],(-1,1)),np.reshape(v[0,:],(1,-1)))
    M_prototype = M[:,0]/np.linalg.norm(M[:,0],2)
    return M_prototype

def operator_norm_pruning(W):
	C_M=[]	
	mean_vec=[]
	for c in range(np.shape(W)[1]):
		A=W[:,c,:]
		A_mean=np.mean(A,0)
		e=np.tile(A_mean,(np.shape(A)[0],1))
		A_centred=A-e
		mean_vec.append(A_mean)
		u,q,v=np.linalg.svd(A_centred)
		u1=np.reshape(u[:,0],(np.shape(A)[0],1))
		v1=np.reshape(v[0,:],(np.shape(A)[1],1))
		c_1=np.matmul(u1,v1.T)
		c_1_norm=c_1[0,:]/np.linalg.norm(c_1[0,:])
		C_M.append(c_1_norm)
	Score=[]
	for Nf in range(np.shape(W)[0]):
		Score.append(np.trace((np.matmul((W[Nf,:,:]-np.array(mean_vec)),np.array(C_M).T))))
	Mse_score=(np.array(Score))**2
	Mse_score_norm=Mse_score/np.max(Mse_score)
	return Mse_score_norm

#% entry-wise l_1 norm based scores
def ICLR_L1_Imp_index(W):
	Score=[]
	for Nf in range(np.shape(W)[0]):
		Score.append(np.sum(np.abs(W[Nf,:,0])))
	return Score/np.max(Score)

#% Geometric median based scores
def ICLR_GM_Imp_index(W):
	G_GM=gmean(np.abs(W.flatten()))
	Diff=[]
	for Nf in range(np.shape(W)[0]):
		F_GM=gmean(np.abs(W[Nf,:,:]).flatten())
		Diff.append((G_GM-F_GM)**2)
	return Diff/np.max(Diff)


def CS_Interspeech(Z):

    d,c,a,b=np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    
    A= np.reshape(Z,(-1,c,d)) # reshape filters
    
    
    N = np.zeros((a*b,d)) 
    
    for i in range(d):
        data= A[:,:,i]
        N[:,i]=rank1_apporx(data)
    
    W= np.zeros((d,d))
    
    for i in range(d):
        for j in range(d):
            W[i,j] = W[i,j] + distance.cosine(N[:,i],N[:,j])
    
    Q=[]
    S=[]
    for i in range(np.shape(W)[0]):
        n=np.argsort(W[i,:])[1]
        Q.append([i,n,W[i,n]])  # store closest pairs with their distance.
        S.append(W[i,n])   # store closest distance for each filter (ordered pairwise distance)
        

    Q_sort=[]
    q=list(np.argsort(S)) # save the indexes of filters with closest pairwise distance.
            
    for i in q: 
        Q_sort.append(Q[i]) # sort closest filter pairs.

    imp_list=[]
    red_list=[]
    
    for i in range(np.shape(W)[0]): 
        index_imp = Q_sort[i][0]
        index_red = Q_sort[i][1]
        if index_imp not in red_list:
            imp_list.append(index_imp)
            red_list.append(index_red)
    return imp_list



def filter_pruning_using_ranked_weighted_degree(filters, num_filters_to_prune, ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)
    print(cosine_similarities)
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))
    
    for i in range(num_filters):
        for j in range(i+1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])
    
    # Compute weighted degree centrality
    weighted_degree_centrality = {node: sum([G.edges[node, neighbor]['weight'] for neighbor in G.neighbors(node)]) for node in G.nodes()}
    
    # Rank the nodes
    ranked_nodes = sorted(weighted_degree_centrality.items(), key=lambda x: x[1], reverse=not ascending)
    
    # Identify filters to prune
    # filters_to_prune = [node for node, centrality in ranked_nodes[:num_filters_to_prune]]
    
    return ranked_nodes#,filters_to_prune

def filter_pruning_using_ranked_betweenness(filters, ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)
    
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))
    
    for i in range(num_filters):
        for j in range(i+1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])
    
    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    
    # Rank the nodes
    ranked_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=not ascending)
    
    # Identify filters to prune
    # filters_to_prune = [node for node, centrality in ranked_nodes[:num_filters_to_prune]]
    
    return ranked_nodes#, ranked_nodes[1,:]

def CS_WASPAA(Z):
    d,c,a,b=np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    A= np.reshape(Z,(-1,c,d)) # reshape filters
    N = np.zeros((a*b,d)) 
    for i in range(d):
        data= A[:,:,i]
        N[:,i]=rank1_apporx(data)
    # W= np.zeros((d,d))
    # It expects filling W or Z directly.
    sim = filter_pruning_using_ranked_weighted_degree(N.T,d)      #cosine_similarity(N.T)
    # scores = Kmeans_filters(sim,d)
    # nodes to keep
    return sim #, score#scores