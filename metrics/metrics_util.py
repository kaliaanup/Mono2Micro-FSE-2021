'''
Created on Mar 27, 2021

@author: kaliaanup
'''
import numpy as np
from scipy.stats import entropy
import networkx as nx
import networkx.algorithms.community as nx_comm
from collections import defaultdict
import itertools
import math

def gen_class_assignment(partition, bc_per_class):

        class_bcs_partition_assignment = {}
        partition_class_bcs_assignment = {}
        for key, assignment in partition.items():

            assignment = str(assignment)
            if key not in class_bcs_partition_assignment:
                class_bcs_partition_assignment[key] = {}
                class_bcs_partition_assignment[key]["business_context"] = []

            class_bcs_partition_assignment[key]['final'] = assignment
            if key in bc_per_class:
                bcs = bc_per_class[key]
            else:
                bcs = ['Unknown']
            class_bcs_partition_assignment[key]["business_context"].extend(bcs)

            if assignment not in partition_class_bcs_assignment:
                partition_class_bcs_assignment[assignment] = {}
                partition_class_bcs_assignment[assignment]['classes'] = []
                partition_class_bcs_assignment[assignment]['business_context'] = []

            partition_class_bcs_assignment[assignment]["classes"].append(key)
            partition_class_bcs_assignment[assignment]['business_context'].extend([bc for bc in bcs if bc not in partition_class_bcs_assignment[assignment]['business_context']])

        return class_bcs_partition_assignment, partition_class_bcs_assignment

def business_context_purity(partition_class_bcs_assignment, result=None):
    #lower is better
        """ The entropy of business context. """
        if result == None:
            result = partition_class_bcs_assignment

        e = []
        for cls, value in result.items():
            bcs = value['business_context']
            entropy_val = entropy([1] * len(bcs))
            print(len(bcs), entropy_val)
            
            e.append(entropy_val)
        return round(np.mean(e), 3)
 
def business_context_purityII(partition_class_bcs_assignment, result=None):
    #lower is better
        """ The entropy of business context. """
        if result == None:
            result = partition_class_bcs_assignment

        e = []
        for cls, value in result.items():
            bcs = value['business_context']
            bcs_val = (1/len(bcs))*math.log(1/len(bcs))
            e.append(bcs_val)
        return round(np.mean(e), 3)
 
 
 
def inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume, result=None):
        """ The percentage of runtime call between two clusters. """
        #lower is better
        if result == None:
            result = class_bcs_partition_assignment

        n_total = 0
        n_inter = 0
        for call, volume in runtime_call_volume.items():
            src, target = call.split("--")
            if src.lower() == str(ROOT).lower() or target.lower() == str(ROOT).lower():
                continue

            if src == target:
                continue

            if src and target:
                
                src_assignment, target_assignment = result[src]['final'],  result[target]['final']
                n_total += np.log(volume) + 1
                if src_assignment != target_assignment:
                    n_inter += np.log(volume) + 1

        try:
            r = n_inter * 1.0/n_total
        except ZeroDivisionError:
            r = float("Inf")

        return round(r,3)
    
def structural_modularity(partition_class_bcs_assignment,runtime_call_volume, result=None):
        """ Structural modularity quality """
        #higehr is better

        if result == None:
            result = partition_class_bcs_assignment
        clusters = list(result.keys())
        n_clusters = len(clusters)

        # scoh: structural cohesiveness of a service
        scoh = []

        # scop: coupling between service
        scop = np.empty([len(result), len(result)], dtype=float)

        for m in range(n_clusters):
            value = result[clusters[m]]

            n_cls = len(value['classes'])
            mu = 0
            for i in range(n_cls-1):
                for j in range(i, n_cls):
                    c1 = value['classes'][i]
                    c2 = value['classes'][j]
                    if c1 + "--" + c2 in runtime_call_volume  or c2 +"--" + c1 in runtime_call_volume :
                        mu += 1
            try:
                scoh.append(mu * 1.0/(n_cls * n_cls))
            except ZeroDivisionError:
                scoh.append(float("Inf"))

        for m in range(n_clusters):
            for n in range(n_clusters):
                sigma = 0
                if m != n:
                    key1 = clusters[m]
                    key2 = clusters[n]
                    value1 = result[key1]
                    value2 = result[key2]
                    c_i = value1['classes']
                    c_j = value2['classes']
                    for i in range(len(c_i)):
                        for j in range(len(c_j)):
                            c1 = c_i[i]
                            c2 = c_j[j]

                            if c1 + "--" + c2 in runtime_call_volume or c2 + "--" + c1 in runtime_call_volume:
                                sigma += 1
                    try:
                        scop[m][n] = sigma * 1.0 / (2 * len(c_i) * len(c_j))
                    except ZeroDivisionError:
                        scop[m][n] = float("Inf")

        p1 = sum(scoh) * 1.0 / len(scoh)
        p2 = 0
        for i in range(len(scop)):
            for j in range(len(scop[0])):
                if i != j:
                    p2 += scop[i][j]

        if len(scop) == 1:
            p2 = 0
        else:
            try:
                p2 = p2 / len(scop) / (len(scop) - 1) * 2
            except ZeroDivisionError:
                p2 = float("Inf")

        # structural cohesion - structral coupling
        smq = p1 - p2
        return round(smq, 3)

def get_call_info(ROOT, runtime_call_volume):
    call_volume = {}
    nodes = []
    for link in runtime_call_volume:
        src = link.split("--")[0]
        tgt = link.split("--")[1]
        
        nodes.append(src)
        nodes.append(tgt)
                
        if src.lower() == str(ROOT).lower() or tgt.lower() == str(ROOT).lower():
                continue

        if src == tgt:
            continue 
        
        #src = link['source']
        #tgt = link['target']
        call_volume[(src, tgt)] = runtime_call_volume[link] 
    
    return list(set(nodes)), call_volume

def modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume, result=None):
    #higher is better
    
    if result == None:
            result = partition_class_bcs_assignment
            
    clusters = []
    
    for p in partition_class_bcs_assignment:
        clusters.append(partition_class_bcs_assignment[p]['classes'])
    
    nodes, call_volume = get_call_info(ROOT, runtime_call_volume)
    
    edges = []
    for c1, c2 in call_volume:  
        edges.append((c1, c2, call_volume[(c1, c2)]))
        
    mq = 0
    for i, c0 in enumerate(clusters):
        mu = 0
        for edge in edges:
            if edge[0] in c0 and edge[1] in c0:
                mu += 1
        
        if mu == 0: continue
        
        eps = 0
        for j, c1 in enumerate(clusters):
            if i == j: continue
            
            for edge in edges:
                if edge[0] in c0 and edge[1] in c1:
                    eps += 1
        
        mq += 2. * mu / (2 * mu + eps)
        
    return round(mq, 3)


def interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume, result=None):
    #lower is better
    
    if result == None:
            result = partition_class_bcs_assignment
            
    clusters = []
    
    for p in partition_class_bcs_assignment:
        clusters.append(partition_class_bcs_assignment[p]['classes'])
 
    nodes, call_volume = get_call_info(ROOT, runtime_call_volume)
    
    edges = []
    for c1, c2 in call_volume:  
        edges.append((c1, c2))
 
    K = len(clusters)
    i = 0
     
    for c0, c1 in itertools.combinations(clusters, 2):
        for x, y in itertools.product(c0, c1):
            if (x, y) in edges or (y, x) in edges:
                i += 1
     
    return round(i * 1. / K, 3)

def non_extreme_distribution(partition_class_bcs_assignment,result=None):
    
        if result == None:
            result = partition_class_bcs_assignment
        clusters = list(result.keys())
        n_clusters = len(clusters)
        
        sum = 0
        class_len = 0
        for cluster in clusters:
            size = len(partition_class_bcs_assignment[cluster]['classes'])
            class_len+=size
            if size >= 5 and size <=20:
                sum+=size
        
        ned = 1
        if class_len > 0 and sum >0:
            ned =  ned -  (sum/class_len)
        
        return round(ned,3)
                
            
            
            
