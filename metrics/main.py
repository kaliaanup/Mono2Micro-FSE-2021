'''
Created on Mar 27, 2021

@author: kaliaanup
'''
from metrics import metrics_util
import json
import os

def gen_partitons(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    count = len(lines)
    partitions = {}

    index = 0
    for line in lines:
        class_str = line.split("=")[1]
        classes = class_str.split(",")
        for cls in classes:
            if cls not in partitions:
                partitions[cls.strip()] = index

        index += 1

    return partitions, count


def gen_mono2micro_format(datasets_runtime, application,count,partitions ):
#     partition = {}
#     for key in partitions:
#         partition[key] = str(partitions[key])
        
    with open(datasets_runtime+application+"/bunch_output/" + application + "_"+str(count)+".json", "w") as f:
        json.dump(partitions, f, indent=4)


if __name__ == "__main__": 
    
    PROJECT_DIR = "/Users/kaliaanup/Desktop/pycharm-workspace/res-workspace/Mono2Micro_Benchmark_II/datasets_runtime"
    
    #app name
    app = "daytrader"
    benchmark_type = "mono2micro"
    ROOT = 'Root' #$Root$
    
    #from mono2micro output dir get partition file, bcs_per_class, runtime_call_volume
    OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output")
    
    
    bcs_per_class = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "bcs_per_class.json"), 'r') as f:
        bcs_per_class = json.load(f)
    
    runtime_call_volume = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "runtime_call_volume.json"), 'r') as f:
        runtime_call_volume = json.load(f)
    
    #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35] #daytrader #Root
    partition_sizes = [5,7,9,11,13] #acmeair #Root
    #partition_sizes = [3,5,7,9,11,13,15,17]#jpetstore #Rootjdiff=3
    #partition_sizes = [2,4,6,8,10,12]#plants #Root
    #partition_sizes = [3,5,7,9,11,13,15,17,19,21]#socialsphere #$Root$
    #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27]#gasnet #$Root$ jdiff=1
    #partition_sizes = [2,4,8,16,32,64,128]#inps #$Root$
    partition = {}
    #generate metrics for mono2micro
    if benchmark_type == "mono2micro":
        
        for k in partition_sizes:
#             with open(os.path.join(OUTPUT_DIR, app+"_n_candidate_"+str(k)+"_repeat_4.csv"), "r") as f:
#                 partition_file = f.readlines() 
            with open(os.path.join(OUTPUT_DIR, "vertical_cluster_assignment_"+str(k)+".json"), 'r') as f:
                partition = json.load(f)
    
            print("-------------m2m metrics--------------")
        
        
            class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
            #print(partition_class_bcs_assignment)
            bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
            bcs2 = metrics_util.business_context_purityII(partition_class_bcs_assignment)
            icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
            #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
            mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ned = metrics_util.non_extreme_distribution(partition_class_bcs_assignment)
            
            
            print(str(bcs)+","+str(bcs2)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))
    
    
            
    #benchmark_type = "fosci"
    if benchmark_type == "fosci":
        OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output")
         
        print("-------------fosci metrics--------------")
    
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35] #daytrader #Root
        partition_sizes = [5,7,9,11,13] #acmeair #Root
        #partition_sizes = [3,5,7,9,11,13,15,17]#jpetstore #Rootjdiff=3
        #partition_sizes = [2,4,6,8,10,12]#plants #Root
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21]#socialsphere #$Root$
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27]#gasnet #$Root$ jdiff=1
        #partition_sizes = [2,4,8,16,32,64,128]#inps #$Root$
        
        
        for k in partition_sizes:
            with open(os.path.join(OUTPUT_DIR, app+"_n_candidate_"+str(k)+"_repeat_4.csv"), "r") as f:
                partition_file = f.readlines() 
              
                for line in partition_file:
                    line = line.replace("\n", "")
                    class_name, partition_id = line.split(",")
                    partition[class_name] = partition_id
                     
            class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
            bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
            icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
            #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
            mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ned = metrics_util.non_extreme_distribution(partition_class_bcs_assignment)
            
            print(str(k)+","+str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))
    
    
    #benchmark_type = "cogcn"
    if benchmark_type == "cogcn":
        OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output")
         
        print("-------------cogcn metrics--------------")
    
        partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35] #daytrader #Root
        #partition_sizes = [5,7,9,11,13] #acmeair #Root
        #partition_sizes = [3,5,7,9,11,13,15,17]#jpetstore #Rootjdiff=3
        #partition_sizes = [2,4,6,8,10,12]#plants #Root
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21]#socialsphere #$Root$
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27]#gasnet #$Root$ jdiff=1
        #partition_sizes = [2,4,8,16,32,64,128]#inps #$Root$
        
        
        for k in partition_sizes:
            
            partition = {}
            with open(os.path.join(OUTPUT_DIR, "vertical_cluster_assignment__"+str(k)+".json"), 'r') as f:
                partition = json.load(f)
                
            
                     
            class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
            bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
            icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
            #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
            mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ned = metrics_util.non_extreme_distribution(partition_class_bcs_assignment)
            
            print(str(k)+","+str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))
            
    if benchmark_type == "mem":
        OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output")
         
        print("-------------mem metrics--------------")
    
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35] #daytrader #Root
        #partition_sizes = [5,7,9,11,13] #acmeair #Root
        #partition_sizes = [3,5,7,9,11,13,15,17]#jpetstore #Rootjdiff=3
        #partition_sizes = [2,4,6,8,10,12]#plants #Root
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21]#socialsphere #$Root$
        #partition_sizes = [3,5,7,9,11,13,15,17,19,21,23,25,27]#gasnet #$Root$ jdiff=1
        partition_sizes = [2,4,8,16,32,64,128]#inps #$Root$
        
        
        for k in partition_sizes:
            
            partition = {}
            with open(os.path.join(OUTPUT_DIR, "vertical_clustering__"+str(k)+".json"), 'r') as f:
                partition = json.load(f)
                
                     
            class_bcs_partition_assignment, partition_class_bcs_assignment = metrics_util.gen_class_assignment(partition, bcs_per_class)
            bcs = metrics_util.business_context_purity(partition_class_bcs_assignment)
            icp = metrics_util.inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
            sm = metrics_util.structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
            #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
            mq = metrics_util.modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ifn = metrics_util.interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
            ned = metrics_util.non_extreme_distribution(partition_class_bcs_assignment)
            
            print(str(k)+","+str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))
    
    
#     if benchmark_type == "bunch":
#         filename = datasets_runtime+application+"/bunch_input/" + application+"_bunch_top.mdg.bunch"
#         partitions, count = gen_partitons(filename)
#         gen_mono2micro_format(datasets_runtime, application, count, partitions)
#     
#             
#         class_bcs_partition_assignment, partition_class_bcs_assignment = gen_class_assignment(partitions, bcs_per_class)
#         bcs = business_context_purity(partition_class_bcs_assignment)
#         icp = inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
#         sm = structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
#         #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
#         mq = modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
#         ifn = interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
#         ned = non_extreme_distribution(partition_class_bcs_assignment)
#         print("k="+str(count)+"| "+str(bcs)+" |"+str(icp)+"| "+str(sm)+" |"+str(mq)+"| "+str(ifn)+" |"+str(ned)+" |"+str(round((0),3)))
#         
#     
#         filename = datasets_runtime+application+"/bunch_input/" + application+"_bunch_median.mdg.bunch"
#         partitions, count = gen_partitons(filename)
#         gen_mono2micro_format(datasets_runtime, application, count, partitions)
#         
#         class_bcs_partition_assignment, partition_class_bcs_assignment = gen_class_assignment(partitions, bcs_per_class)
#         bcs = business_context_purity(partition_class_bcs_assignment)
#         icp = inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
#         sm = structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
#         #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
#         mq = modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
#         ifn = interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
#         ned = non_extreme_distribution(partition_class_bcs_assignment)
#         print("k="+str(count)+"| "+str(bcs)+" |"+str(icp)+"| "+str(sm)+" |"+str(mq)+"| "+str(ifn)+" |"+str(ned)+" |"+str(round((0),3)))
#         
#         
#         filename = datasets_runtime+application+"/bunch_input/" + application+"_bunch_detailed.mdg.bunch"
#         partitions, count = gen_partitons(filename)
#         gen_mono2micro_format(datasets_runtime, application, count, partitions)
#         
#         class_bcs_partition_assignment, partition_class_bcs_assignment = gen_class_assignment(partitions, bcs_per_class)
#         bcs = business_context_purity(partition_class_bcs_assignment)
#         icp = inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
#         sm = structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
#         #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
#         mq = modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
#         ifn = interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
#         ned = non_extreme_distribution(partition_class_bcs_assignment)
#         print("k="+str(count)+"| "+str(bcs)+" |"+str(icp)+"| "+str(sm)+" |"+str(mq)+"| "+str(ifn)+" |"+str(ned)+" |"+str(round((0),3)))
