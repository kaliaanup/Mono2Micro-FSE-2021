'''
Created on Mar 27, 2021

@author: kaliaanup
'''
import json
import os
import shutil

if __name__ == '__main__':
    
    #initial config
    app = "App1"
    PROJECT_DIR = "/Users/kaliaanup/Desktop/pycharm-workspace/res-workspace/Mono2Micro_Benchmark_II/datasets_runtime/"
    ROOT = '$Root$' # $Root$ or Root

    runtime_call_volume = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "runtime_call_volume.json"), 'r') as f:
        runtime_call_volume = json.load(f)

    INPUT_DIR = os.path.join(PROJECT_DIR, app, "bunch_input")
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    os.makedirs(INPUT_DIR)
    
    
    with open(os.path.join(INPUT_DIR, app+"_bunch.mdg"), "w") as f:
        #generate bunch specific format
        for link in runtime_call_volume:
            src = link.split("--")[0]
            tgt = link.split("--")[1]
            
            if src.lower() == str(ROOT).lower() or tgt.lower() == str(ROOT).lower():
                    continue
    
            if src == tgt:
                continue 
            
            line = str(src)+" "+ str(tgt)+" "+str(runtime_call_volume[link])+"\n"
            f.write(line)
    
    f.close()
