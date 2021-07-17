'''
Created on Mar 21, 2021

@author: kaliaanup
#mono2micro columns
#1589149456313,
#runtime_info,
#ConnectionManager returns to BookingServiceImpl OR  (ConnectionManager calls BookingServiceImpl)
#acmeair/src/main/java/com/acmeair/mongo/ConnectionManager.java:ConnectionManager:getConnectionManager,
#acmeair/src/main/java/com/acmeair/mongo/services/BookingServiceImpl.java:BookingServiceImpl:initialization

#fosci columns
traceID,order,structtype,method1,method2,m1_para,m2_para,class1,class2,m1_return,m2_return
#0, 
#480,
#null,
#net.jforum.SessionFacade.setAttribute,
#net.jforum.context.web.WebSessionContext.setAttribute,
#"java.lang.String,java.lang.Object",
#"java.lang.String,java.lang.Object",
#net.jforum.SessionFacade,
#net.jforum.context.web.WebSessionContext,
#void,
#void
#trace_id = elements[0] 
#order
#null
method_a = elements[3]
method_b = elements[4]
#null
#null  
class_a = elements[7]
class_b = elements[8]
#null
#null

'''
import os
import shutil

if __name__ == '__main__':
    
    #initial config
    application = "App1"
    datasets_runtime = "/Users/kaliaanup/Desktop/pycharm-workspace/res-workspace/Mono2Micro_Benchmark_II/datasets_runtime/"
    ROOT = '$Root$' # $Root$ or Root
    
    #input and output directories
    taggedDir = os.path.join(datasets_runtime, application, "mono2micro_input")
    outDir = os.path.join(datasets_runtime, application, "fosci_input")
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
    os.makedirs(outDir)
    
    ORIGINAL_TRACE_FILES = [f for f in os.listdir(taggedDir) if f.endswith('.txt')]
    TRACE_FILES = [os.path.join(taggedDir, f) for f in ORIGINAL_TRACE_FILES]
    
    input_file = open(os.path.join(outDir, "input_file.txt"), "w")
    input_file.write("traceID,order,structtype,method1,method2,m1_para,m2_para,class1,class2,m1_return,m2_return\n")  
    
    traceID=0
    valid_file = False
    for file in TRACE_FILES:
        if not file.endswith(".txt"):
            continue
    
        with open(file, "r") as f:
            #print("Reading processed file \"{}\".".format(file))
            sentences = f.readlines()
        
        order = 0
        
        for sentence in sentences:
            #handle inner classes in mono2micro by replacing :: with <>, in mono2micro we do not split inner classes from parent classes and hence 
            #for partitioning we ignore inner classes
            sentence = sentence.replace("::", "<>")
            
            if 'calls' in sentence and ROOT not in sentence:
                
                valid_file = True #when files have at least one trace where an app class calls another app class 
                attributes = sentence.split(',')
                #print(sentence)
                time = attributes[0]
                label = attributes[1].strip()
                call = attributes[2]
                source_class = str(call.split('calls')[0].strip())
                dest_class = str(call.split('calls')[1].strip())
                
                if "<>" in source_class:
                    inner_source_class = source_class.split("<>")[1]
                    source_class = source_class.split("<>")[0]+"::"+str(inner_source_class)
                    
                    
                if "<>" in dest_class:
                    inner_dest_class = dest_class.split("<>")[1]
                    dest_class = dest_class.split("<>")[0]+"::"+str(inner_dest_class)
                    
                #considering inner class method as parent class method
                #we addded path for package, although fosci uses it to separate classes however it does not use it
                source_method = "path."+source_class+"."+attributes[3].split(':')[2].strip()
                dest_method = "path."+dest_class+"."+attributes[4].split(':')[2].strip()
                
                input_file.write(str(traceID)+","+str(order)+",null,"+str(source_method)+","+str(dest_method)+",null,null,"+str(source_class)+","+str(dest_class)+",null,null\n")
                
                
                order+=1
                
        if valid_file: 
            traceID+=1
    input_file.close()      
                