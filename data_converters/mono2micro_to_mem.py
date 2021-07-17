import os
import re
import sys
import json
import logging
import subprocess
from typing import DefaultDict, Dict
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from ipdb import set_trace
import javalang
from collections import defaultdict
import sentencepiece as spm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Logging Config
logging.basicConfig(format='[+] %(message)s', level=logging.INFO)

# Add project source to path
ROOT = Path(os.path.abspath(os.path.join(
    os.getcwd().split('src')[0], 'src')))

if ROOT not in sys.path:
    sys.path.append(ROOT)


class Converter:
    def __init__(self, datasets_dir, verbose=False, visualize=False, use_root=False, keep_return=False):
        self.verbose = verbose
        self.use_root = use_root
        self.visualize = visualize
        self.keep_return = keep_return
        self.datasets_dir = datasets_dir
    
    def __enter__(self):
        """ Upon entry, create necessary directories
        """
        if self.verbose: logging.info("Processing {}".format(self.datasets_dir.stem))
        self.mono2micro_input_dir = self.datasets_dir.joinpath("mono2micro_input")
        self.output_path = self.datasets_dir.joinpath("mem_input")
        self.sources_path = ROOT.joinpath('sources', self.datasets_dir.stem)
        # Make dirs
        self.output_path.mkdir(parents=True, exist_ok=True)

        return self
    
    @staticmethod
    def _draw_call_graph(call_graph, fname="tmp.pdf"):
        """ Convert callgraph to a pdf

        Parameters
        ----------
        G: nx.DiGraph
            Input graph
        fname: str
            File name to save as
        """
        temp_dot = ".tmp.dot"
        try:
            nx.drawing.nx_pydot.write_dot(call_graph, temp_dot)
            subprocess.run(["dot", "-Tpdf", temp_dot, "-o", fname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            if os.path.exists(temp_dot):
                os.remove(temp_dot)

    @staticmethod
    def _calltrace_to_callgraph(trace_df, weighted: bool=False, keep_root: bool=False) -> nx.DiGraph:
        """ Convert all call traces to a callgraph

        Parameters
        ----------
        trace_df: pandas.DataFrame
            The call trace over several runs a dataframe 
        weighted: bool (default: False)
            If true, the edges will be weighted by how many times an edge is invoked.
        remove_root: bool (default: True)
            If true, the ROOT node (a common entrypoint to all methods) will be discarded.
        
        Returns
        -------
        nx.DiGraph
            Call graph
        """

        if not weighted:
            trace_df = trace_df.drop_duplicates()
        # Aggregate the run callgraph edge counts
        grouped = trace_df.groupby(trace_df.columns.tolist()).size().reset_index().rename(columns={0: 'label'})
        
        # Initialize and populate (weighted) directed-graph 
        directed_graph = nx.DiGraph()
        dynamic_callgraph = nx.from_pandas_edgelist(grouped, edge_attr=True, create_using=directed_graph)
        
        if not keep_root and dynamic_callgraph.has_node('Root'):
            dynamic_callgraph.remove_node('Root')

        return dynamic_callgraph
    
    def convert(self) -> None:
        traces = list()
        for trace_file in self.mono2micro_input_dir.glob("*.txt"):
            trace = pd.read_csv(trace_file, usecols=[2], index_col=None, header=None, names=['Path'])
            
            if self.keep_return:
                trace = trace.applymap(lambda x: re.sub(" returns to ", "-->", x))    
            else:
                # Remove the return calls 
                trace = trace[~trace.Path.str.contains(" returns to ")]
            
            # Unify 'calls' and/or 'returns to'
            trace = trace.applymap(lambda x: re.sub(" calls ", "-->", x))
            
            # Normalize <None>ROOT<None>
            trace = trace.applymap(lambda x: "Root" if x.find("None") > -1 else x) 
            # Create source and target columns
            trace[['source', 'target']] = trace['Path'].str.split('-->', 1, expand=True)
            trace = trace.drop('Path', axis=1)
            traces.append(trace)

        trace_dynamic_callgraphs = [self._calltrace_to_callgraph(trace, weighted=False, keep_root=self.use_root) for trace in traces]
        all_traces = pd.concat(traces)
        global_dynamic_callgraph = self._calltrace_to_callgraph(all_traces, weighted=True, keep_root=self.use_root)
        
        if self.visualize:
            self._draw_call_graph(global_dynamic_callgraph, fname="{}.pdf".format(self.datasets_dir.stem))
        
        self.class_names = {name: id_ for id_, name in enumerate(global_dynamic_callgraph.nodes())}
        self.mapping = {id_: name for name, id_ in self.class_names.items()}
        num_classes = len(self.class_names)

        self.co_occurance_matrix = np.eye(num_classes)

        # Process each trace individually
        for trace_cfg in trace_dynamic_callgraphs:
            for entrypoint in trace_cfg.nodes():
                for exitpoint in trace_cfg.nodes():
                    if entrypoint != exitpoint:
                        row = self.class_names[entrypoint]
                        col = self.class_names[exitpoint]
                        if nx.algorithms.simple_paths.is_simple_path(trace_cfg, [entrypoint, exitpoint]):
                            self.co_occurance_matrix[row, col] += 1

        # ----------------
        #  Generate TFIDF
        # ----------------
        corpus = len(self.mapping) * [""]
        for class_path in self.sources_path.glob("**/*/*.java"):
            class_name = class_path.stem
            if class_name not in self.class_names:
                continue
            row_id = self.class_names[class_name]
            with open(class_path, 'r') as java_file: 
                try:
                    java_file_str = java_file.read().encode('utf-8')
                except UnicodeDecodeError:
                    java_file_str = java_file.read().encode('latin-1')
                all_java_tokens = javalang.tokenizer.tokenize(java_file_str)
                tokenized_code = [t.value for t in all_java_tokens if t.__class__.__name__ == 'Identifier']
                tokenized_code = ' '.join(tokenized_code)
                processor = spm.SentencePieceProcessor()
                processor.Load(model_file=ROOT.joinpath('resources', 'sentencepiece.bpe.model').__str__())
                tokens = processor.Encode(tokenized_code, out_type=str)
                corpus[row_id] = " ".join(tokens)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        self.sim = cosine_similarity(X)


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Upon exit, save everything.
        """ 
        with open(self.output_path.joinpath('co_occurrence.txt'), 'w') as cooccurrence_file:
            for row_id, row in enumerate(self.co_occurance_matrix):
                for col_id, col in enumerate(row):
                    if row_id != col_id:
                        print(self.mapping[row_id], self.mapping[col_id], col, file=cooccurrence_file, sep=',')

        with open(self.output_path.joinpath('cosine_sim.txt'), 'w') as cosine_sim_file:
            for row_id, row in enumerate(self.sim):
                for col_id, col in enumerate(row):
                    if row_id != col_id:
                        print(self.mapping[row_id], self.mapping[col_id], col, file=cosine_sim_file, sep=',')

        with open(self.output_path.joinpath('class_names.txt'), 'w') as class_name_file:
            for _, class_name in self.mapping.items():
                print(class_name, file=class_name_file)

        

if __name__ == "__main__":
    dataset_base = ROOT.joinpath("datasets_runtime")
    
    opt = {
        "use_root": False,
        "keep_return": False,
        "verbose": True,
        "visualize": False, 
    }

    print("Run configurations: ")
    for k, v in opt.items():
        print("\t{}: {}".format(k,v), end="\n")

    for project_dir in dataset_base.glob("*"):
        if project_dir.is_dir():
            opt.update({"datasets_dir": project_dir})
            with Converter(**opt) as converter:
                converter.convert()
            
