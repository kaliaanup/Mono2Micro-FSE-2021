import os
import re
import sys
import json
import logging
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from pdb import set_trace
from collections import defaultdict

# Logging Config
logging.basicConfig(format='[+] %(message)s', level=logging.INFO)

# Add project source to path
root = Path(os.path.abspath(os.path.join(
    os.getcwd().split('src')[0], 'src')))

if root not in sys.path:
    sys.path.append(root)

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
        self.input_path = self.datasets_dir.joinpath("cogcn_input")
        self.output_path = self.datasets_dir.joinpath("cogcn_output")
        
        # Make dirs
        self.input_path.mkdir(parents=True, exist_ok=True)
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
            If true, the root node (a common entrypoint to all methods) will be discarded.
        
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
        """ 
        Convert static data and runtime traces to a format that can work with cogcn.
        """

        # ---------------------------
        #   1. Process call traces.
        # ---------------------------

        traces = list()
        for trace_file in self.mono2micro_input_dir.glob("*.txt"):
            trace = pd.read_csv(trace_file, usecols=[2], index_col=None, header=None, names=['Path'])
            
            if not self.keep_return:
                # Remove the return calls 
                trace = trace[~trace.Path.str.contains(" returns to ")]
            else:
                trace = trace.applymap(lambda x: re.sub(" returns to ", "-->", x))    
            
            # Unify 'calls' and/or 'returns to'
            trace = trace.applymap(lambda x: re.sub(" calls ", "-->", x))
            
            # Normalize <None>Root<None>
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
        
        self.class_names = defaultdict(None)
        self.mapping = defaultdict(None)
        for id_, name in enumerate(global_dynamic_callgraph.nodes()):
            self.class_names[name] =  id_
            self.mapping[id_] = name
        num_classes = len(self.class_names)

        self.entrypoint_matrix = np.zeros((num_classes, num_classes))
        self.inheritance_matrix = np.zeros((num_classes, num_classes))
        self.co_occurrence_matrix = np.eye(num_classes)

        # Process each trace individually
        for trace_cfg in trace_dynamic_callgraphs:
            for entrypoint in trace_cfg.nodes():
                for exitpoint in trace_cfg.nodes():
                    if entrypoint != exitpoint:
                        row = self.class_names[entrypoint]
                        col = self.class_names[exitpoint]
                        if nx.has_path(trace_cfg, entrypoint, exitpoint):
                            self.entrypoint_matrix[row, col] = 1
                            self.co_occurrence_matrix[row, col] += 1

        # --------------------------
        #   2. Process inheritance
        # --------------------------
        with open(self.datasets_dir.joinpath("refTable-rich.json"), "r+") as ref_table_file:
            ref_table = json.load(ref_table_file)

        inherit = ref_table["Inherit"]["Details"]
        for class_name, relationships in inherit.items():
            row_id = self.class_names[class_name]
            for rel_list in relationships.values():
                for class_name_2 in rel_list:
                    col_id = self.class_names[class_name_2]
                    if row_id is None or col_id is None:
                        continue
                    else:
                        self.inheritance_matrix[row_id, col_id] = 1

        self.cogcn_feature_matrix = np.concatenate((self.entrypoint_matrix, self.co_occurrence_matrix, self.inheritance_matrix)).T
        self.cogcn_feature_matrix *= self.cogcn_feature_matrix.sum(axis=1)[:, np.newaxis] ** -1

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Upon exit, save everything.
        """
        np.savetxt(self.input_path.joinpath('struct.csv'), self.entrypoint_matrix, delimiter=',', fmt="%d")
        np.savetxt(self.input_path.joinpath('content.csv'), self.cogcn_feature_matrix, delimiter=',', fmt="%f")
        np.savetxt(self.input_path.joinpath('inheritance.csv'), self.inheritance_matrix, delimiter=',', fmt="%d")
        np.savetxt(self.input_path.joinpath('cooccurrence.csv'), self.co_occurrence_matrix, delimiter=',', fmt="%d")
        # Save the mappings
        with open(self.output_path.joinpath("mapping.json"), 'w') as mapping_file:
            json.dump(self.mapping, mapping_file, indent=4)

if __name__ == "__main__":
    dataset_base = root.joinpath("datasets_runtime")
    
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
            
