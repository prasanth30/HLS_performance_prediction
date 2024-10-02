import os
import networkx as nx
import json
import numpy as np
import pandas as pd
import rich
from utils import Data_Generator, pragmas
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys
# sys.path.append('../Data/')
# base_path = '/home/karakanaidu/experiment/chip/Data/'
base_path = './../../Data'

dgn = Data_Generator(base_path)

class Design:
    """
    Design class to store design information
    """
    def __init__(self, kernel_name, version, design):
        self.kernel_name = kernel_name
        self.version = version
        self.design = design['point']
        self.graph = dgn.get_graph(kernel_name)
        self.src_code = dgn.get_src_code(kernel_name)
        self.valid = design['valid']
        
        self.perf = design['perf']
        self.res_util = design['res_util']
        self.latency = np.log(1e7/(1+self.perf))/2
        self.util_BRAM = self.res_util['util-BRAM']
        self.util_LUT = self.res_util['util-LUT']
        self.util_FF = self.res_util['util-FF']
        self.util_DSP = self.res_util['util-DSP']

        self.X = {
            'design': self.design,
            'version': self.version,
            #'graph': item.graph,
            'kernel_name': self.kernel_name,
            'src_code': self.src_code,
        }
        self.y = {'valid':self.valid,
            'perf':self.perf,
            'util-BRAM':self.res_util['util-BRAM'],
            'util-LUT':self.res_util['util-LUT'],
            'util-FF':self.res_util['util-FF'],
            'util-DSP':self.res_util['util-DSP'],
            'latency':self.latency,
        }
        
class Data_Generator():
    def __init__(self, base_path):
        self.all_graphs = {}
        self.all_src_codes = {}
        self.base_path = base_path
        
    def get_graph(self, kernel_name):
        if kernel_name in self.all_graphs.keys():
            return self.all_graphs[kernel_name]
        
        g_path = os.path.join(self.base_path, "train_data", "data", "graphs", f"{kernel_name}_processed_result.gexf")
        g = nx.read_gexf(g_path)
        self.all_graphs[kernel_name] = g
        return g

    def get_src_code(self, kernel_name):
        if kernel_name in self.all_src_codes.keys():
            return self.all_src_codes[kernel_name]
        
        g_path = os.path.join(self.base_path, "train_data", "data", "sources", f"{kernel_name}_kernel.c")
        with open(g_path, "r") as f:
            src = f.read()
        
        self.all_src_codes[kernel_name] = src
        return src
        
if __name__ == "__main__":
    dg = Data_Generator(base_path)
    g = dg.get_graph('2mm')
    print(g)