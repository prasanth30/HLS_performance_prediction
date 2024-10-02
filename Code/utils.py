import os
import torch
import networkx as nx
import numpy as np


from sklearn.metrics import f1_score

class pragmas:
    pragmas = ['__PARA__L0', '__PARA__L0_0', 
               '__PARA__L0_1', '__PARA__L0_1_0', 
               '__PARA__L0_2', '__PARA__L0_2_0', 
               '__PARA__L0_3', '__PARA__L0_3_0', 
               '__PARA__L1', '__PARA__L2', '__PARA__L3', '__PARA__L4', '__PARA__L5', '__PARA__L6', 
               '__PARA__L7', '__PARA__L7_0', 
               '__PARA__L8', 
               '__PIPE__L0', '__PIPE__L0_1', '__PIPE__L0_2', '__PIPE__L0_3', 
               '__PIPE__L1', '__PIPE__L2', '__PIPE__L3', '__PIPE__L4', '__PIPE__L5', '__PIPE__L7', 
               '__TILE__L0', '__TILE__L0_1', '__TILE__L0_2', '__TILE__L0_3', 
               '__TILE__L1', '__TILE__L2', '__TILE__L3', '__TILE__L4', '__TILE__L5']
    categorical_space = [
                '__PIPE__L0','__PIPE__L1','__PIPE__L2','__PIPE__L3',
                '__PIPE__L0_1','__PIPE__L0_2','__PIPE__L0_3',
                '__PIPE__L4','__PIPE__L5','__PIPE__L7',
    ]
    integer_space = [
                '__PARA__L0', 
                '__PARA__L0_0', '__PARA__L0_1', '__PARA__L0_1_0', 
                '__PARA__L0_2', '__PARA__L0_2_0', '__PARA__L0_3', '__PARA__L0_3_0', 
                '__PARA__L1', '__PARA__L2', '__PARA__L3', '__PARA__L4', '__PARA__L5', '__PARA__L6', '__PARA__L7', 
                '__PARA__L7_0', '__PARA__L8', 
                '__TILE__L0', '__TILE__L0_1', '__TILE__L0_2', '__TILE__L0_3', 
                '__TILE__L1', '__TILE__L2', '__TILE__L3', '__TILE__L4', '__TILE__L5'
    ]
    mx_range = {
        '__PARA__L0': 494,
        '__PARA__L0_0': 124,
        '__PARA__L0_1': 124,
        '__PARA__L0_1_0': 32,
        '__PARA__L0_2': 60,
        '__PARA__L0_2_0': 79,
        '__PARA__L0_3': 59,
        '__PARA__L0_3_0': 79,
        '__PARA__L1': 410,
        '__PARA__L2': 410,
        '__PARA__L3': 400,
        '__PARA__L4': 400,
        '__PARA__L5': 400,
        '__PARA__L6': 400,
        '__PARA__L7': 239,
        '__PARA__L7_0': 100,
        '__PARA__L8': 50,
        '__TILE__L0': 494,
        '__TILE__L0_1': 59,
        '__TILE__L0_2': 60,
        '__TILE__L0_3': 59,
        '__TILE__L1': 410,
        '__TILE__L2': 240,
        '__TILE__L3': 400,
        '__TILE__L4': 199,
        '__TILE__L5': 70
    }
    space = {
        '__PARA__L0':[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 24, 25, 26, 29, 30, 32, 40, 60, 64, 80, 90, 
100, 116, 120, 129, 200, 250, 390, 400, 410, 494],
        '__PARA__L1':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 29, 30, 31, 32, 40, 
        50, 58, 64, 70, 80, 88, 90, 100, 118, 120, 124, 129, 220, 240, 250, 390, 400, 410],
        '__PARA__L2':[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 15, 16, 20, 22, 24, 25, 26, 29, 30, 32, 40, 50, 58, 60, 64, 80, 
        88, 100, 116, 118, 120, 128, 199, 200, 240, 390, 400, 410],
        '__PARA__L3':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 20, 22, 24, 25, 29, 30, 32, 58, 60, 64, 70, 79, 80, 
        88, 100, 120, 128, 200, 220, 400],
        '__PARA__L4':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 20, 22, 24, 25, 29, 30, 32, 58, 64, 70, 80, 88, 100, 
        120, 199, 400],
        '__PARA__L5':[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 24, 25, 29, 30, 32, 50, 58, 64, 70, 100, 120, 240, 
        400],
        '__PIPE__L0':['', 'NA', 'flatten', 'off'],
        '__PIPE__L1':['', 'NA', 'flatten', 'off'],
        '__PIPE__L2':['', 'NA', 'flatten', 'off'],
        '__PIPE__L3':['', 'NA', 'flatten', 'off'],
        '__TILE__L0':[1, 2, 4, 8, 25, 30, 40, 60, 80, 90, 100, 116, 120, 200, 250, 390, 400, 410, 494],
        '__TILE__L1':[1, 2, 4, 8, 18, 20, 30, 40, 50, 58, 80, 88, 100, 118, 120, 124, 240, 400, 410],
        '__TILE__L2':[1, 2, 4, 8, 13, 18, 30, 40, 50, 58, 60, 80, 88, 100, 199, 240],
        '__TILE__L3':[1, 2, 4, 8, 18, 50, 79, 80, 120, 400],
        '__PARA__L6':[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 29, 30, 32, 58, 60, 64, 80, 100, 120, 239, 400],
        '__PARA__L7':[1, 2, 4, 5, 8, 10, 16, 20, 32, 80, 239],
        '__PARA__L8':[1, 2, 4, 5, 8, 10, 16, 25, 32, 50],
        '__PIPE__L4':['', 'NA', 'flatten', 'off'],
        '__PIPE__L5':['', 'NA', 'flatten', 'off'],
        '__TILE__L4':[1, 2, 4, 8, 18, 70, 199],
        '__TILE__L5':[1, 2, 4, 8, 70],
        '__PARA__L0_0':[1, 2, 4, 8, 16, 31, 32, 124],
        '__PARA__L0_1':[1, 2, 4, 8, 16, 31, 32, 59, 124],
        '__PARA__L7_0':[1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 100],
        '__PIPE__L7':['', 'NA', 'flatten', 'off'],
        '__PARA__L0_1_0':[1, 2, 4, 8, 16, 32],
        '__PARA__L0_2':[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 30, 32, 60],
        '__PARA__L0_2_0':[1, 2, 4, 8, 16, 32, 79],
        '__PARA__L0_3':[1, 2, 4, 8, 16, 32, 59],
        '__PARA__L0_3_0':[1, 2, 4, 8, 16, 32, 79],
        '__PIPE__L0_1':['', 'NA', 'flatten', 'off'],
        '__PIPE__L0_2':['', 'NA', 'flatten', 'off'],
        '__PIPE__L0_3':['', 'NA', 'flatten', 'off'],
        '__TILE__L0_1':[1, 2, 4, 8, 59],
        '__TILE__L0_2':[1, 2, 60],
        '__TILE__L0_3':[1, 2, 4, 8, 59],
    }
    kernels = ['2mm', 'bicg-medium', 'atax-medium', 'jacobi-2d', 'heat-3d', 'symm-opt', 'covariance', 
               'correlation', 'syrk', 'aes', 'atax', 'gemver', 'seidel-2d', 'nw', 'mvt-medium', 
               'stencil_stencil2d', 'spmv-ellpack', 'gemver-medium', 'doitgen-red', 'gemm-ncubed', 
               'gesummv-medium', 'md', 'gemm-p-large', 'adi', 'stencil', 'trmm', 'symm', 'doitgen', 
               'stencil-3d', 'fdtd-2d', 'gemm-p', 'mvt', 'gemm-blocked', 'gesummv', 'syr2k', 'trmm-opt', 
               '3mm', 'bicg-large', 'fdtd-2d-large', 'bicg', 'jacobi-1d', 'spmv-crs', 'symm-opt-medium']
    
class target:
    targets = ['valid', 'latency', 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']
    categorical_targets = ['valid']
    continuous_targets = ['latency', 'util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']
    
def collate_fn_test(ls:list):
    ret = [inst['design'] for inst in ls]
    return {'X':ret}

def collate_fn_train(ls:list):
    ret = [inst['X'] for inst in ls]
    kl = ['valid','latency','util-BRAM', 'util-LUT', 'util-FF', 'util-DSP']
    target = {key:torch.tensor([inst['y'][key] for inst in ls]) for key in kl}
    # target = [inst['y'] for inst in ls]
    return {'X':ret,'y':target}

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def inv_sigmoid(y):
    epsilon = 1e-10
    return np.log(y+epsilon) - np.log(1-y+epsilon)

