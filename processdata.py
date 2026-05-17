
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import glob

import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

from build_anm_graphs import build_anm_ligand_graph, build_anm_protein_graph, build_anm_complex_graph



def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]

        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

    return edge_index_inter


def mols2graphs(complex_path, label, dis_threshold=5.):

    data = None

    if os.path.isfile(complex_path):

        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        x_l, edge_index_l = mol2graph(ligand)
        x_p, edge_index_p = mol2graph(pocket)
        x = torch.cat([x_l, x_p], dim=0)
        edge_index_intra = torch.cat([edge_index_l, edge_index_p+atom_num_l], dim=-1)
        edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        y = torch.FloatTensor([label])
        pos = torch.concat([pos_l, pos_p], dim=0)
        split = torch.cat([torch.zeros((atom_num_l, )), torch.ones((atom_num_p,))], dim=0)

        data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)


    else:
        print('mols2graphs failed', complex_path, 'does not exist')

    return data




def processdata_single(pKa, cid, complex_dir, complex_dir_new, graph_dir_cid):

    graph_type = 'Graph_CPES'
    dis_threshold = 5

    complex_path = os.path.join(complex_dir_new, f"{cid}_{dis_threshold}A.rdkit")
    data_complex = mols2graphs(complex_path=complex_path, label=pKa, dis_threshold=dis_threshold)

    data_anm_ligand = build_anm_ligand_graph(cid, complex_dir, complex_dir_new, K_modes=100, K_u=200)
    data_anm_protein = build_anm_protein_graph(cid, complex_dir, complex_dir_new, K_modes=100, K_u=200)
    data_anm_complex = build_anm_complex_graph(cid, complex_dir, complex_dir_new, inter_cutoff=3.0, K_modes=100, K_u=200)

    os.makedirs(graph_dir_cid)
    graph_path = os.path.join(graph_dir_cid, f"{graph_type}-{cid}.pt")
    if data_complex is not None and data_anm_ligand is not None and data_anm_protein is not None and data_anm_complex is not None:
        data_dict = {'data_complex': data_complex, 'data_anm_ligand': data_anm_ligand, 'data_anm_protein': data_anm_protein, 'data_anm_complex': data_anm_complex}
        torch.save(data_dict, graph_path)
        print('Graph has been saved to', os.path.basename(graph_path))
    else:
        print('Graph failed to save', graph_path)


def processdata(data_dir, data_df, save_dir, graph_dir):
    num_process = 10


    pKa_list = []
    cid_list = []
    complex_dir_list = []
    complex_dir_new_list = []
    graph_dir_list = []
    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        complex_dir_new = os.path.join(save_dir, cid)
        graph_dir_cid = os.path.join(graph_dir, cid)

        pKa_list.append(pKa)
        cid_list.append(cid)
        complex_dir_list.append(complex_dir)
        complex_dir_new_list.append(complex_dir_new)
        graph_dir_list.append(graph_dir_cid)

    print('Generate complex graph by multi-thread processing')

    pool = multiprocessing.Pool(num_process)
    pool.starmap(processdata_single,
                    zip(pKa_list, cid_list, complex_dir_list, complex_dir_new_list, graph_dir_list))
    pool.close()
    pool.join()



if __name__ == '__main__':

    data_root = "./data"
    save_root = "./preprocess"
    graph_root = "./graph"
    dataset_list = ['testcsar', 'test2013', 'test2016', 'test2019', 'train', 'valid']
    csvfile_list = ['csar.csv', 'test2013.csv', 'test2016.csv', 'test2019.csv', 'train.csv', 'valid.csv']

    for i in range(len(dataset_list)):
        print('   Dataset Processing: ', dataset_list[i])
        data_dir = os.path.join(data_root, dataset_list[i])
        save_dir = os.path.join(save_root, dataset_list[i])
        graph_dir = os.path.join(graph_root, dataset_list[i])
        data_df = pd.read_csv(os.path.join(data_root, csvfile_list[i]))

        processdata(data_dir, data_df, save_dir, graph_dir)


