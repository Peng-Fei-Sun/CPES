
import glob
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class GraphAnmDataset(Dataset):
    def __init__(self, root: str, pattern: str = "Graph_CPES-*.pt"):

        subdirs = [os.path.join(root, name) for name in os.listdir(root)]
        expected = len(subdirs)
        files = []
        actual = 0
        for d in subdirs:
            hits = glob.glob(os.path.join(d, pattern))
            if hits:
                actual += 1
                files.extend(hits)
        self.files = sorted(files)
        missing = expected - actual


        self._data = [torch.load(path, map_location="cpu") for path in self.files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = self._data[idx]
        return d


def collate_fn(samples):
    batch_complex = Batch.from_data_list([s["data_complex"] for s in samples])
    batch_anm_ligand = Batch.from_data_list([s["data_anm_ligand"] for s in samples])
    batch_anm_protein = Batch.from_data_list([s["data_anm_protein"] for s in samples])
    batch_anm_complex = Batch.from_data_list([s["data_anm_complex"] for s in samples])
    return batch_complex, batch_anm_ligand, batch_anm_protein, batch_anm_complex
