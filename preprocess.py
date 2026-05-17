
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def generate_pocket(data_dir, save_dir, distance=5):
    complex_id = os.listdir(data_dir)
    for cid in complex_id:
        print(cid)
        complex_dir = os.path.join(data_dir, cid)
        complex_dir_new = os.path.join(save_dir, cid)
        os.makedirs(complex_dir_new, exist_ok=True)
        lig_native_path = os.path.join(complex_dir, f"{cid}_ligand.mol2")
        protein_path= os.path.join(complex_dir, f"{cid}_protein.pdb")


        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid}_ligand around {distance}')
        pymol.cmd.save(os.path.join(complex_dir_new, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')

def generate_complex(data_dir, data_df, save_dir, distance=5, input_ligand_format='mol2'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        cid, pKa = row['pdbid'], float(row['-logKd/Ki'])
        complex_dir = os.path.join(data_dir, cid)
        complex_dir_new = os.path.join(save_dir, cid)
        pocket_path = os.path.join(save_dir, cid, f'Pocket_{distance}A.pdb')

        ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.{input_ligand_format}')
        ligand_path_new = os.path.join(save_dir, cid, f'{cid}_ligand.{input_ligand_format}')
        ligand_path = ligand_path_new.replace(f".{input_ligand_format}", ".pdb")
        os.system(f'obabel {ligand_input_path} -O {ligand_path} -d')


        save_path = os.path.join(complex_dir_new, f"{cid}_{distance}A.rdkit")
        ligand = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path, removeHs=True)
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

if __name__ == '__main__':
    distance = 5
    input_ligand_format = 'mol2'

    data_root = "./data"
    save_root = "./preprocess"
    dataset_list = ['testcsar', 'test2013', 'test2016', 'test2019', 'train', 'valid']
    csvfile_list = ['csar.csv', 'test2013.csv', 'test2016.csv', 'test2019.csv', 'train.csv', 'valid.csv']

    for i in range(len(dataset_list)):
        print('   Dataset Processing: ', dataset_list[i])
        data_dir = os.path.join(data_root, dataset_list[i])
        save_dir = os.path.join(save_root, dataset_list[i])
        data_df = pd.read_csv(os.path.join(data_root, csvfile_list[i]))


        generate_pocket(data_dir=data_dir, save_dir=save_dir, distance=distance)



        generate_pocket(data_dir=data_dir, save_dir=save_dir, distance=10)

        generate_complex(data_dir=data_dir, data_df=data_df, save_dir=save_dir, distance=distance, input_ligand_format=input_ligand_format)




