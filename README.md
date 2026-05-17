# CPES: Curvature-Informed Potential Energy Surface

Official implementation of CPES: Curvature-Informed Potential Energy Surface for protein-ligand binding affinity prediction.

---

## Setup Environment

Create the conda environment using:

```bash
conda env create -f environment.yml
conda activate CPES
```

---

## Step1: Preprocess Raw Data

The PDBbind dataset can be downloaded from: [PDBbind+](https://www.pdbbind-plus.org.cn/). 
Place the raw pdb files in: `./data`

Preprocess the raw protein-ligand complex data:

```bash
python preprocess.py
```

---

## Step2: Construct Graph Data

Construct graph-based representations for model training:

```bash
python processdata.py
```

---

## Step3: Train and Evaluate the Model

Train and evaluate the CPES model:

```bash
python train.py
```

---

## Citation

Coming soon.

<br>

