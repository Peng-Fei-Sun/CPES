# CCSNet: Connectivity-Based Curvature Spectrum

Official implementation of CCSNet: Connectivity-Based Curvature Spectrum for Protein–Ligand Binding Affinity Prediction.

---

## Setup Environment

Create the conda environment using:

```bash
conda env create -f environment.yml
conda activate CCSNet
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

Train and evaluate the CCSNet model:

```bash
python train.py
```

---

## Citation

Coming soon.

<br>

