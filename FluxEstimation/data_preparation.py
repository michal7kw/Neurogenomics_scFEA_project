import argparse
import time
import warnings

import numpy as np 
import pandas as pd
import sys
import os

module_path = os.path.abspath(os.path.join('.', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

# from ClassFlux import FLUX
from DatasetFlux import MyDataset
# from scFEA
# from scFEA_grad
from util import *

data_path ='data'
input_path ='input'
res_dir ='output'
moduleGene_file ='module_gene_complete_mouse_m168.csv'
cm_file ='cmMat_complete_mouse_c70_m168.csv'
sc_imputation =True
cName_file='cName_complete_mouse_c70_m168.csv'
fileName='output/ad1212_flux.csv'
balanceName='output/ad1212_balance.csv'
EPOCH=100

# study = 'AD'
study = 'Stroke'

geneExpr = pd.read_pickle('./input/geneExpr' + study + '.pkl')
geneExprScale = pd.read_pickle('./input/geneExpr' + study + '_transformed.pkl')

moduleGene = pd.read_csv(
            data_path + '/' + moduleGene_file,
            sep=',',
            index_col=0)

moduleGene_lower = moduleGene.applymap(lambda x: x.lower() if isinstance(x, str) else x)
moduleGene = moduleGene_lower
moduleLen = [moduleGene.iloc[i,:].notna().sum() for i in range(moduleGene.shape[0]) ]
moduleLen = np.array(moduleLen)

# find existing gene
module_gene_all = []
for i in range(moduleGene.shape[0]):
    for j in range(moduleGene.shape[1]):
        if pd.isna(moduleGene.iloc[i,j]) == False:
            module_gene_all.append(moduleGene.iloc[i,j])


data_gene_all = geneExpr.columns

# Convert sets to lowercase
data_gene_all = [gene.lower() for gene in data_gene_all]
module_gene_all = [gene.lower() for gene in module_gene_all]

print("data_gene_all: ", sorted(data_gene_all))
print("module_gene_all: ", sorted(module_gene_all))
print("data_gene_all len: ", len(data_gene_all))
print("module_gene_all len: ", len(module_gene_all))

data_gene_all = set(data_gene_all)
module_gene_all = set(module_gene_all)
print("data_gene_all len: ", len(data_gene_all))
print("module_gene_all len: ", len(module_gene_all))


# Find intersection in lowercase
gene_overlap = list(data_gene_all.intersection(module_gene_all))
gene_overlap.sort()

# Optional: Map back to original case (choosing data_gene_all as source)
# gene_overlap = [gene for gene in data_gene_all if gene.lower() in gene_overlap]

print("Gene overlap:", gene_overlap)
print("Gene overlap len:", len(gene_overlap))



cmMat = pd.read_csv(
        data_path + '/' + cm_file,
        sep=',',
        header=None)
cmMat = cmMat.values
print(cmMat[:5,:5])


if cName_file != 'noCompoundName':
    print("Load compound name file, the balance output will have compound name.")
    cName = pd.read_csv(
            data_path + '/' + cName_file,
            sep=',',
            header=0)
    cName = cName.columns
print("Load data done.")
print(cName[:5])


geneExpr.columns = geneExpr.columns.str.lower()
geneExpr

print("Starting process data...")
emptyNode = []
# extract overlap gene
geneExpr = geneExpr[gene_overlap] 
print(f'geneExpr: {geneExpr.head()}')

gene_names = geneExpr.columns
print(f'gene_names: {gene_names[:5]}')

cell_names = geneExpr.index.astype(str)
print(f'cell_names: {cell_names[:5]}')

n_modules = moduleGene.shape[0]
n_genes = len(gene_names)
n_cells = len(cell_names)
n_comps = cmMat.shape[0]
print(f'n_modules: {n_modules}, n_genes: {n_genes}, n_cells: {n_cells}, n_comps: {n_comps}')

geneExprDf = pd.DataFrame(columns = ['Module_Gene'] + list(cell_names))
print(geneExprDf)

for i in range(n_modules):
    genes = moduleGene.iloc[i,:].values.astype(str)
    genes = [g for g in genes if g != 'nan']
    if not genes:
        emptyNode.append(i)
        continue
    temp = geneExpr.copy()
    temp.loc[:, [g for g in gene_names if g not in genes]] = 0
    temp = temp.T
    temp['Module_Gene'] = ['%02d_%s' % (i,g) for g in gene_names]
    # geneExprDf = geneExprDf.append(temp, ignore_index = True, sort=False)
    geneExprDf = pd.concat([geneExprDf, temp], ignore_index=True, sort=False)
geneExprDf.index = geneExprDf['Module_Gene']
geneExprDf.drop('Module_Gene', axis = 'columns', inplace = True)


geneExprDf.to_pickle('geneExpr' + study + '_df.pkl')
