import argparse
import time
import warnings

# tools
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import magic
from tqdm import tqdm

test_file = "counts_AD.csv"
sc_imputation = True

geneExprAD = pd.read_csv(
            './' + test_file,
            index_col=0)
geneExprAD = geneExprAD.T
geneExprAD = geneExprAD * 1.0

if sc_imputation == True:
    magic_operator = magic.MAGIC(solver='approximate')  # Use the 'approximate' solver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geneExprAD = magic_operator.fit_transform(geneExprAD)


geneExprAD[geneExprAD < 0] = 0

if geneExprAD.max().max() > 50:
    geneExprAD = (geneExprAD + 1).apply(np.log2)  

geneExprAD.to_pickle('geneExprAD.pkl')

geneExprAD_transformed = geneExprAD.sum(axis=1)
stand = geneExprAD_transformed.mean()
geneExprAD_transformed = geneExprAD_transformed / stand

print(geneExprAD_transformed.shape)
geneExprAD_transformed.to_pickle('geneExprAD_transformed.pkl')

del(geneExprAD_transformed)
del(geneExprAD)

test_file = "counts_stroke.csv"

geneExprStroke = pd.read_csv(
            './' + test_file,
            index_col=0)
geneExprStroke = geneExprStroke.T
geneExprStroke = geneExprStroke * 1.0

min_value = geneExprStroke.min(numeric_only=True).min()
max_value = geneExprStroke.max(numeric_only=True).max()

print("The overall range of values in the geneExprStroke dataframe is:", min_value, "to", max_value)

if sc_imputation == True:
    magic_operator = magic.MAGIC(solver='approximate')  # Use the 'approximate' solver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geneExprStroke = magic_operator.fit_transform(geneExprStroke)


geneExprStroke[geneExprStroke < 0] = 0

if geneExprStroke.max().max() > 50:
    geneExprStroke = (geneExprStroke + 1).apply(np.log2) 


geneExprStroke.to_pickle('geneExprStroke.pkl')

geneExprStroke_transformed = geneExprStroke.sum(axis=1)
stand = geneExprStroke_transformed.mean()
geneExprStroke_transformed = geneExprStroke_transformed / stand


print(geneExprStroke_transformed.shape)

geneExprStroke_transformed.to_pickle('geneExprStroke_transformed.pkl')
