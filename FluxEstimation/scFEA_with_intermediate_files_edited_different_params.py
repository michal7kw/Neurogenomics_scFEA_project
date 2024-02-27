##################### SETUP ####################################################
import argparse
import time
import warnings

# tools
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import magic
from tqdm import tqdm
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
test_file ='Neuro1.csv'
moduleGene_file ='module_gene_complete_mouse_m168.csv'
cm_file ='cmMat_complete_mouse_c70_m168.csv'
sc_imputation =True
cName_file='cName_complete_mouse_c70_m168.csv'
fileName='output/AD_flux_lambda_02_02_100_100.csv'
balanceName='output/AD_balance_lambda_02_02_100_100.csv'
EPOCH=100

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

print(f"CUDA available: {cuda_available}")

# If CUDA is available, print the number of CUDA devices and their names
if cuda_available:
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")


# choose cpu or gpu automatically
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("now you are using device: ", device)

##################### LOAD EXPRESSION DATA ####################################################
geneExpr = pd.read_pickle('./input/geneExprAD.pkl')
geneExprScale = pd.read_pickle('./input/geneExprAD_transformed.pkl')
geneExprScale = torch.FloatTensor(geneExprScale.values).to(device)

##################### LOAD MODULES DATA ####################################################
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

##################### LOAD STOICHIOMETRY DATA ####################################################
cmMat = pd.read_csv(
        data_path + '/' + cm_file,
        sep=',',
        header=None)
cmMat = cmMat.values
print(cmMat[:5,:5])
cmMat = torch.FloatTensor(cmMat).to(device)


if cName_file != 'noCompoundName':
    print("Load compound name file, the balance output will have compound name.")
    cName = pd.read_csv(
            data_path + '/' + cName_file,
            sep=',',
            header=0)
    cName = cName.columns
print("Load data done.")
print(cName[:5])


##################### PROCESS DATA ####################################################
geneExpr.columns = geneExpr.columns.str.lower()

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

##################### FORMAT DATA ####################################################
# for i in range(n_modules):
#     genes = moduleGene.iloc[i,:].values.astype(str)
#     genes = [g for g in genes if g != 'nan']
#     if not genes:
#         emptyNode.append(i)
#         continue
#     temp = geneExpr.copy()
#     temp.loc[:, [g for g in gene_names if g not in genes]] = 0
#     temp = temp.T
#     temp['Module_Gene'] = ['%02d_%s' % (i,g) for g in gene_names]
#     # geneExprDf = geneExprDf.append(temp, ignore_index = True, sort=False)
#     geneExprDf = pd.concat([geneExprDf, temp], ignore_index=True, sort=False)
# geneExprDf.index = geneExprDf['Module_Gene']
# geneExprDf.drop('Module_Gene', axis = 'columns', inplace = True)

geneExprDf = pd.read_pickle('./input/geneExprAD_df.pkl')
print(geneExprDf.shape)
print(geneExprDf.max())

X = geneExprDf.values.T
X = torch.FloatTensor(X).to(device)

print(X.shape)

geneExprDf = geneExprDf
print(geneExprDf.index[400:450])

geneExprDf.index = [i.split('_')[0] for i in geneExprDf.index]
geneExprDf.index = geneExprDf.index.astype(int) 

module_scale = geneExprDf.groupby(geneExprDf.index).sum().T  
module_scale.shape

module_scale = torch.FloatTensor(module_scale.values/ moduleLen) 
print("Process data done.")

##################### NN MODEL ####################################################
LEARN_RATE = 0.001

class FLUX(nn.Module):
    def __init__(self, matrix, n_modules, f_in = 50, f_out = 1):
        super(FLUX, self).__init__()
        # gene to flux
        self.inSize = f_in     
        
        self.m_encoder = nn.ModuleList([
                                        nn.Sequential(nn.Linear(self.inSize,8, bias = False),
                                                      nn.Tanhshrink(),
                                                      nn.Linear(8, f_out),
                                                      nn.Tanhshrink()
                                                      )
                                        for i in range(n_modules)])

    
    def updateC(self, m, n_comps, cmMat): # stoichiometric matrix
        
        c = torch.zeros((m.shape[0], n_comps))
        for i in range(c.shape[1]):
            tmp = m * cmMat[i,:]
            c[:,i] = torch.sum(tmp, dim=1)
        
        return c
        

    def forward(self, x, n_modules, n_genes, n_comps, cmMat):
        print(x.shape)
        m = torch.Tensor().to(device) # ADDED

        for i in range(n_modules):
            x_block = x[:, i*n_genes: (i+1)*n_genes]
            subnet = self.m_encoder[i]
            m_block = subnet(x_block)
            m = torch.cat((m, m_block), 1) if m.size(0) else m_block
        c = self.updateC(m, n_comps, cmMat)
        return m, c
    
def myLoss(m, c, lamb1 = 0.2, lamb2= 0.2, lamb3 = 10, lamb4 = 10, geneScale = None, moduleScale = None):    
    
    # balance constrain
    total1 = torch.pow(c, 2)
    total1 = torch.sum(total1, dim = 1) 
    
    # non-negative constrain
    error = torch.abs(m) - m
    total2 = torch.sum(error, dim=1)
    
    
    # sample-wise variation constrain 
    diff = torch.pow(torch.sum(m, dim=1) - geneScale, 2)
    #total3 = torch.pow(diff, 0.5)
    if sum(diff > 0) == m.shape[0]: # solve Nan after several iteraions
        total3 = torch.pow(diff, 0.5)
    else:
        total3 = diff
    
    # module-wise variation constrain
    if lamb4 > 0 :
        corr = torch.FloatTensor(np.ones(m.shape[0]))
        for i in range(m.shape[0]):
            corr[i] = pearsonr(m[i, :], moduleScale[i, :])
        corr = torch.abs(corr)
        penal_m_var = torch.FloatTensor(np.ones(m.shape[0])) - corr
        total4 = penal_m_var
    else:
        total4 = torch.FloatTensor(np.zeros(m.shape[0]))
            
    # loss
    loss1 = torch.sum(lamb1 * total1)
    loss2 = torch.sum(lamb2 * total2)
    loss3 = torch.sum(lamb3 * total3)
    loss4 = torch.sum(lamb4 * total4)
    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss1, loss2, loss3, loss4

# =============================================================================
#NN
torch.manual_seed(16)
print(f'X: {X.size()}, n_modules: {n_modules}, n_genes: {n_genes}')
net = FLUX(X, n_modules, f_in = n_genes, f_out = 1).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = LEARN_RATE)
# =============================================================================

BATCH_SIZE = max(int(geneExpr.shape[0]/8), 1)  # Ensure batch size is at least 1
#Dataloader
dataloader_params = {'batch_size': BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': 0,
                        'pin_memory': False}

# dataSet = MyDataset(X, geneExprScale, module_scale)
# train_loader = torch.utils.data.DataLoader(dataset=dataSet,
#                                             **dataloader_params)

dataSet = MyDataset(X.cpu(), geneExprScale.cpu(), module_scale.cpu())  # No need to move dataset to GPU upfront
train_loader = torch.utils.data.DataLoader(dataset=dataSet, **dataloader_params)

#for i, (X_batch, X_scale_batch, m_scale_batch) in enumerate(train_loader):
#    print(f"Batch {i}: X_batch shape: {X_batch.shape}")

##################### TRAIN ####################################################
print("Starting train neural network...")
start = time.time()  
#   training
loss_v = []
loss_v1 = []
loss_v2 = []
loss_v3 = []
loss_v4 = []
net.train()
timestr = time.strftime("%Y%m%d-%H%M%S")
lossName = "./output/lossValue_" + timestr + ".txt"
file_loss = open(lossName, "a")

LAMB_BA = 1
LAMB_NG = 1 
LAMB_CELL =  1
LAMB_MOD = 1e-2 
    
# Training loop setup
start = time.time()  # To measure the total training time
loss_v, loss_v1, loss_v2, loss_v3, loss_v4 = [], [], [], [], []

# Ensure file_loss is opened before the loop and closed properly after
with open("./output/lossValue_" + timestr + ".txt", "a") as file_loss:
    net.train()
    for epoch in tqdm(range(EPOCH)):
        epoch_loss, epoch_loss1, epoch_loss2, epoch_loss3, epoch_loss4 = 0, 0, 0, 0, 0

        for i, (D, D_scale, m_scale) in enumerate(train_loader):
            D_batch = D.to(device).float()
            X_scale_batch = D_scale.to(device).float()
            m_scale_batch = m_scale.to(device).float()

            # Forward pass
            out_m_batch, out_c_batch = net(D_batch, n_modules, n_genes, n_comps, cmMat)
            loss_batch, loss1_batch, loss2_batch, loss3_batch, loss4_batch = myLoss(
                out_m_batch, out_c_batch,
                lamb1=LAMB_BA, lamb2=LAMB_NG, lamb3=LAMB_CELL, lamb4=LAMB_MOD,
                geneScale=X_scale_batch, moduleScale=m_scale_batch
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            # Accumulate losses for logging
            epoch_loss += loss_batch.item()
            epoch_loss1 += loss1_batch.item()
            epoch_loss2 += loss2_batch.item()
            epoch_loss3 += loss3_batch.item()
            epoch_loss4 += loss4_batch.item()

        # Logging after each epoch
        file_loss.write(f'epoch: {epoch+1}, loss: {epoch_loss:.8f}, balance: {epoch_loss1:.8f}, '
                        f'negative: {epoch_loss2:.8f}, cellVar: {epoch_loss3:.8f}, moduleVar: {epoch_loss4:.8f}\n')

        # Append epoch losses for plotting
        loss_v.append(epoch_loss)
        loss_v1.append(epoch_loss1)
        loss_v2.append(epoch_loss2)
        loss_v3.append(epoch_loss3)
        loss_v4.append(epoch_loss4)

# Measure and print training time
end = time.time()
print("Training time: ", end - start)

# Plotting the training loss curves
plt.plot(loss_v, '--', label='Total Loss')
plt.plot(loss_v1, label='Balance')
plt.plot(loss_v2, label='Negative')
plt.plot(loss_v3, label='Cell Variation')
plt.plot(loss_v4, label='Module Variation')
plt.legend()
imgName = './' + res_dir + '/loss_' + timestr + ".png"
plt.savefig(imgName)

##################### TESTING AND SAVING RESULTS ##############################

#    Dataloader
dataloader_params = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 0,
                        'pin_memory': False}

# print(f"X.shape: {X.shape}")
# print(f"geneExprScale: {geneExprScale.shape}")
# print(f"module_scale: {module_scale.shape}")

dataSet2 = MyDataset(X, geneExprScale, module_scale)
test_loader2 = torch.utils.data.DataLoader(dataset=dataSet2,
                        **dataloader_params)

#testing
fluxStatuTest = np.zeros((n_cells, n_modules), dtype='f') #float32
balanceStatus = np.zeros((n_cells, n_comps), dtype='f')
net.eval()
for epoch in range(1):
    loss, loss1, loss2 = 0,0,0
    
    for i, (D, D_scale, _) in enumerate(test_loader2):

        D_batch = Variable(D.float().to(device))
        # print(f"D_batch: {D_batch.shape}")

        out_m_batch, out_c_batch = net(D_batch, n_modules, n_genes, n_comps, cmMat)
        # print(f"out_m_batch: {out_m_batch.shape}")
        # print(f"out_c_batch: {out_c_batch.shape}")

        out_m = out_m_batch.detach().cpu().numpy()
        out_c = out_c_batch.detach().cpu().numpy()
        # print(f"out_m: {out_m.shape}")
        # print(f"out_c: {out_c.shape}")

        # save data
        fluxStatuTest[i, :] = out_m
        balanceStatus[i, :] = out_c
        
                

# save to file
if fileName == 'NULL':
    # user do not define file name of flux
    fileName = "./" + res_dir + "/" + test_file[-len(test_file):-4] + "_module" + str(n_modules) + "_cell" + str(n_cells) + "_batch" + str(BATCH_SIZE) + \
                "_LR" + str(LEARN_RATE) + "_epoch" + str(EPOCH) + "_SCimpute_" + str(sc_imputation)[0] + \
                "_lambBal" + str(LAMB_BA) + "_lambSca" + str(LAMB_NG) + "_lambCellCor" + str(LAMB_CELL) + "_lambModCor_1e-2" + \
                '_' + timestr + ".csv"
setF = pd.DataFrame(fluxStatuTest)
setF.columns = moduleGene.index
setF.index = geneExpr.index.tolist()
setF.to_csv(fileName)

setB = pd.DataFrame(balanceStatus)
setB.rename(columns = lambda x: x + 1)
setB.index = setF.index
if cName_file != 'noCompoundName':
    setB.columns = cName
if balanceName == 'NULL':
    # user do not define file name of balance
    balanceName = "./output/balance_" + timestr + ".csv"
setB.to_csv(balanceName)


print("scFEA job finished. Check result in the desired output folder.")


