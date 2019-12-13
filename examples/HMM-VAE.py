# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../tmp'))
	print(os.getcwd())
except:
	pass

# %%
# Add the path of the beer source code ot the PYTHONPATH.
from collections import defaultdict
import random
import sys
sys.path.append('/home/danliwoo/gplab/beer')
from beer import __init__

import copy

import beer
import numpy as np
import torch

# For plotting.
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, gridplot
from bokeh.models import LinearAxis, Range1d
output_notebook()

# Convenience functions for plotting.
import plotting

# %% [markdown]
# ### Set GPU

# %%
torch.cuda.set_device("cuda:3")
torch.cuda.current_device()
# device = torch.device("cuda:5")
# xxx.to(device)

# %% [markdown]
# ### Synthetic Data

# %%
import synthetic_data
data, states = synthetic_data.generate_sequential_data()

# %% [markdown]
# ### Construct Graph
# This graph describe the transformation of hidden state.

# %%
graph = beer.graph.Graph()

# Initial and final state are non-emitting.
s0 = graph.add_state()
s4 = graph.add_state()
graph.start_state = s0
graph.end_state = s4

s1 = graph.add_state(pdf_id=0)
s2 = graph.add_state(pdf_id=1)
s3 = graph.add_state(pdf_id=2)
graph.add_arc(s0, s1) # default weight=1
graph.add_arc(s1, s1)
graph.add_arc(s1, s2)
graph.add_arc(s2, s2)
graph.add_arc(s2, s3)
graph.add_arc(s3, s3)
graph.add_arc(s3, s1)
graph.add_arc(s1, s4)
graph.add_arc(s2, s4)
graph.add_arc(s3, s4)

graph.normalize()
graph


# %%
cgraph = graph.compile()
cgraph.final_log_probs

# %% [markdown]
# ### Pretrain HMM 

# %%
# We use the global mean/cov. matrix of the data to initialize the mixture.
data_mean = torch.from_numpy(data.mean(axis=0)).float()
data_var = torch.from_numpy(np.cov(data.T)).float()
trans_mat = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
transitions = torch.from_numpy(trans_mat).float()

# HMM (full cov).
modelset = beer.NormalSet.create(data_mean, data_var, size=len(transitions),
                                prior_strength=1., noise_std=0, 
                                cov_type='full')
hmm_full = beer.HMM.create(cgraph, modelset).double().cuda()

print(hmm_full)


# %%
epochs = 2
lrate = 1.
X = torch.from_numpy(data).cuda()

optim = beer.VBConjugateOptimizer(hmm_full.mean_field_factorization(), lrate)

elbos = []

for epoch in range(epochs):
    optim.init_step()
    elbo = beer.evidence_lower_bound(hmm_full, X, datasize=len(X), viterbi=False)
    elbo.backward()
    elbos.append(float(elbo) / len(X)) 
    optim.step()


# %%
fig = figure()
fig.line(range(len(elbos)), elbos)
show(fig)


# %%
fig = figure(title='hmm_full', width=250, height=250)
fig.circle(data[:, 0], data[:, 1], alpha=.1)
plotting.plot_hmm(fig, hmm_full.cpu(), alpha=.3, colors=['blue', 'red', 'green'])
show(fig)

# %% [markdown]
# ### Train VAE

# %%
encoder = beer.nnet.ResidualFeedForwardNet(dim_in=2, nblocks=2, block_width=2)
decoder = beer.nnet.ResidualFeedForwardNet(dim_in=2, nblocks=2, block_width=2)
vae = beer.VAE(hmm_full, encoder, decoder).double().cuda()


# %%
# small number of epochs for testing
epochs = 10
update_prior_after_epoch = 3
prior_lrate = 1.
cjg_optim = beer.VBConjugateOptimizer(vae.mean_field_factorization(), lrate=0)
std_optim = torch.optim.Adam(vae.parameters(), lr=1e-3)
optim = beer.VBOptimizer(cjg_optim, std_optim)


# %%
elbos = []
for e in range(epochs):
    optim.init_step()
    elbo = beer.evidence_lower_bound(vae, X, nsamples=5)
    elbo.backward()
    optim.step()

    if e >= update_prior_after_epoch:
        cjg_optim.lrate = prior_lrate
    elbos.append(float(elbo) / len(X))


# %%
fig = figure()
fig.line(range(len(elbos[300:])), elbos[300:])
show(fig)
elbos[-1]


# %%
fig = figure(title='hmm_full', width=250, height=250)
fig.circle(data[:, 0], data[:, 1], alpha=.1)
plotting.plot_hmm(fig, hmm_full.cpu(), alpha=.3, colors=['blue', 'red', 'green'])
show(fig)


# %%
vae.encoder.dim_out

# %% [markdown]
# ### Save Model

# %%
torch.save(vae.state_dict(), 'hmm-vae-cuda.pkl')

# %% [markdown]
# ### Load Model

# %%
vae.load_state_dict(torch.load('hmm-vae-cuda.pkl'))