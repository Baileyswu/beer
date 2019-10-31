import copy
import sys
sys.path.append('/home/danliwoo/gplab/beer')
from beer import __init__
import beer
import numpy as np
import torch

# For plotting.
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, gridplot
from bokeh.models import LinearAxis, Range1d

# Convenience functions for plotting.
import synthetic_data

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
graph.add_arc(s0, s1)
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


# %%
# We use the global mean/cov. matrix of the data to initialize the mixture.
data, trans_mat = synthetic_data.generate_sequential_data()
data_mean = torch.from_numpy(data.mean(axis=0)).float()
data_var = torch.from_numpy(np.cov(data.T)).float()

init_states = torch.LongTensor([0])
final_states = torch.LongTensor([2])
transitions = torch.from_numpy(trans_mat).float()

# HMM (full cov).
modelset = beer.NormalSet.create(data_mean, data_var, size=len(transitions),
    prior_strength=1., noise_std=0, 
    cov_type='full')
hmm_full = beer.HMM.create(cgraph, modelset)


models = {
    'hmm_full': hmm_full.double()
}

print(hmm_full) 


# %%
epochs = 1
lrate = 1.
X = torch.from_numpy(data).double()

optims = {
    model_name: beer.VBConjugateOptimizer(model.mean_field_factorization(), lrate)
    for model_name, model in models.items()
}

elbos = {
    model_name: [] 
    for model_name in models
}  

for epoch in range(epochs):
    for name, model in models.items():
        optim = optims[name]
        optim.init_step()
        elbo = beer.evidence_lower_bound(model, X, datasize=len(X), viterbi=False)
        elbo.backward()
        elbos[name].append(float(elbo) / len(X)) 
        optim.step()


# %%
fig = figure()
fig.line(range(len(elbos['hmm_full'])), elbos['hmm_full'])
show(fig)


# %%
encoder = beer.nnet.ResidualFeedForwardNet(dim_in=2, nblocks=2, block_width=2)
decoder = beer.nnet.ResidualFeedForwardNet(dim_in=2, nblocks=2, block_width=2)
vae = beer.VAE(hmm_full, encoder, decoder).double()


# %%
epochs = 1
update_prior_after_epoch = 10
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
