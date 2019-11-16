# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import sys
sys.path.append('/home/danliwoo/gplab/beer')
from beer import __init__
# %%
from IPython import get_ipython

# %% [markdown]
# # VAE-GMM
# 
# This notebook illustrate how to combine a Variational AutoEncoder (VAE) and a Gaussian Mixture Model (GMM) with the [beer framework](https://github.com/beer-asr/beer).

# Add the path of the beer source code ot the PYTHONPATH.
from collections import defaultdict
import random
import math
import yaml
import numpy as np
import torch
import torch.optim
from torch import nn



# For plotting.
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, gridplot
from bokeh.models import LinearAxis, Range1d

# Beer framework
import beer

# Convenience functions for plotting.
import plotting

output_notebook(verbose=False)

# %% [markdown]
# ## Data 
# 
# As a simple example we consider the following synthetic data: 

# %%
def generate_cluster(npoints, mean, angle):
    x = np.random.randn(npoints) 
    data = np.c_[x, np.cos(x) + np.random.randn(npoints) * 1e-1] 
    R = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    return ((data - np.array([0, 1])) @ R)  + mean

data1 = generate_cluster(npoints=50, mean=np.array([0., -1.]), angle=0)
data2 = generate_cluster(npoints=50, mean=np.array([1., 0.]), angle=np.pi/2)
data3 = generate_cluster(npoints=50, mean=np.array([0., 1.]), angle=np.pi)
data4 = generate_cluster(npoints=50, mean=np.array([-1., 0.]), angle=3 * np.pi/2)


X = torch.from_numpy(np.r_[data1, data2, data3, data4])

fig = figure()
fig.circle(data1[:, 0], data1[:, 1], color='blue')
fig.circle(data2[:, 0], data2[:, 1], color='red')
fig.circle(data3[:, 0], data3[:, 1], color='green')
fig.circle(data4[:, 0], data4[:, 1], color='orange')


show(fig)

# %% [markdown]
# ## Model Creation
# 

# %%
data = np.vstack((data1, data2, data3, data4))


# %%
data_mean = torch.from_numpy(data.mean(axis=0)).double()
data_var = torch.from_numpy(np.var(data, axis=0)).double()

gaussians = beer.NormalSet.create(
    data_mean, data_var,      # use to set the mean/variance of the prior
    size=50,                  # total number of components in the mixture
    prior_strength=1e-3,        # how much the prior affect the training ("pseudo-counts")
    noise_std=1.,             # standard deviation of the noise to initialize the mean of the posterior
    cov_type='full',          # type of the covariance matrix  ('full', 'diagonal' or 'isotropic')
)

gmm = beer.Mixture.create(
    gaussians, 
    prior_strength=1.         # how much the prior over the weights will affect the training ("pseudo-counts")
)

gmm = gmm.double()            # set all the parameters in double precision
#gmm = gmm.cuda()             # move the model on a GPU. If you do so, you'll have
                              # to move the data as well.
    
    
# Fit the GMM to the data as initialization.
# Note that this initialization is valid as 
# we assume a residual network as encoder (see cells below)
optim = beer.VBConjugateOptimizer(gmm.mean_field_factorization(), lrate=1.)

last_ev = -np.infty
epochs = 0
while True:
    epochs += 1
    optim.init_step()
    elbo = beer.evidence_lower_bound(gmm, X)
    elbo.backward()
    optim.step()
    if np.absolute(last_ev - elbo.value) < 100:
        break
    last_ev = elbo.value
print("epochs:", epochs, "elbo:", elbo.value)
    
print(gmm)


# %%
encoder = beer.nnet.ResidualFeedForwardNet(dim_in=2, nblocks=2, block_width=20)
decoder = beer.nnet.ResidualFeedForwardNet(dim_in=2, nblocks=2, block_width=20)
vae = beer.VAE(gmm, encoder, decoder).double()


# %%
epochs = 1
update_prior_after_epoch = 100
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
    
fig = figure()
fig.line(range(len(elbos)), elbos)
show(fig)


# %%
posts = vae.posteriors(X)
rX = vae.dec_mean_layer(vae.decoder(posts.params.mean)).detach().numpy()
weights = gmm.categorical.mean.numpy()
print(weights)

fig = figure(title='Components', width=400, height=400)
for i in range(len(posts)):
    normal = posts[i]
    mean = normal.params.mean.detach().numpy()
    cov = normal.params.diag_cov.detach().diag().numpy()
    plotting.plot_normal(fig, mean, cov, alpha=.5, n_std_dev=2, color='salmon')
# fig.circle(data[:, 0], data[:, 1], alpha=.5)
for weight, normal in zip(weights, gmm.modelset):
    mean = normal.mean.numpy()
    cov = normal.cov.numpy()
    plotting.plot_normal(fig, mean, cov, alpha=.5 * weight, color='green')
    
fig2 = figure(width=400, height=400, x_range=fig.x_range)
fig2.circle(data[:, 0], data[:, 1])
fig2.circle(rX[:, 0], rX[:, 1], color='red')
#fig2 = figure(width=400, height=400, y_range=(-0.1, 1.1), title='Mixing weights')
#fig2.vbar(range(len(weights)), width=.5, top=weights)
#fig2.xaxis.ticker = list(range(len(weights)))
#fig2.xgrid.visible = False
show(gridplot([[fig, fig2]]))


# %%
posts[0].params.diag_cov.diag()


# %%
epochs = 2000
cjg_optim = beer.VBConjugateOptimizer(vae.mean_field_factorization(), lrate=1)
std_optim = torch.optim.Adam(vae.parameters(), lr=1e-3)
optim = beer.VBOptimizer(cjg_optim, std_optim)

elbos = []
for e in range(epochs):
    optim.init_step()
    elbo = beer.evidence_lower_bound(vae, X, nsamples=5)
    elbo.backward()
    optim.step()
    elbos.append(float(elbo) / len(X))
    
fig = figure()
fig.line(range(len(elbos)), elbos)
show(fig)


# %%
weights = gmm.categorical.mean.numpy()
fig = figure(title='Components', width=400, height=400,
             x_range=(-10, 10), y_range=(-5, 15))
fig.circle(data[:, 0], data[:, 1], alpha=.5)
for weight, normal in zip(weights, gmm.modelset):
    mean = normal.mean.numpy()
    cov = normal.cov.numpy()
    plotting.plot_normal(fig, mean, cov, alpha=.5 * weight, color='green')
    
fig2 = figure(width=400, height=400, y_range=(-0.1, 1.1), title='Mixing weights')
fig2.vbar(range(len(weights)), width=.5, top=weights)
fig2.xaxis.ticker = list(range(len(weights)))
fig2.xgrid.visible = False
show(gridplot([[fig, fig2]]))

# %% [markdown]
# ### 1. Pre-training

# %%
def train_cvb(model, X, epochs=1, nbatches=1, lrate_nnet=1e-3, 
              update_prior=True, update_nnet=True, kl_weight=1., state=None,
              nsamples=1, callback=None):
    
    batches = X.view(nbatches, -1, 2)

    prior_parameters = model.bayesian_parameters() if update_prior else model.normal.bayesian_parameters()
    
    if state is None:
        if update_nnet:
            std_optimizer = torch.optim.Adam(model.modules_parameters(), lr=lrate_nnet, 
                                             weight_decay=1e-2)
        else:
            std_optimizer = None
        optimizer = beer.CVBOptimizer(prior_parameters, std_optim=std_optimizer)
        batch_stats = defaultdict(lambda: defaultdict(lambda: None))
    else:
        optimizer, batch_stats = state

    for epoch in range(epochs):
        # Randomized the order of the batches.
        batch_ids = list(range(len(batches)))
        random.shuffle(batch_ids)
        
        for batch_id in batch_ids:
            optimizer.init_step(batch_stats[batch_id])
            kwargs = {'kl_weight': kl_weight}
            elbo = beer.collapsed_evidence_lower_bound(model, batches[batch_id], 
                                                       nsamples=nsamples,
                                                       **kwargs)
            batch_stats[batch_id] = elbo.backward()
            optimizer.step()
            
        if callback is not None:
            callback()
            
    return (optimizer, batch_stats)


def train_svb(model, X, epochs=1, nbatches=1, lrate_nnet=1e-3,
              lrate_prior=1e-1, update_prior=True, update_nnet=True, 
              kl_weight=1., state=None, nsamples=1, callback=None):
    
    batches = X.view(nbatches, -1, 2)
    
    mf_groups = model.mean_field_groups if update_prior else model.normal.mean_field_groups
    nnet_parameters = model.modules_parameters() if update_nnet else range(0)

    if state is None:
        std_optimizer = torch.optim.Adam(nnet_parameters, lr=lrate_nnet, 
                                         weight_decay=1e-2)
        optimizer = beer.BayesianModelOptimizer(mf_groups, lrate=lrate_prior, 
                                                std_optim=std_optimizer)
    else:
        optimizer = state
    
    for epoch in range(epochs):
        # Randomized the order of the batches.
        batch_ids = list(range(len(batches)))
        random.shuffle(batch_ids)
        for batch_id in batch_ids:
            optimizer.init_step()
            kwargs = {'kl_weight': kl_weight, 'datasize': len(X)}
            elbo = beer.evidence_lower_bound(model, batches[batch_id], 
                                             **kwargs)
            elbo.backward()
            optimizer.step()
        
        if callback is not None:
            callback()
            
        # Monitor the evidence lower bound after each epoch.
        #elbo = beer.evidence_lower_bound(model, X, **kwargs)
        #elbos.append(float(elbo) / len(X))
    
    return optimizer


# %%
def plot_latent_space(fig, model, X, use_mean=True):
    enc_states = vae.encoder(X)
    post_params = vae.encoder_problayer(enc_states)
    samples, _ = vae.encoder_problayer.samples_and_llh(post_params, use_mean=use_mean)
    samples = samples.data.numpy()
    fig.circle(samples[:, 0], samples[:, 1])
    
def plot_density(fig, model, x_range, y_range, nsamples=10, marginal=False):
    xy = np.mgrid[x_range[0]:x_range[1]:100j, y_range[0]:y_range[1]:100j].reshape(2,-1).T
    xy = torch.from_numpy(xy).float()
    
    mllhs = []
    for i in range(nsamples):
        if marginal:
            mllhs.append(model.marginal_log_likelihood(xy).view(-1, 1))
        else:
            mllhs.append(model.expected_log_likelihood(xy).view(-1, 1))
    mllhs = torch.cat(mllhs, dim=-1).mean(dim=-1)
    mllhs = mllhs.detach().numpy().reshape(100, 100)
    mlhs = np.exp(mllhs)
    width, height = x_range[1] - x_range[0] / 100, y_range[1] - y_range[0] / 100
    fig.image(image=[mlhs.T], x=x_range[0], y=y_range[0], dw=2 * width, dh=2 * height)


# %%
def create_vae(mean, var):
    prior = beer.dists.NormalDiagonalCovariance()
    return beer.VAE()


# %%
vae = create_vae(global_mean, global_var)

svb_elbos = []
svb_elbos2 = []
svb_elbos_test = []
def log_pred():
    elbo = beer.evidence_lower_bound(vae, X)
    svb_elbos.append(float(elbo) / len(X))
    elbo = beer.collapsed_evidence_lower_bound(vae, X, kl_weight=1.)
    svb_elbos2.append(float(elbo) / len(X))
    elbo = beer.evidence_lower_bound(vae, test_X, datasize=len(test_X), use_mean=False)
    svb_elbos_test.append(float(elbo) / len(test_X))
    
# training the vae.
state = train_svb(vae, X, epochs=5000, nbatches=10, nsamples=1, callback=log_pred,
                  update_prior=True)

# Plotting
fig1 = figure(width=300, height=300)
fig1.line(range(len(svb_elbos)), svb_elbos, legend='ELBO')
fig1.legend.location = 'bottom_right'

fig2 = figure(width=300, height=300, x_range=(-7, 7), y_range=(-7, 7))
mean, cov = vae.latent_model.mean, vae.latent_model.cov
plotting.plot_normal(fig2, mean.numpy(), cov.numpy(),alpha=.1)
plot_latent_space(fig2, vae, X, use_mean=False)

fig3 = figure(width=300, height=300, x_range=x_range, y_range=y_range)
plot_density(fig3, vae, x_range, y_range, nsamples=100)

show(gridplot([[fig1, fig2, fig3]]))


# %%
vae = create_vae(global_mean, global_var)

cvb_elbos_test = []
cvb_elbos = []
cvb_elbos2 = []
def log_pred():
    elbo = beer.evidence_lower_bound(vae, X)
    cvb_elbos.append(float(elbo) / len(X))
    elbo = beer.collapsed_evidence_lower_bound(vae, X, kl_weight=1.)
    cvb_elbos2.append(float(elbo) / len(X))
    elbo = beer.collapsed_evidence_lower_bound(vae, test_X, kl_weight=1., use_mean=False)
    cvb_elbos_test.append(float(elbo) / len(test_X))

# training the vae.
#state = train_cvb(vae, X, epochs=10, nbatches=10, callback=log_pred,  kl_weight=0., update_prior=True)
state = train_cvb(vae, X, epochs=5000, nbatches=10, callback=log_pred, 
                  nsamples=1, update_prior=True)

# Plotting
fig1 = figure(width=300, height=300)
fig1.line(range(len(cvb_elbos)), cvb_elbos, legend='ELBO')
fig1.legend.location = 'bottom_right'

fig2 = figure(width=300, height=300, x_range=(-7, 7), y_range=(-7, 7))
mean, cov = vae.latent_model.mean, vae.latent_model.cov
plotting.plot_normal(fig2, mean.numpy(), cov.numpy(),alpha=.1)
plot_latent_space(fig2, vae, X, use_mean=False)

fig3 = figure(width=300, height=300, x_range=x_range, y_range=y_range)
plot_density(fig3, vae, x_range, y_range, nsamples=100, marginal=True)

show(gridplot([[fig1, fig2, fig3]]))


# %%
# Plotting

fig1 = figure(title='ELBO (train set)', width=300, height=300, y_range=(-5, 1))
fig1.line(range(len(svb_elbos2)), svb_elbos, color='blue', legend='SVB')
fig1.line(range(len(cvb_elbos2)), cvb_elbos, color='green', legend='CVB')
fig1.legend.location = 'bottom_right'

fig2 = figure(title='Col. ELBO (train set)', width=300, height=300, y_range=(-5, 1))
fig2.line(range(len(svb_elbos2)), svb_elbos2, color='blue', legend='SVB')
fig2.line(range(len(cvb_elbos2)), cvb_elbos2, color='green', legend='CVB')
fig2.legend.location = 'bottom_right'

fig3 = figure(title='log pred. (test set)', width=300, height=300, y_range=(-5, 1))
fig3.line(range(len(svb_elbos_test)), svb_elbos_test, color='blue', legend='SVB')
fig3.line(range(len(cvb_elbos_test)), cvb_elbos_test, color='green', legend='CVB')
fig3.legend.location = 'bottom_right'

show(gridplot([[fig1, fig2, fig3]]))

