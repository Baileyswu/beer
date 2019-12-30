
'''Implementation of the Variational Auto-Encoder with arbitrary
prior over the latent space.

'''

import torch
from .basemodel import Model
from ..dists.normaldiag import NormalDiagonalCovariance

__all__ = ['VAE', 'MvVAE']
    
# Parameterization of the Normal using the
# log diagonal covariance matrix.
class MeanLogDiagCov(torch.nn.Module):

    def __init__(self, mean, log_diag_cov):
        super().__init__()
        self.mean = mean
        self.log_diag_cov = log_diag_cov

    @property
    def diag_cov(self):
        # Make sure the variance is never 0.
        return 1e-5 + self.log_diag_cov.exp()
        
        
class VAE(Model):
    
    def __init__(self, prior, encoder, decoder):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.enc_mean_layer = torch.nn.Linear(encoder.dim_out, decoder.dim_in)
        self.enc_var_layer = torch.nn.Linear(encoder.dim_out, decoder.dim_in)
        self.dec_mean_layer = torch.nn.Linear(decoder.dim_out, encoder.dim_in)
        self.dec_var_layer = torch.nn.Linear(decoder.dim_out, encoder.dim_in)
        torch.nn.init.zeros_(self.enc_var_layer.weight)
         
    def posteriors(self, X):
        'Forward the data to the encoder to get the variational posteriors.'
        H = self.encoder(X)
        return NormalDiagonalCovariance(
            MeanLogDiagCov(self.enc_mean_layer(H), self.enc_var_layer(H))
        )
    
    def pdfs(self, Z):
        'Return the normal densities given the latent variable Z'
        X = self.decoder(Z)
        return NormalDiagonalCovariance(
            MeanLogDiagCov(self.dec_mean_layer(X), self.dec_var_layer(X))
        )
    
    ####################################################################
    # Model interface.

    def normalloss(self, X):
        H = self.encoder(X)
        return torch.nn.functional.normalize(self.enc_var_layer(H))

    def mean_field_factorization(self):
        return self.prior.mean_field_factorization()

    def sufficient_statistics(self, data):
        return data

    def expected_log_likelihood(self, data, nsamples=1, llh_weight=1.,
                                kl_weight=1., **kwargs):
        posts = self.posteriors(data)
        
        # Local KL-divergence. There is a close for solution
        # for this term but we use sampling as it allows to
        # change the prior (GMM, HMM, ...) easily.
        samples = posts.sample(nsamples)
        s_samples = posts.sufficient_statistics(samples).mean(dim=1)
        ent = -posts(s_samples, pdfwise=True)
        s_samples = self.prior.sufficient_statistics(samples.view(-1, samples.shape[-1]))
        s_samples = s_samples.reshape(len(samples), -1, s_samples.shape[-1]).mean(dim=1)
        self.cache['prior_stats'] = s_samples
        xent = -self.prior.expected_log_likelihood(s_samples)
        local_kl_div = xent - ent
        
        # Approximate the expected log-likelihood with the
        # reparameterization trick.
        pdfs = self.pdfs(samples.view(-1, samples.shape[-1]))
        r_data = data[:, None, :].repeat(1, nsamples, 1).view(-1, data.shape[-1])
        llh = pdfs(pdfs.sufficient_statistics(r_data), pdfwise=True)
        llh = llh.reshape(len(data), nsamples, -1).mean(dim=1)
        
        return llh_weight * llh - kl_weight * local_kl_div + self.normalloss(data).mean()

    def accumulate(self, stats, parent_msg=None):
        return self.prior.accumulate(self.cache['prior_stats'])

class MvVAE(Model):
    def __init__(self, prior, enc1, dec1, enc2, dec2):
        super().__init__()
        self.prior = prior
        self.enc1, self.dec1 = enc1, dec1
        self.enc2, self.dec2 = enc2, dec2
        assert dec1.dim_in == dec2.dim_in, "decoders have the same dim_in"
        self.enc_mean_layer = torch.nn.Linear(enc1.dim_out + enc2.dim_out, dec1.dim_in)
        self.enc_var_layer = torch.nn.Linear(enc1.dim_out + enc2.dim_out, dec1.dim_in)
        self.dec1_mean_layer = torch.nn.Linear(dec1.dim_in, enc1.dim_in)
        self.dec1_var_layer = torch.nn.Linear(dec1.dim_in, enc1.dim_in)
        self.dec2_mean_layer = torch.nn.Linear(dec2.dim_in, enc2.dim_in)
        self.dec2_var_layer = torch.nn.Linear(dec2.dim_in, enc2.dim_in)
        torch.nn.init.zeros_(self.enc_var_layer.weight)
        
    def posteriors(self, X):
        'Forward the two-view data to the encoder to get the shared variational posteriors.'
        X1, X2 = X[0], X[1]
        H1, H2 = self.enc1(X1), self.enc2(X2)
        H = torch.cat((H1, H2), 1)
        return NormalDiagonalCovariance(
            MeanLogDiagCov(self.enc_mean_layer(H), self.enc_var_layer(H))
        )

    def pdfs(self, Z, view):
        'Return the normal desities in specific view given the latent variable Z'
        assert view == 1 or view == 2, "View number is constrained in 1 or 2"
        if view == 1:
            X = self.dec1(Z)
            return NormalDiagonalCovariance(
                MeanLogDiagCov(self.dec1_mean_layer(X), self.dec1_var_layer(X))
            )
        else:
            X = self.dec2(Z)
            return NormalDiagonalCovariance(
                MeanLogDiagCov(self.dec2_mean_layer(X), self.dec2_var_layer(X))
            )

    ############################################################################
    # Model interface

    def mean_field_factorization(self):
        return self.prior.mean_field_factorization()

    def sufficient_statistics(self, data):
        return data

    def llh_in_view(self, view_n, samples, data, n_samples):
        pdfs = self.pdfs(samples.view(-1, samples.shape[-1]), view_n)
        r_data = data[:, None, :].repeat(1, n_samples, 1).view(-1, data.shape[-1])
        llh = pdfs(pdfs.sufficient_statistics(r_data), pdfwise=True)
        llh = llh.reshape(len(data), n_samples, -1).mean(dim=1)
        return llh
    
    def normalloss(self, X):
        X1, X2 = X[0], X[1]
        H1, H2 = self.enc1(X1), self.enc2(X2)
        H = torch.cat((H1, H2), 1)
        return torch.nn.functional.normalize(self.enc_var_layer(H))

    def expected_log_likelihood(self, s_stats, n_samples=1, llh_weight=1., kl_weight=1., **kwargs):
        posts = self.posteriors(s_stats)
        samples = posts.sample(n_samples)

        # Approximate the expected log-likelihood in two views with reparameterization trick.
        llh = self.llh_in_view(1, samples, s_stats[0], n_samples) \
            + self.llh_in_view(2, samples, s_stats[1], n_samples)

        # Local KL-divergence.
        ent = -posts(posts.sufficient_statistics(samples).mean(dim=1), pdfwise=True)
        s_samples = self.prior.sufficient_statistics(samples.view(-1, samples.shape[-1]))
        s_samples = s_samples.reshape(len(samples), -1, s_samples.shape[-1]).mean(dim=1)
        self.cache['prior_stats'] = s_samples
        xent = -self.prior.expected_log_likelihood(s_samples)

        local_kl_div = xent - ent
        return llh_weight * llh - kl_weight * local_kl_div

    def accumulate(self, s_stats, parent_msg=None):
        return self.prior.accumulate(self.cache['prior_stats'])