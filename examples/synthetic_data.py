import numpy as np

def generate_sequential_data():
    nsamples = 600
    ndim = 2
    nstates = 3
    trans_mat = np.array([[.3, .7, 0], [0, .5, .5], [.7, 0, .3]])

    means = [np.array([-1.5, 4]) * 2,np.array([5, 5]) * 2, np.array([1, -2])] * 2
    covs = [np.array([[.75, -.5], [-.5, 2.]]), np.array([[2, 1], [1, .75]]), np.array([[1, 0], [0, 1]])]

    states = np.zeros(nsamples, dtype=int)
    data = np.zeros((nsamples, ndim))
    states[0] = 0
    data[0] = np.random.multivariate_normal(means[states[0]], covs[states[0]], size=1)

    for n in range(1, nsamples):
        states[n] = np.random.choice(np.arange(nstates), p=trans_mat[states[n-1]])
        data[n] = np.random.multivariate_normal(means[states[n]], covs[states[n]], size=1)

    return data, states

def generate_multiview_sequential_data():
    nsamples = 400
    ndim = 2
    nstates = 3
    views = 2

    trans_mat = np.array([[.3, .7, 0], [0, .5, .5], [.7, 0, .3]])

    means_x1 = [np.array([-1.5, 4]), np.array([5, 5]), np.array([1, -2])]
    means_x2 = [np.array([-1.5, 4]) * 2, np.array([5, 5]) * 2, np.array([1, -2]) * 2]
    covs = [np.array([[.75, -.5], [-.5, 2.]]), np.array([[2, 1], [1, .75]]), np.array([[1, 0], [0, 1]])]

    states = np.zeros(nsamples, dtype=int)
    data = np.zeros((views, nsamples, ndim))
    states[0] = 0
    data[0][0] = np.random.multivariate_normal(means_x1[states[0]], covs[states[0]], size=1)
    data[1][0] = np.random.multivariate_normal(means_x2[states[0]], covs[states[0]], size=1)

    for n in range(1, nsamples):
        states[n] = np.random.choice(np.arange(nstates), p=trans_mat[states[n-1]])
        data[0][n] = np.random.multivariate_normal(means_x1[states[n]], covs[states[n]], size=1)
        data[1][n] = np.random.multivariate_normal(means_x2[states[n]], covs[states[n]], size=1)

    return data, states