import numpy as np
import time

def calc_white_mat(X):
    _, S, Vh = np.linalg.svd(X)
    X = np.dot(np.transpose(Vh),np.diag(1/(np.sqrt(S) + 1E-10)))
    return X


def maximization_plda(data, Ey, Eyy):
    n_samples = data.shape[1]
    data_square = np.dot(data,np.transpose(data))
    phi = np.dot(np.dot(data,np.transpose(Ey)),np.linalg.inv(Eyy))
    sigma = 1/n_samples * (data_square - np.dot(np.dot(phi,Ey),np.transpose(data)))
    return phi, sigma


def expectation_plda(data, phi, sigma, spk_counts):
    n_phi = phi.shape[1]
    n_samples = data.shape[1]
    n_speakers = spk_counts.shape[0]
    Ey = np.zeros((n_phi,n_samples))
    Eyy = np.zeros((n_phi,n_phi))
    uniq_freqs = np.unique(spk_counts)
    inv_terms = np.zeros((len(uniq_freqs),n_phi,n_phi))
    phiT_inv_sigma_phi = np.dot(np.dot(np.transpose(phi),np.linalg.inv(sigma)),phi)
    I = np.eye(n_phi)
    for i, val in enumerate(uniq_freqs):
        n_phiT_inv_sigma_phi = val * phiT_inv_sigma_phi
        Cyy = np.linalg.pinv(I + n_phiT_inv_sigma_phi)
        inv_terms[i] = Cyy
    data = np.dot(np.linalg.inv(sigma),data)
    cnt = 0
    for spk in range(n_speakers):
        n_sessions = spk_counts[spk]
        idx = np.arange(cnt,cnt+n_sessions)
        cnt += n_sessions
        l_data = data[:,idx]
        phiT_inv_sigma_y = np.sum(np.dot(np.transpose(phi),l_data),axis=1)
        Cyy = np.reshape(inv_terms[uniq_freqs == n_sessions],(n_phi,n_phi))
        Ey_spk = np.dot(Cyy,phiT_inv_sigma_y)
        Eyy_spk = Cyy + np.outer(Ey_spk,Ey_spk)
        Eyy = Eyy + n_sessions * Eyy_spk
        Ey[:,idx] = np.tile(Ey_spk.reshape(n_phi,1), (1, n_sessions))
    return Ey, Eyy


def gplda_em(data, spk_labs, n_phi, n_iter):
    n_dim = data.shape[0]
    n_obs = data.shape[1]
    if n_obs != len(spk_labs):
        raise Exception('Number of data samples should match the number of labels')
    idx = np.argsort(spk_labs)
    spk_labs = spk_labs[idx]
    data = data[:,idx]
    _, ia, ic = np.unique(spk_labs,return_index=True, return_inverse=True)
    spk_counts, _ = np.histogram(ic,bins=np.arange(0,ia.size+1)) # number of sessions per speaker
    mean = np.mean(data,axis=1,dtype=np.float64)
    data = np.apply_along_axis(lambda x, y : x - y,0,data,mean)
    norm = np.sqrt(np.sum(data**2,axis=0))
    data = np.apply_along_axis(lambda x, y: x / y, 1, data, norm)
    W1 = calc_white_mat(np.cov(data))
    data = np.dot(np.transpose(W1),data) #whitening the data
    print('Randomly initializing the PLDA hyperparameters ...')
    sigma = 100*np.random.randn(n_dim,n_dim)#covariance matrix of the residual term
    #sigma = np.loadtxt('sig.txt',dtype=np.float64)
    phi = np.random.randn(n_dim,n_phi) #factor loading matrix (Eignevoice matrix)
    #phi = np.loadtxt('phi.txt',dtype=np.float64)
    phi = np.apply_along_axis(lambda x, y : x - y,0,phi,np.mean(phi,axis=1))
    W2 = calc_white_mat(np.dot(np.transpose(phi),phi))
    phi = np.dot(phi,W2) #orthogonalize Eigenvoices (columns)
    print('Re-estimating the Eigenvoice subspace with {} factors ...'.format(n_phi))
    for iter in range(n_iter):
        print('EM iter#: {} \t'.format(iter))
        bgn_time = time.time()
        # expectation
        Ey, Eyy = expectation_plda(data, phi, sigma, spk_counts)
        # maximization
        phi_n, sigma_n = maximization_plda(data, Ey, Eyy)
        d_phi = np.sum(np.sum(np.abs(phi_n - phi)))
        d_sigma = np.sum(np.sum(np.abs(sigma_n - sigma)))
        print('[elaps = {:.2f} s], d_phi = {}, d_sigma = {}'.format(time.time()-bgn_time,d_phi,d_sigma))
        phi = phi_n
        sigma = sigma_n
    return phi, sigma, W1, mean


def score_plda(phi,sigma,W1,mean,model_ivecs, unk_ivecs):
    # prepare model i-vectors
    model_ivecs = np.apply_along_axis(lambda x, y: x - y, 0, model_ivecs, mean)
    norm = np.sqrt(np.sum(model_ivecs ** 2, axis=0))
    model_ivecs = np.apply_along_axis(lambda x, y: x / y, 1, model_ivecs, norm)
    model_ivecs = np.dot(np.transpose(W1),model_ivecs) #whitening the data
    # prepare unknown i-vectors
    unk_ivecs = np.apply_along_axis(lambda x, y: x - y, 0, unk_ivecs, mean)
    norm = np.sqrt(np.sum(unk_ivecs ** 2, axis=0))
    unk_ivecs = np.apply_along_axis(lambda x, y: x / y, 1, unk_ivecs, norm)
    unk_ivecs = np.dot(np.transpose(W1), unk_ivecs)  # whitening the data
    # score data
    n_phi = phi.shape[0]
    sigma_ac = np.dot(phi,np.transpose(phi))
    sigma_tot = sigma_ac + sigma
    inv_sigma_tot = np.linalg.pinv(sigma_tot)
    inv_sigma = np.linalg.pinv(sigma_tot - np.dot(np.dot(sigma_ac,inv_sigma_tot),sigma_ac))
    Q = inv_sigma_tot - inv_sigma
    P = np.dot(np.dot(inv_sigma_tot,sigma_ac),inv_sigma)
    U, S, _ = np.linalg.svd(P)
    lambda_v = np.diag(S[0:n_phi])
    U_k = U[:,0:n_phi]
    Q_hat = np.dot(np.dot(np.transpose(U_k),Q),U_k)
    model_ivecs = np.dot(np.transpose(U_k),model_ivecs)
    unk_ivecs = np.dot(np.transpose(U_k), unk_ivecs)
    score_h1 = np.diag(np.dot(np.dot(np.transpose(model_ivecs),Q_hat),model_ivecs))
    score_h2 = np.diag(np.dot(np.dot(np.transpose(unk_ivecs), Q_hat), unk_ivecs))
    score_h1h2 = 2*np.dot(np.dot(np.transpose(model_ivecs), lambda_v), unk_ivecs)
    scores = np.apply_along_axis(lambda x, y: x + y, 0, score_h1h2, score_h1)
    scores = np.apply_along_axis(lambda x, y: x + y, 1, scores, score_h2)
    return scores


#data = np.loadtxt('data.txt',dtype=np.float64)
#sid = np.loadtxt('sid.txt',dtype=np.float64)
#phi, sigma, W1, mean = gplda_em(data,sid,19,10)

#phi = np.loadtxt('phi.txt')
#sigma = np.loadtxt('sig.txt')
#W1 = np.loadtxt('w.txt')
#mean = np.loadtxt('m.txt')
#model_ivecs = np.loadtxt('mod_ivec.txt')
#unk_ivecs = np.loadtxt('unk_ivec.txt')

#scores = score_plda(phi,sigma,W1,mean,model_ivecs, unk_ivecs)