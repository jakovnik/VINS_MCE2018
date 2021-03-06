import numpy as np
from vins_utils import load_ivector, length_norm, make_spkvec, calculate_EER, get_trials_label_with_confusion
from vins_utils import calculate_EER_with_confusion, load_test_info
from gplda_em import gplda_em, score_plda
import sys
import os

ver = sys.argv[1] #'gplda180829v1'
n_phi = int(sys.argv[2]) #300
mtn = int(sys.argv[3])
n_iter = int(sys.argv[4]) #20
c_try = int(sys.argv[5])
model_dir = '../models/'+ver
output_dir = '../csv_output/'
test_filename = '../data/tst_evaluation_keys.csv'

## making dictionary to find blacklist pair between train and test dataset
bl_match = np.loadtxt('../data/bl_matching.csv',dtype=np.str_,skiprows=1)
dev2train={}
train2id={}
test2train={}
for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    dev2train[line_s[1].split('_')[-1]]= line_s[3].split('_')[-1]
    train2id[line_s[3].split('_')[-1]]= line_s[0].split('_')[-1]
    test2train[line_s[2].split('_')[-1]]= line_s[3].split('_')[-1]
print('Loading i-vectors...')
trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('../data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('../data/trn_background.csv')
dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('../data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('../data/dev_background.csv')
tst_id, tst_utt, tst_ivector = load_ivector('../data/tst_evaluation.csv')
print('Calculating speaker mean vector...')
spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector,trn_bl_id)
print('Applying length normalization...')
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)
if not os.path.isfile(os.path.join(model_dir,'plda'+str(c_try)+'.npz')):
    print('Estimating PLDA parameters...')
    trn_ivector = np.append(trn_bl_ivector,trn_bg_ivector,axis=0)
    trn_id = np.append(trn_bl_id,trn_bg_id,axis=0)
    phi, sigma, W1, mean = gplda_em(np.transpose(trn_ivector),trn_id,n_phi,n_iter)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    np.savez(os.path.join(model_dir,'plda'+str(c_try)+'.npz'),phi=phi,sigma=sigma,W1=W1,mean=mean)
else:
    print('Loading PLDA parameters...')
    data = np.load(os.path.join(model_dir,'plda'+str(c_try)+'.npz'))
    phi = data['phi']
    sigma = data['sigma']
    W1 = data['W1']
    mean = data['mean']
print('\nDev set score using train set :')
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))
scores = score_plda(phi,sigma,W1,mean,np.transpose(spk_mean),np.transpose(dev_ivector))
tst_scores_e = score_plda(phi,sigma,W1,mean,np.transpose(spk_mean),np.transpose(tst_ivector))
if mtn == 1:
    print('Applying multi-target normalization...')
    blscores = score_plda(phi,sigma,W1,mean,np.transpose(spk_mean),np.transpose(trn_bl_ivector))
    mnorm_mu = np.mean(blscores,axis=1)
    mnorm_std = np.std(blscores,axis=1)
    for iter in range(np.shape(scores)[1]):
        scores[:,iter]= (scores[:,iter] - mnorm_mu) / mnorm_std
    for iter in range(np.shape(scores)[1]):
        tst_scores_e[:,iter]= (tst_scores_e[:,iter] - mnorm_mu) / mnorm_std
dev_scores = np.max(scores,axis=0)
print('Development set')
# Top-S detector EER
dev_EER = calculate_EER(dev_trials, dev_scores)
dev_identified_label = spk_mean_label[np.argmax(scores,axis=0)]
dev_trials_label = np.append( dev_bl_id,dev_bg_id)
dev_trials_utt_label = np.append( dev_bl_utt,dev_bg_utt)
# Top-1 detector EER
dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train, dev_trials)
dev_EER_confusion = calculate_EER_with_confusion(dev_scores,dev_trials_confusion)
print('Generating submission file on development set')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
filename =  os.path.join(output_dir,ver + '_' + str(c_try) + '_dev_fixed_primary.csv')
with open(filename, "w") as text_file:
    for iter,score in enumerate(dev_scores):
        id_in_trainset = dev_identified_label[iter].split('_')[0]
        input_file = dev_trials_utt_label[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))
print('Test set')
tst_scores = np.max(tst_scores_e,axis=0)
tst_trials, tst_trials_label = load_test_info(test_filename)
# Top-S detector EER
dev_EER = calculate_EER(tst_trials, tst_scores)
tst_identified_label = spk_mean_label[np.argmax(tst_scores_e,axis=0)]
# Top-1 detector EER
tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train, tst_trials)
dev_EER_confusion = calculate_EER_with_confusion(tst_scores,tst_trials_confusion)
filename =  os.path.join(output_dir,ver + '_' + str(c_try) +  '_tst_fixed_primary.csv')
with open(filename, "w") as text_file:
    for iter,score in enumerate(tst_scores):
        id_in_trainset = tst_identified_label[iter].split('_')[0]
        input_file = tst_utt[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))

