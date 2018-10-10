import numpy as np
from vins_utils import load_ivector, length_norm, make_spkvec, calculate_EER, get_trials_label_with_confusion
from vins_utils import calculate_EER_with_confusion, load_test_info
from gplda_em import gplda_em, score_plda
import os
import sys
import cntk as C
from sklearn.externals import joblib

ver = sys.argv[1] #'lda_plda_180927v1t' #
n_lda_cmp = int(sys.argv[2]) #300#
n_phi = int(sys.argv[3]) #100#
n_iter = int(sys.argv[4]) #10#
c_try = int(sys.argv[5]) #0#
dnn_model_name = sys.argv[6] #'../tijana/dnn_model_final90.model'
dnn_threshold = float(sys.argv[7]) # 0.01
dnn_penalty = float(sys.argv[8]) #300

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
print('Loading i-vectors ...')
trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('../data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('../data/trn_background.csv')
dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('../data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('../data/dev_background.csv')
tst_id, tst_utt, tst_ivector = load_ivector('../data/tst_evaluation.csv')
print('Applying length normalization 1...')
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)
print('Preparing evaluation data...')
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))
print('Loading DNN model')
if not os.path.isfile(dnn_model_name):
    print('Model "{}" does not exist'.format(dnn_model_name))
    sys.exit(1)
else:
    model = C.load_model(dnn_model_name)
print('Scoring on development set...')
dnn_scores = C.softmax(model.eval(dev_ivector)).eval()
dnn_scores = np.max(dnn_scores,axis=1)
identified_backgrounds = dnn_scores <= dnn_threshold
print('Scoring on test set...')
dnn_tst_scores = C.softmax(model.eval(tst_ivector)).eval()
dnn_tst_scores = np.max(dnn_tst_scores,axis=1)
identified_backgrounds_tst = dnn_tst_scores <= dnn_threshold
print('Loading LDA ...')
lda_file = os.path.join(model_dir,'lda_clf.pkl')
if not os.path.isfile(lda_file):
    print('LDA transformation "{}" does not exist'.format(lda_file))
    sys.exit(1)
else:
    clf = joblib.load(lda_file)
print('Applying LDA transformation...')
trn_bl_ivector = clf.transform(trn_bl_ivector)
trn_bg_ivector = clf.transform(trn_bg_ivector)
dev_bl_ivector = clf.transform(dev_bl_ivector)
dev_bg_ivector = clf.transform(dev_bg_ivector)
tst_ivector = clf.transform(tst_ivector)
print('Calculating speaker mean vector...')
spk_mean, spk_mean_label = make_spkvec(trn_bl_ivector,trn_bl_id)
print('Applying length normalization 2...')
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)


if not os.path.isfile(os.path.join(model_dir,'plda'+str(c_try)+'.npz')):
    print('PLDA transformation "{}" does not exist'.format(lda_file))
    sys.exit(1)
else:
    print('Loading PLDA parameters...')
    data = np.load(os.path.join(model_dir,'plda'+str(c_try)+'.npz'))
    phi = data['phi']
    sigma = data['sigma']
    W1 = data['W1']
    mean = data['mean']
print('Prepare development set:')
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))
print('Scoring development set:')
scores = score_plda(phi,sigma,W1,mean,np.transpose(spk_mean),np.transpose(dev_ivector))
print('Scoring test set:')
tst_scores_e = score_plda(phi,sigma,W1,mean,np.transpose(spk_mean),np.transpose(tst_ivector))
print('Development set')
scores[:, np.transpose(identified_backgrounds)] -= dnn_penalty
dev_scores = np.max(scores,axis=0)
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
    os.makedirs(output_dir)
filename =  os.path.join(output_dir,ver + '_' + str(c_try) + '_dev_fixed_primary.csv')
with open(filename, "w") as text_file:
    for iter,score in enumerate(dev_scores):
        id_in_trainset = dev_identified_label[iter].split('_')[0]
        input_file = dev_trials_utt_label[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))
print('Test set')
tst_scores_e[:, np.transpose(identified_backgrounds_tst)] -= dnn_penalty
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

