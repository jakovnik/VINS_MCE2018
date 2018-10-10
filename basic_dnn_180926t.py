import numpy as np
from vins_utils import load_ivector, length_norm, calculate_EER, get_trials_label_with_confusion
from vins_utils import calculate_EER_with_confusion, load_test_info
import sys
import os
import cntk as C

model_name = '../tijana/dnn_model_final90.model' #sys.argv[1] #
mtn = 1# int(sys.argv[2]) #0
output_dir = '../csv_output/'
test_filename = '../data/tst_evaluation_keys.csv'


## making dictionary to find blacklist pair between train and test dataset
bl_match = np.loadtxt('../data/bl_matching_dev.csv', dtype=np.str_,skiprows=1)
n_bl_spks = len(bl_match)
unique_train_id = np.array(['xxxx']*n_bl_spks)
for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    trn_id = line_s[2].split('_')[-1]
    unique_train_id[iter] = trn_id
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
dev_bl_id, dev_bl_utt, dev_bl_ivector = load_ivector('../data/dev_blacklist.csv')
dev_bg_id, dev_bg_utt, dev_bg_ivector = load_ivector('../data/dev_background.csv')
tst_id, tst_utt, tst_ivector = load_ivector('../data/tst_evaluation.csv')
print('Applying length normalization...')
dev_bl_ivector = length_norm(dev_bl_ivector)
dev_bg_ivector = length_norm(dev_bg_ivector)
tst_ivector = length_norm(tst_ivector)
print('Preparing evaluation data...')
dev_ivector = np.append(dev_bl_ivector, dev_bg_ivector,axis=0)
dev_trials = np.append( np.ones([len(dev_bl_id), 1]), np.zeros([len(dev_bg_id), 1]))
print('Loading DNN model')
if not os.path.isfile(model_name):
    print('Model "{}" does not exist'.format(model_name))
    sys.exit(1)
else:
    model = C.load_model(model_name)
print('Scoring on development set...')
scores = C.softmax(model.eval(dev_ivector)).eval()
print('Scoring on test set...')
tst_scores_e = C.softmax(model.eval(tst_ivector)).eval()
if mtn == 1:
    print('Applying multi-target normalization...')
    _, _, trn_bl_ivector = load_ivector('../data/trn_blacklist.csv')
    blscores = C.softmax(model.eval(trn_bl_ivector)).eval()
    mnorm_mu = np.mean(blscores,axis=0)
    mnorm_std = np.std(blscores,axis=0)
    for iter in range(np.shape(scores)[0]):
        scores[iter,:]= (scores[iter,:] - mnorm_mu) / mnorm_std
    for iter in range(np.shape(scores)[0]):
        tst_scores_e[iter,:]= (tst_scores_e[iter,:] - mnorm_mu) / mnorm_std
dev_scores = np.max(scores,axis=1)
print('Development set')
# Top-S detector EER
dev_EER = calculate_EER(dev_trials, dev_scores)
dev_identified_label = unique_train_id[np.argmax(scores,axis=1)]
dev_trials_label = np.append( dev_bl_id,dev_bg_id)
dev_trials_utt_label = np.append( dev_bl_utt,dev_bg_utt)
# Top-1 detector EER
dev_trials_confusion = get_trials_label_with_confusion(dev_identified_label, dev_trials_label, dev2train, dev_trials)
dev_EER_confusion = calculate_EER_with_confusion(dev_scores,dev_trials_confusion)
print('Generating submission file on development set')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
_, pure_m_name = os.path.split(model_name)
filename =  os.path.join(output_dir,pure_m_name + '_dev_fixed_primary.csv')
with open(filename, "w") as text_file:
    for iter,score in enumerate(dev_scores):
        id_in_trainset = dev_identified_label[iter].split('_')[0]
        input_file = dev_trials_utt_label[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))
print('Test set')
tst_scores = np.max(tst_scores_e,axis=1)
tst_trials, tst_trials_label = load_test_info(test_filename)
# Top-S detector EER
dev_EER = calculate_EER(tst_trials, tst_scores)
tst_identified_label = unique_train_id[np.argmax(tst_scores_e,axis=1)]
# Top-1 detector EER
tst_trials_confusion = get_trials_label_with_confusion(tst_identified_label, tst_trials_label, test2train, tst_trials)
dev_EER_confusion = calculate_EER_with_confusion(tst_scores,tst_trials_confusion)
filename =  os.path.join(output_dir,pure_m_name + '_tst_fixed_primary.csv')
with open(filename, "w") as text_file:
    for iter,score in enumerate(tst_scores):
        id_in_trainset = tst_identified_label[iter].split('_')[0]
        input_file = tst_utt[iter]
        text_file.write('%s,%s,%s\n' % (input_file,score,train2id[id_in_trainset]))








