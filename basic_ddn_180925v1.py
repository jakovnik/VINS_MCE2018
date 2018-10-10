import numpy as np
from vins_utils import load_ivector, length_norm, create_dnn_model
import cntk as C
import time
import sys
import os

ver = sys.argv[1] #'dnn180925v1' #
seed = int(sys.argv[2]) #12 #
model_dir = '../models/'+ver
loss_dir = '../loss_output/'

def train(train_x, train_y, seed, model_dir, loss_dir):
    input_dim = 600
    output_dim = 3631
    num_epochs = 100
    hidden_layer_type = ['TANH', 'TANH']
    hidden_layer_size = [1024, 1024]
    momentum = 0.9
    finetune_lr = 0.01
    l2_regularization_weight = 0.00001
    C.cntk_py.set_fixed_random_seed(seed)
    print('Creating DNN model...')
    input = C.input_variable(input_dim)
    output = C.input_variable(output_dim)
    dnn_model = create_dnn_model(input, hidden_layer_type, hidden_layer_size, output_dim)
    epoch_num = 0
    current_finetune_lr = finetune_lr
    current_momentum = momentum
    train_loss_output = []
    print('Learning...')
    while (epoch_num < num_epochs):
        print ('started epoch %i' % epoch_num)
        epoch_num += 1
        sub_start_time = time.time()
        lr_schedule = C.learning_rate_schedule(current_finetune_lr, C.UnitType.minibatch)
        momentum_schedule = C.momentum_schedule(current_momentum)
        learner = C.momentum_sgd(dnn_model.parameters, lr_schedule, momentum_schedule, unit_gain = False,
                                 l1_regularization_weight=0, l2_regularization_weight= l2_regularization_weight )
        #learner = C.adadelta(dnn_model.parameters, lr_schedule, rho=0.95, epsilon=1e-8, l1_regularization_weight=0,
        #                    l2_regularization_weight= 0.00001 )
        loss=C.cross_entropy_with_softmax(dnn_model,output)
        error=loss
        trainer = C.Trainer(dnn_model, (loss, error), [learner])
        train_error = []
        for i in range(len(train_x)):
            temp_train_x = np.float32(train_x[i])
            temp_train_y = np.float32(train_y[i])
            trainer.train_minibatch({input: temp_train_x, output: temp_train_y})
            train_error.append(trainer.previous_minibatch_loss_average)
        this_train_loss = np.mean(train_error)
        sub_end_time = time.time()
        print ('time for 1 epoch is %.1f' % (sub_end_time-sub_start_time))
        train_loss_output.append(this_train_loss)
        print('loss is %.4f' % this_train_loss)
        if np.remainder(epoch_num,10) == 0:
            nnets_file_name = 'dnn_model_ep'+np.str(epoch_num)+'.model'
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            dnn_model.save(os.path.join(model_dir,nnets_file_name))
            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)
            np.savetxt(os.path.join(loss_dir,'loss_curve_ep'+np.str(epoch_num)+'.csv'), train_loss_output)
    nnets_file_name = 'dnn_model_final.model'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    dnn_model.save(os.path.join(model_dir, nnets_file_name))
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    np.savetxt(os.path.join(loss_dir, 'loss_curve_final' + np.str(epoch_num) + '.csv'), train_loss_output)


## making dictionary to find blacklist pair between train and test dataset
bl_match = np.loadtxt('../data/bl_matching_dev.csv', dtype=np.str_,skiprows=1)
n_bl_spks = len(bl_match)
dev2train={}
dev2id={}
train2dev={}
train2id={}
unique_train_id = np.array(['xxxx']*n_bl_spks)
for iter, line in enumerate(bl_match):
    line_s = line.split(',')
    trn_id = line_s[2].split('_')[-1]
    dev2train[line_s[1].split('_')[-1]]= trn_id
    dev2id[line_s[1].split('_')[-1]]= line_s[0].split('_')[-1]
    train2dev[trn_id]= line_s[1].split('_')[-1]
    train2id[trn_id]= line_s[0].split('_')[-1]
    unique_train_id[iter] = trn_id
print('Loading i-vectors ...')
trn_bl_id, trn_bl_utt, trn_bl_ivector = load_ivector('../data/trn_blacklist.csv')
trn_bg_id, trn_bg_utt, trn_bg_ivector = load_ivector('../data/trn_background.csv')
print('Applying length normalization...')
trn_bl_ivector = length_norm(trn_bl_ivector)
trn_bg_ivector = length_norm(trn_bg_ivector)
print('Creating DNN outputs...')
trn_bl_out=np.zeros((len(trn_bl_id),n_bl_spks))
for iter, val in enumerate(unique_train_id):
    trn_bl_out[trn_bl_id==val,iter] = 1
trn_bg_out = np.ones((len(trn_bg_id),n_bl_spks))*(1/n_bl_spks)
trn_ivector_ordered = np.append(trn_bl_ivector, trn_bg_ivector,axis=0)
trn_labs_ordered = np.append(trn_bl_out, trn_bg_out, axis=0)
print('Shuffling data...')
s = np.arange(trn_labs_ordered.shape[0])
np.random.shuffle(s)
trn_ivector = trn_ivector_ordered[s,:]
trn_labs = trn_labs_ordered[s,:]
print('Start training')
train(trn_ivector, trn_labs, seed, model_dir, loss_dir)


