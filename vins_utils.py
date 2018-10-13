import numpy as np
from sklearn.metrics import roc_curve
import cntk as C

#These function are taken from mce2018_baseline_dev.py

def load_ivector(filename):
    utt = np.loadtxt(filename,dtype=np.str_,delimiter=',',skiprows=1,usecols=[0])
    ivector = np.loadtxt(filename,dtype='float32',delimiter=',',skiprows=1,usecols=range(1,601))
    spk_id = []
    for iter in range(len(utt)):
        spk_id = np.append(spk_id,utt[iter].split('_')[0])
    return spk_id, utt, ivector


def length_norm(mat):
# length normalization (l2 norm)
# input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
    norm_mat = []
    for line in mat:
        temp = line/(np.math.sqrt(sum(np.power(line,2)))+1e-16)
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def make_spkvec(mat, spk_label):
    # calculating speaker mean vector
    # input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
    #        spk_label = string vector ex) ['abce','cdgd']

    #     for iter in range(len(spk_label)):
    #         spk_label[iter] = spk_label[iter].split('_')[0]

    spk_label, spk_index = np.unique(spk_label, return_inverse=True)
    spk_mean = []
    mat = np.array(mat)

    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index == i)], axis=0))
    spk_mean = length_norm(spk_mean)
    return spk_mean, spk_label


def make_spkvec_no_norm(mat, spk_label):
    # calculating speaker mean vector
    # input: mat = [utterances X vector dimension] ex) (float) 8631 X 600
    #        spk_label = string vector ex) ['abce','cdgd']

    #     for iter in range(len(spk_label)):
    #         spk_label[iter] = spk_label[iter].split('_')[0]
    spk_label, spk_index = np.unique(spk_label, return_inverse=True)
    spk_mean = []
    mat = np.array(mat)
    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        spk_mean.append(np.mean(mat[np.nonzero(spk_index == i)], axis=0))
    return np.array(spk_mean), spk_label


def calculate_EER(trials, scores):
    # calculating EER of Top-S detector
    # input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background)
    #        scores = float vector
    # Calculating EER
    fpr, tpr, threshold = roc_curve(trials, scores, pos_label=1)
    fnr = 1 - tpr
    pos = np.argmin(abs(fnr - fpr))
    EER_threshold = threshold[pos]
    # print EER_threshold
    EER_fpr = fpr[pos]
    EER_fnr = fnr[pos]
    EER = 0.5 * (EER_fpr + EER_fnr)
    EER1 = -1
    if pos < len(fpr) - 1:
        y11 = fpr[pos]
        y12 = fpr[pos + 1]
        y21 = fnr[pos]
        y22 = fnr[pos + 1]
        EER1 = (y11 * y22 - y12 * y21) / (y11 - y12 - y21 + y22)
    print("Top S detector EER is {:0.2f}%/{:0.2f}%".format(EER * 100,EER1*100))
    return EER, EER_threshold


def get_trials_label_with_confusion(identified_label, groundtruth_label, dict4spk, is_trial):
    # determine if the test utterance would make confusion error
    # input: identified_label = string vector, identified result of test utterance among multi-target from the detection system
    #        groundtruth_label = string vector, ground truth speaker labels of test utterances
    #        dict4spk = dictionary, convert label to target set, ex) train2dev convert train id to dev id
    trials = np.zeros(len(identified_label))
    for iter in range(0, len(groundtruth_label)):
        enroll = identified_label[iter].split('_')[0]
        test = groundtruth_label[iter].split('_')[0]
        if is_trial[iter]:
            if enroll == dict4spk[test]:
                trials[iter] = 1  # for Target trial (blacklist speaker)
            else:
                trials[iter] = -1  # for Target trial (backlist speaker), but fail on blacklist classifier

        else:
            trials[iter] = 0  # for non-target (non-blacklist speaker)
    return trials


def calculate_EER_with_confusion(scores, trials):
    # calculating EER of Top-1 detector
    # input: trials = boolean(or int) vector, 1: postive(blacklist) 0: negative(background) -1: confusion(blacklist)
    #        scores = float vector

    # exclude confusion error (trials==-1)
    scores_wo_confusion = scores[np.nonzero(trials != -1)[0]]
    trials_wo_confusion = trials[np.nonzero(trials != -1)[0]]

    # dev_trials contain labels of target. (target=1, non-target=0)
    fpr, tpr, threshold = roc_curve(trials_wo_confusion, scores_wo_confusion, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    pos = np.argmin(abs(fnr - fpr))
    EER_threshold = threshold[pos]
    # EER withouth confusion error
    EER = fpr[pos]
    # Add confusion error to false negative rate(Miss rate)
    total_negative = len(np.nonzero(np.array(trials_wo_confusion) == 0)[0])
    total_positive = len(np.nonzero(np.array(trials_wo_confusion) == 1)[0])
    fp = fpr * np.float(total_negative)
    fn = fnr * np.float(total_positive)
    fn += len(np.nonzero(trials == -1)[0])
    total_positive += len(np.nonzero(trials == -1)[0])
    fpr = fp / total_negative
    fnr = fn / total_positive
    # EER with confusion Error
    pos = np.argmin(abs(fnr - fpr))
    EER_threshold = threshold[pos]
    EER_fpr = fpr[pos]
    EER_fnr = fnr[pos]
    EER = 0.5 * (EER_fpr + EER_fnr)
    EER1 = -1
    if pos < len(fpr) - 1:
        y11 = fpr[pos]
        y12 = fpr[pos + 1]
        y21 = fnr[pos]
        y22 = fnr[pos + 1]
        EER1 = (y11 * y22 - y12 * y21) / (y11 - y12 - y21 + y22)
    print("Top 1 detector EER is {:0.2f}%/{:0.2f}% (Total confusion error is {:d})".format((EER * 100),(EER1 * 100),
                                                                                  len(np.nonzero(trials == -1)[0])))
    return EER, EER_threshold



# These function are our own

# It extends the dataset with i-vectors representing linear combination existed i-vectors (all coefficents are
# non-negative and its sum is equal to 1)
def extend_dataset(ivecs, ids):
    unique_ids, output_ids = np.unique(ids, return_inverse=True)
    add_ivecs = []
    add_ids = []
    for i, id in enumerate(unique_ids):
        idx = i == output_ids
        this_spk_n_inst = sum(idx)
        for j in range(3*this_spk_n_inst):
            weights = np.random.rand(1,this_spk_n_inst)
            weights /= (sum(weights)+1E-16)
            new_ivec = np.dot(weights,ivecs[idx,:])
            add_ivecs.append(new_ivec)
            add_ids.append(id)
    n_vec = len(add_ivecs)
    if n_vec > 0:
        f_dim = add_ivecs[0].size
        ivecs = np.concatenate((ivecs,np.asarray(add_ivecs).reshape(n_vec,f_dim)),axis=0)
        ids = np.concatenate((ids,np.asarray(add_ids)),axis=0)
    return ivecs, ids


# To randomize training data in epochs.
class MyMiniBatch:
    my_permutation = np.array([])


    def __init__(self,bl_id,bg_id):
        self.unq_bl_id, self.spk_bl_id = np.unique(bl_id, return_inverse=True)
        self.unq_bg_id, self.spk_bg_id = np.unique(bg_id, return_inverse=True)
        self.n_spk_bl = len(self.unq_bl_id)
        self.n_spk_bg = len(self.unq_bg_id)
        self.n_spk_tot = self.n_spk_bl + self.n_spk_bg


    def get_minibatch(self,mb_size,ivec_bl,ivec_bg):
        if len(self.my_permutation) == 0:
            self.my_permutation = np.random.permutation(self.n_spk_tot)
        idx = np.arange(min(len(self.my_permutation),mb_size))
        this_mb = self.my_permutation[idx]
        self.my_permutation = np.delete(self.my_permutation,idx)
        spk_ids = []
        ivecs = np.array([])
        for i in this_mb:
            if i < self.n_spk_bl:
                idx = self.spk_bl_id == i
                spk_ids.extend([i]*sum(idx))
                ivecs = np.append(ivecs,ivec_bl[idx,:],axis=0) if len(ivecs) > 0 else ivec_bl[idx,:]
            else:
                k = i - self.n_spk_bl
                idx = self.spk_bg_id == k
                spk_ids.extend([i] * sum(idx))
                ivecs = np.append(ivecs, ivec_bg[idx, :], axis=0) if len(ivecs) > 0 else ivec_bg[idx, :]
        return ivecs, spk_ids


    def get_total_num_ivec(self):
        return len(self.spk_bl_id) + len(self.spk_bg_id)



def adjust_input_data_balanced(ivecs, ids,i_vector_dim,output_dim):
    mb_size = ivecs.shape[0]
    equal_counter = 0
    for i in range(mb_size):
        for j in range(i+1,mb_size):
            if ids[i] == ids[j]:
                equal_counter += 1
    i1 = np.ndarray(shape=(2*equal_counter, 1, i_vector_dim), dtype=np.float32)
    i2 = np.ndarray(shape=(2*equal_counter, 1, i_vector_dim), dtype=np.float32)
    t1 = np.ndarray(shape=(2*equal_counter, 1, output_dim), dtype=np.float32)
    w1 = np.ndarray(shape=(2*equal_counter, 1, 1), dtype=np.float32)
    weight_equal = 1
    weight_diff = 1
    unq_ids, unq_inv = np.unique(ids,return_inverse=True)
    n_samples = unq_inv.size
    counter = 0
    if output_dim == 1:
        for i, val in enumerate(unq_ids):
            idx = np.nonzero(unq_inv == i)[0]
            for j in idx:
                for k in idx:
                    if j < k:
                        #add positive example
                        i1[counter, :, :] = ivecs[j, :]
                        i2[counter, :, :] = ivecs[k, :]
                        t1[counter, 0, 0] = 1
                        w1[counter, 0, 0] = weight_equal
                        counter += 1
                        #add negative example
                        t = j
                        while t in idx:
                            t = np.random.randint(0,n_samples)
                        i1[counter, :, :] = ivecs[j, :]
                        i2[counter, :, :] = ivecs[t, :]
                        t1[counter, 0, 0] = 0
                        w1[counter, 0, 0] = weight_diff
                        counter += 1
    else:
        for i, val in enumerate(unq_ids):
            idx = np.nonzero(unq_inv == i)[0]
            for j in idx:
                for k in idx:
                    if j < k:
                        #add positive example
                        i1[counter, :, :] = ivecs[j, :]
                        i2[counter, :, :] = ivecs[k, :]
                        t1[counter, 0, 0] = 1
                        t1[counter, 0, 1] = 0
                        w1[counter, 0, 0] = weight_equal
                        counter += 1
                        #add negative example
                        t = j
                        while t in idx:
                            t = np.random.randint(0,n_samples)
                        i1[counter, :, :] = ivecs[j, :]
                        i2[counter, :, :] = ivecs[t, :]
                        t1[counter, 0, 0] = 0
                        t1[counter, 0, 1] = 1
                        w1[counter, 0, 0] = weight_diff
                        counter += 1
    return i1, i2, t1, w1



def score_models(distance_measure, unk_ivecs, spk_ivecs, calc_softmax= False):
    print('Score models')
    n_inputs = unk_ivecs.shape[0]
    n_spks = spk_ivecs.shape[0]
    scores = np.zeros(shape=(n_spks,n_inputs),dtype=np.float32)
    if calc_softmax:
        for j in range(n_spks):
            spk = spk_ivecs[j, :]
            print("speaker {}\n".format(j))
            for i in range(n_inputs):
                scores[j, i] = C.softmax(distance_measure.eval({distance_measure.arguments[0]: unk_ivecs[i, :],
                                                                distance_measure.arguments[1]: spk})).eval()[0, 0]
    else:
        for j in range(n_spks):
            spk = spk_ivecs[j, :]
            print("speaker {}\n".format(j))
            for i in range(n_inputs):
                scores[j, i] = distance_measure.eval({distance_measure.arguments[0]: unk_ivecs[i, :],
                                                               distance_measure.arguments[1]: spk})[0, 0]
    return scores



def score_models_fast(distance_measure, unk_ivecs, spk_ivecs, siam_output= 'hl2', calc_softmax= False):
    print('Score models')
    #n_spks = spk_ivecs.shape[0]
    node_in_graph = distance_measure.find_all_with_name(siam_output)
    #prov_f_dim = node_in_graph[0].shape[0]
    prov_output = C.combine(node_in_graph[0])
    #prov_spk = np.zeros(shape=(n_spks,prov_f_dim),dtype=np.float32)
    print('Transform blacklist speaker i-vectors')
    prov_spk = prov_output.eval(spk_ivecs)
    print('Transform dev. speaker i-vectors')
    prov_unk = prov_output.eval(unk_ivecs)
    print('Normalize transformed vectors')
    prov_spk=length_norm(prov_spk)
    prov_unk=length_norm(prov_unk)
    print('Calculate scores')
    scores = np.dot(prov_spk,prov_unk.transpose())
    if calc_softmax:
        scores = 1 / (1 + np.exp(-2 * scores))
    return scores


def score_models_fast_dense(distance_measure, unk_ivecs, spk_ivecs, siam_output= 'hl2', out_level='out_level', calc_softmax= False):
    print('Score models')
    #n_spks = spk_ivecs.shape[0]
    node_in_graph = distance_measure.find_all_with_name(siam_output)
    #prov_f_dim = node_in_graph[0].shape[0]
    prov_output = C.combine(node_in_graph[0])
    #prov_spk = np.zeros(shape=(n_spks,prov_f_dim),dtype=np.float32)
    print('Transform blacklist speaker i-vectors')
    prov_spk = prov_output.eval(spk_ivecs)
    print('Transform dev. speaker i-vectors')
    prov_unk = prov_output.eval(unk_ivecs)
    prov_unk = np.append(prov_unk,prov_unk,axis=1)
    n_spk = prov_spk.shape[0]
    n_unk = prov_unk.shape[0]
    scores = np.zeros(shape=(n_spk,n_unk),dtype=np.float32)
    n_feat = prov_spk.shape[1]
    g = distance_measure.find_by_name('out_level')
    substitutions = {g.inputs[2]: C.input_variable(800)}
    dense_out = g.clone('freeze',substitutions)
    print('Calculate scores')
    for i,v in enumerate(prov_spk):
        prov_unk[:,:n_feat] = v
        scores[i,:] = dense_out.eval(prov_unk)[:,0]
    if calc_softmax:
        scores = 1 / (1 + np.exp(-2 * scores))
    return scores



##################################
# Criteron functions definitions #
##################################
i_vector_dim = 600
from cntk.layers.typing import Tensor
from cntk.ops.functions import Function

def create_criterion_function_squared_error(model):
    @Function
    def criterion(input1 : Tensor[i_vector_dim], input2 : Tensor[i_vector_dim],
                  target : Tensor[1], weight : Tensor[1]):
        output = model(input1,input2)
        loss = C.squared_error(output, target) * weight
        errs = C.classification_error(output, target)
        return loss, errs
    return criterion


def create_criterion_function_cross_entropy(model):
    @Function
    def criterion(input1 : Tensor[i_vector_dim], input2 : Tensor[i_vector_dim],
                  target : Tensor[2], weight : Tensor[1]):
        output = model(input1,input2)
        loss = C.cross_entropy_with_softmax(output, target) * weight
        errs = C.classification_error(output, target)
        return loss, errs
    return criterion



def create_denoised_target_vectors(i_vecs, spk_label):
    # denoised_target vector is speaker mean vector
    spk_label, spk_index = np.unique(spk_label, return_inverse=True)
    i_vecs = np.array(i_vecs)
    targeted_vectors = np.zeros(i_vecs.shape)
    # calculating speaker mean i-vector
    for i, spk in enumerate(spk_label):
        idx = np.nonzero(spk_index == i)
        targeted_vectors[idx,:] = np.mean(i_vecs[idx], axis=0)
    i_vecs = length_norm(i_vecs)
    targeted_vectors = length_norm(targeted_vectors)
    return i_vecs, targeted_vectors


def load_test_info(filename):
    print('Loading test set information')
    tst_info = np.loadtxt(filename, dtype=np.str_, delimiter=',', skiprows=1, usecols=range(0, 3))
    tst_trials = []
    tst_trials_label = []
    for iter in range(len(tst_info)):
        tst_trials_label.extend([tst_info[iter, 0]])
        if tst_info[iter, 1] == 'background':
            tst_trials = np.append(tst_trials, 0)
        else:
            tst_trials = np.append(tst_trials, 1)
    return tst_trials, tst_trials_label


def create_dnn_model(features, hidden_layer_type, hidden_layer_size, n_out):
    assert len(hidden_layer_size) == len(hidden_layer_type)
    n_layers = len(hidden_layer_size)
    my_layers = list()
    for i in range(n_layers):
        if (hidden_layer_type[i] == 'TANH'):
            my_layers.append(C.layers.Dense(hidden_layer_size[i], activation=C.tanh, init=C.layers.glorot_uniform()))
        elif (hidden_layer_type[i] == 'RELU'):
            my_layers.append(C.layers.Dense(hidden_layer_size[i], activation=C.relu, init=C.layers.glorot_uniform()))
        elif (hidden_layer_type[i] == 'LSTM'):
            my_layers.append(C.layers.Recurrence(C.layers.LSTM(hidden_layer_size[i])))
        elif (hidden_layer_type[i] == 'BLSTM'):
            my_layers.append([C.layers.Recurrence(C.layers.LSTM(hidden_layer_size[i])),
                              C.layers.Recurrence(C.layers.LSTM(hidden_layer_size[i]), go_backwards=True),
                              C.splice])
        else:
            raise Exception('Unknown hidden layer type')

    my_layers.append(C.layers.Dense(n_out, activation=None))
    my_model = C.layers.Sequential([my_layers])
    my_model = my_model(features)
    return my_model




