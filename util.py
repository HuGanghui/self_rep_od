import pandas as pd
import numpy as np
import random
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score

import time
import datetime

mem = Memory("./dataset/svm_data")

@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def dataLoading(path, logfile=None):

    # loading data
    df = pd.read_csv(path)
    labels = df['class']
    x_df = df.drop(['class'], axis=1)
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    if logfile:
        logfile.write("Data shape: (%d, %d)\n" % x.shape)

    return x, labels


# random sampling with replacement
def random_list(start, stop, length):
    if length >= 0:
        length = int(length)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))  # including start and stop
    return random_list

def get_best_f1_score(fpr, tpr, threshold):
    f1s = 2 * fpr * tpr / (fpr + tpr)
    max_args = np.argmax(f1s)
    return f1s[max_args], fpr[max_args], tpr[max_args], threshold[max_args]

def get_range_proba(predict, label, delay=7):
    # TODO 这里有个小bug，对于最后一段异常没有进行魔改，可以通过在结尾加上一个标签为0的点来解决这个问题
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

def aucPerformance(scores, labels, logfile=None, criterion=None):
    roc_auc = roc_auc_score(labels, scores)
#    print(roc_auc)
    ap = average_precision_score(labels, scores)
    fpr, tpr, thre = precision_recall_curve(labels, scores)
    maxs, argmax_fpr, argmax_tpr, argmax_thre = get_best_f1_score(fpr, tpr, thre)
    scores[scores >= argmax_thre] = int(1)
    scores[scores < argmax_thre] = int(0)
    scores.astype(int)
    delay = 7
    proba = get_range_proba(scores, np.asarray(labels), delay)
    f1score = f1_score(labels, proba)
    start_time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    if criterion is None:
        print(start_time)
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
        if logfile:
            logfile.write(start_time)
            logfile.write("AUC-ROC: %.4f, AUC-PR: %.4f\n" % (roc_auc, ap))
            logfile.write("best f1_score: %.4f, fpr: %.4f, tpr: %.4f, thre: %.4f \n" %
               (maxs, argmax_fpr, argmax_tpr, argmax_thre))
            logfile.write("best f1_score: %f (pei_dan's)\n" % (f1score))

    else:
        print(start_time)
        print(criterion + ": AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
        if logfile:
            logfile.write(start_time)
            logfile.write(criterion + ": AUC-ROC: %.4f, AUC-PR: %.4f\n" % (roc_auc, ap))        
            logfile.write("best f1_score: %.4f, fpr: %.4f, tpr: %.4f, thre: %.4f \n" %
               (maxs, argmax_fpr, argmax_tpr, argmax_thre))
            logfile.write("best f1_score: %f (pei_dan's)\n" % (f1score))

#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1],[0,1],'r--')
#    plt.xlim([-0.001, 1])
#    plt.ylim([0, 1.001])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show();

    return roc_auc, ap


def tic_time():
    print("=====================================================")
    tic_datetime = datetime.datetime.now()
    print("tic_datetime:", tic_datetime)
    print("tic_datetime.strftime:", tic_datetime.strftime('%Y-%m-%d %H:%M:%S.%f'))
    tic_walltime = time.time()
    print("tic_walltime:", tic_walltime)
    # tic_cpu = time.clock()
    # print("tic_cpu:", tic_cpu)
    print("=====================================================\n")
