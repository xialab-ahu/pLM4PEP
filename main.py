import sys
import numpy as np
import warnings
import os

import sklearn
import torch
from sklearn.metrics import confusion_matrix
from sklearn import metrics
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD as svd
import pickle
import math
from feature import *
from ML_grid_search_model import *

def readpeptides(posfile, negfile):  # 读取肽序列文件
    posdata = open(posfile, 'r')    #读取阳性样本文件
    pos = []
    for l in posdata.readlines():
        if l[0] == '>':
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    negdata = open(negfile, 'r')    #读取阴性样本文件
    neg = []
    for l in negdata.readlines():
        if l[0] == '>':
            continue
        else:
            neg.append(l.strip('\t0\n'))
    negdata.close()
    return pos, neg


def combinefeature(pep, featurelist, dataset):  #把featurelist里包含的feature堆叠起来
    a = np.empty([len(pep), 1]) #初始化特征矩阵，行数为训练集样本数，列数为1
    fname = []  #创建空列表保存特征名称
    scaling = StandardScaler()
    # pca = svd(n_components=300)
    vocab_name = []
    # print(a)
    if 'aap' in featurelist:
        aapdic = readAAP("./aap/aap_minmaxscaler_general.txt")  #读取AAP特征编码方法
        f_aap = np.array([aap(pep, aapdic, 1)]).T
        a = np.column_stack((a, f_aap))
        # a = scaling.fit_transform(a)
        fname.append('AAP')
        # print(f_aap)
    if 'aat' in featurelist:
        aatdic = readAAT("./aat/aat_minmaxscaler_general.txt")  #读取AAT特征编码方法
        f_aat = np.array([aat(pep, aatdic, 1)]).T
        a = np.column_stack((a, f_aat))
        # a = scaling.fit_transform(a)
        fname.append('AAT')
        # print(f_aat)
    if 'dpc' in featurelist:
        f_dpc, name = DPC(pep)  #使用函数DPC来提取特征
        # f_dpc = np.average(f_dpc, axis =1)
        a = np.column_stack((a, np.array(f_dpc)))
        # fname = fname + name
        fname = fname + ['dpc']*len(f_dpc[1])
    if 'aac' in featurelist:
        f_aac, name = AAC(pep)  #使用函数AAC来提取特征
        a = np.column_stack((a, np.array(f_aac)))
        # fname = fname + name
        fname = fname + ['aac']*len(f_aac[1])
    if 'paac' in featurelist:
        f_paac, name = PAAC(pep)    #使用函数PAAC来提取特征
        # f_paac = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_paac)))
        # fname = fname + name
        fname = fname + ['paac']*len(f_paac[1])
    if 'qso' in featurelist:
        f_qso, name = QSO(pep)  #使用函数QSO来提取特征
        # f_pa = pca.fit_transform(f_paac)
        a = np.column_stack((a, np.array(f_qso)))
        # fname = fname + name
        fname = fname + ['qso']*len(f_qso[1])

    if 'ctd' in featurelist:
        f_ctd, name = CTD(pep)  #使用函数CTD来提取特征
        a = np.column_stack((a, np.array(f_ctd)))
        # fname = fname + name
        fname = fname + ['ctd']*len(f_ctd[1])


    if 'bertfea' in featurelist:
        if len(pep) == 500:     #训练集
            f_bertfea = torch.load('./esm2/train_dataset_500.pt')
            #加载预训练好的esm2模型
            # b = sklearn.preprocessing.MinMaxScaler().fit_transform(f_bertfea)
            b = f_bertfea


        if len(pep) == 607:     #独立测试集1
            f_bertfea = torch.load(
                './esm2/independent_test_dataset1_607.pt')
            b = f_bertfea

        if len(pep) == 612:     #独立测试集2
            f_bertfea = torch.load(
                './esm2/independent_test_dataset2_612.pt')
            b = f_bertfea

        if len(pep) == 1018:       #独立测试集3
            f_bertfea = torch.load(
                './esm2/independent_test_dataset3_1018.pt')
            b = f_bertfea
        b = sklearn.preprocessing.MinMaxScaler().fit_transform(b)
        a = np.column_stack((a, b))
        fname = fname + ['bertfea'] * b.shape[1]
        print(b.shape)
        # sklearn.preprocessing.MinMaxScaler().fit_transform(f_bertfea)
        # b= sklearn.preprocessing.MinMaxScaler().fit_transform(f_bertfea)


    if 'GGAP' in featurelist:
        f_ggap = np.array(GGAP(pep))       #利用函数GGAP来提取特征
        # print(f_ggap.shape)
        a = np.column_stack((a, f_ggap))
        fname = fname + ['GGAP'] * f_ggap.shape[1]

    if 'ASDC' in featurelist:
        f_asdc = np.array(ASDC(pep))    #利用函数ASDC提取特征
        # print(f_asdc.shape)
        a = np.column_stack((a, f_asdc))
        fname = fname + ['ASDC'] * f_asdc.shape[1]

    if 'PSAAC' in featurelist:
        f_psaac = np.array(PSAAC(pep))  #利用函数PSAAC提取特征
        # print(f_psaac.shape)
        a = np.column_stack((a, f_psaac))
        fname = fname + ['PSAAC'] * f_psaac.shape[1]

    return a[:, 1:], fname, vocab_name


def run_training(pos, neg, pos1, neg1, pos2, neg2, pos3, neg3, dataset, feature_list):      #定义训练的函数，同时传入训练集和三个独立测试集的正负样本
    pep_combined = pos + neg    #训练集
    test1 = pos1 + neg1     #独立测试集1
    test2 = pos2 + neg2     #独立测试集2
    test3 = pos3 + neg3     #独立测试集3
    pickle_info={}      #保存模型
    #print(pep_combined)
    # aap aat dpc aac kmer protvec paac qso ctd
    featurelist = feature_list  #输入特征列表

    print(featurelist)  #打印特征列表
    pickle_info['featurelist'] = featurelist    #保存特征列表
    features, fname, vocab = combinefeature(pep_combined, featurelist, dataset) #处理训练集
    test1_features, __,__ = combinefeature(test1, featurelist, dataset) #处理独立测试集
    test2_features, __,__ = combinefeature(test2, featurelist, dataset)
    test3_features, __,__ = combinefeature(test3, featurelist, dataset)
    # torch.save(test1_features,'test1_features.pt')
    # torch.save(test2_features, 'test2_features.pt')
    # torch.save(test3_features, 'test3_features.pt')
    # torch.save(features,'features.pt')
    # torch.save(fname,'fname.pt')
    # torch.save(vocab,'vocab.pt')

    # features=torch.load('features.pt')
    # fname=torch.load('fname.pt')
    # vocab=torch.load('vocab.pt')
    # test1_features= torch.load('test1_features.pt')
    # test2_features= torch.load('test2_features.pt')
    # test3_features= torch.load('test3_features.pt')
    print(len(features[0]))
    target = [1] * len(pos) + [0] * len(neg)    #生成数据集标签
    target1 = [1] * len(pos1) + [0] * len(neg1)
    target2 = [1] * len(pos2) + [0] * len(neg2)
    target3 = [1] * len(pos3) + [0] * len(neg3)

    '''for i in range(len(features)):
    	print(features[i])'''
    find_all = lambda data, s: [r for r in range(len(data)) if data[r] == s]    #保存特征的index
    # data = 'afegjijefoeghhijowefnhudishujewnfuwifdnfe'
    # r_list = find_all(data, 'e')
    # print(r_list)

    # w = 0.3
    # p = 0.15
    w0 = 0.1  # 0.1
    p0 = 1  # 0.1-others without notation
    for w in range(0,1,1):
        w=w/10.0    #参数w在0到1内以步长0.1遍历，此处不遍历
        sortbestacc=[]
        sortacc=[]
        sortauroc=[]
        for p in range(1,2,1):
            p = 1     #最终模型里不降维
            filter_features = np.empty([len(pep_combined), 1])
            filter_features_test1 = np.empty([len(test1), 1])
            filter_features_test2 = np.empty([len(test2), 1])
            filter_features_test3 = np.empty([len(test3), 1])       #创建空矩阵，存放降维后的特征
            for key in featurelist:     #遍历特征列表里的特征
                fea_index=find_all(fname,key)       #从已经合并后的特征矩阵种找出所需特征对应的index
                train_features_bds = features[:,fea_index]      #根据index提取特征
                test1_features_bds = test1_features[:,fea_index]
                test2_features_bds = test2_features[:,fea_index]
                test3_features_bds = test3_features[:,fea_index]
                train_labels_bds = target

                # test_x['ath_independent_test'] = ath_independent_test_features[key]
                # test_x['fabaceae_independent_test'] = fabaceae_independent_test_features[key]
                # test_x['hybirdspecies_independent_test'] = hybirdspecies_independent_test_features[key]

                if key == 'bertfea':   #esm2在降维时使用参数w和p

                    filtered_feature_index = [1]
                    print('EFISS para w&p', w, p)
                else:       #其余特征在降维时使用参数w0和p0

                    filtered_feature_index = [1]

                if len(filtered_feature_index) != 0:
                    train_features_bds = train_features_bds
                    test1_features_bds = test1_features_bds
                    test2_features_bds = test2_features_bds
                    test3_features_bds = test3_features_bds
                    print(key, train_features_bds.shape)
                filter_features=np.column_stack((filter_features, train_features_bds))  #在各个数据集上把降维后的特征堆叠起来
                filter_features_test1 = np.column_stack((filter_features_test1, test1_features_bds))
                filter_features_test2 = np.column_stack((filter_features_test2, test2_features_bds))
                filter_features_test3 = np.column_stack((filter_features_test3, test3_features_bds))
            pickle_info['feat_name'] = fname
            pickle_info['vocab'] = vocab
            #print(features)

            #print(pep_combined)

            # train(pep_combined, features, target, pickle_info, dataset)
            alldata, best_acc, acc, auroc=train(pep_combined, filter_features[:, 1:], target, pickle_info, dataset)
            sortbestacc.append((p,best_acc))
            sortacc.append((p,acc))
            sortauroc.append((p,auroc))
            # with open('./model/svmshift-1-FRL_data-bert-esm.pickle', 'rb') as fin:
            #     alldata = pickle.load(fin)

            print('########################### predict ath_independent_test')
            predict_indep(filter_features_test1[:, 1:],target1,alldata)

            print('########################### predict fabaceae_independent_test')
            predict_indep(filter_features_test2[:, 1:],target2,alldata)

            print('########################### predict hybirdspecies_independent_test')
            predict_indep(filter_features_test3[:, 1:],target3,alldata)
        sortbestacc.sort(key=lambda x:x[1],reverse=True)    #按照第二个元素降序排序
        sortacc.sort(key=lambda x:x[1],reverse=True)
        sortauroc.sort(key=lambda x:x[1],reverse=True)

        print('Best P bestacc', sortbestacc[0],'in W=',w,'||acc',sortacc[0],'||auroc',sortauroc[0])

def train(peptides, features, target, pickle_info, dataset):    #定义训练模型
    # print(features.shape)
    scaling = StandardScaler()
    scaling.fit(features)   #学习如何对特征归一化
    print('max(features[:,0])', max(features[:,0]))
    x = scaling.transform(features)     #对特征进行归一化处理
    # x = features
    #print(max(x[:,1.txt]))
    y = np.array(target)
    cv = StratifiedKFold(n_splits=5)    #五折交叉验证

    # 切换模型
    # GBDT=gbdt_grid_search(x, y, cv)
    LR=lr_grid_search(x, y, cv)
    # ab=ab_grid_search(x, y, cv)
    # xgb1=xgb_grid_search(x, y, cv)
    # svm1=svm_grid_search(x, y, cv)  #参数寻优
    # rf=rf_grid_search(x, y, cv)
    # ert=ert_grid_search(x, y, cv)
    # knn1=knn_grid_search(x, y, cv)
    # ann1=ann_grid_search(x, y, cv)
    # all_model=[LR,ab,svm1,rf,ert,knn1,ann1]
    model=LR  #选取LR作为分类器
    # for model in all_model:
    # print(model)

    # for model in  all_model:
    # print(model)

    # aapdic = readAAP("./aap/aap_minmaxscaler_general.txt")
    # aatdic = readAAT("./aat/aat_minmaxscaler_general.txt")
    # pickle_info ['aap'] = aapdic
    # pickle_info ['aat'] = aatdic
    pickle_info ['scaling'] = scaling
    pickle_info ['model'] = model
    pickle_info ['training_features'] = features
    pickle_info ['training_targets'] = y
    # pickle.dump(pickle_info, open("./model/svmshift-1-"+dataset+"-bert-esm.pickle", "wb"))
    print("Best parameters: ", model.best_params_)
    print("Best accuracy: :", model.best_score_)

    cv_accracy = model.cv_results_['mean_test_ACC'][model.best_index_]      #模型最优参数下的五折交叉验证取平均
    cv_auprc = model.cv_results_['mean_test_AUPRC'][model.best_index_]
    cv_precision = model.cv_results_['mean_test_prec'][model.best_index_]
    cv_recall = model.cv_results_['mean_test_recall'][model.best_index_]
    cv_auroc = model.cv_results_['mean_test_AUROC'][model.best_index_]
    cv_f1 = model.cv_results_['mean_test_f1'][model.best_index_]

    y_train_t=y.tolist()
    y_train_t.count(1)
    y_train_t.count(0)
    TP1=y_train_t.count(1)*cv_recall
    FP1=(TP1/cv_precision)-TP1
    TN1=y_train_t.count(0)-FP1
    FN1=y_train_t.count(1)-TP1
    print('TP:',TP1,',TN:',TN1,',FP:',FP1,',FN:',FN1)
    cv_specificity = Specificity=TN1/(TN1+FP1)

    if ((float(TP1 + FP1) * float(TN1 + FN1)) != 0):
        cv_MCC = float(TP1*TN1-FP1*FN1)/ math.sqrt(float(TP1 + FP1) * float(TP1 + FN1) * float(TN1 + FP1) * float(TN1 + FN1))
        print('Specificity_train:',cv_specificity,',ACC_train:',cv_accracy,',Precision_train:',cv_precision,',Recall_train:',cv_recall,',F1Score_train:',cv_f1,',MCC_train:',cv_MCC,',auprc_train:',cv_auprc,',auroc_train:',cv_auroc)
    else:
        print('Specificity_train,ACC_train,Precision_train,Recall_train,F1Score_train,auprc_train,auroc_train:',
              cv_specificity,cv_accracy,cv_precision,cv_recall,cv_f1,cv_auprc,cv_auroc)
    return pickle_info, model.best_score_, cv_accracy, cv_auroc


def predict_indep(features, target, alldata):
    # print(features.shape)
    # print(alldata.keys())
    model1 = alldata['model']   #读取之前保存的模型
    f_scaling = alldata['scaling']      #读取之前保存的归一化方法
    #f_scaling.fit(features)
    x_test = f_scaling.transform(features)
    y = np.array(target)
    y_pred = model1.predict(x_test)
    #y_scores = model1.decision_function(x_test)
    #print(y_scores)
    y_scores = model1.predict_proba(x_test)[:, 1]

    TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
    print('TN, FP, FN, TP:', TN, FP, FN, TP)

    Specificity = TN / (TN + FP)        #计算各类指标
    ACC = float(TP + TN) / float(TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1Score = 2 * TP / (2 * TP + FP + FN)
    #MCC = float(TP * TN - FP * FN) / math.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))

    p, r, thresh = metrics.precision_recall_curve(y, y_scores)
    pr_auc = metrics.auc(r, p)

    ro_auc = metrics.roc_auc_score(y, y_scores)

    #print('Specificity:', Specificity, 'ACC:', ACC, 'Precision:', Precision, 'Recall:', Recall,
          #'F1Score:', F1Score, 'MCC:', MCC, 'auprc:', pr_auc, 'auroc:', ro_auc)
    print('Specificity:', Specificity, 'ACC:', ACC, 'Precision:', Precision, 'Recall:', Recall,
           'F1Score:', F1Score, 'auprc:', pr_auc, 'auroc:', ro_auc)

if __name__ == "__main__":
    # dataset = sys.argv[1]
    dataset = 'FRL_data'
    pos, neg = readpeptides("./datasets/"+dataset+"/train_pos.txt",
                            "./datasets/"+dataset+"/train_neg.txt")
    #print(pos, neg)
    # all_feature=['ISAAC','ASDC','GGAP','bertfea','ctd','qso','paac','aac','dpc']
    # all_feature=['PSAAC','ASDC','bertfea','qso','paac','aac','dpc']
    all_feature=['bertfea']

    # all_feature=['bertfea']
    pos1, neg1 = readpeptides("./datasets/"+dataset+"/test_pos.txt",
                            "./datasets/"+dataset+"/test_neg.txt")
    print('########################### ath_independent_test loaded')

    pos2, neg2 = readpeptides("./datasets/"+dataset+"/test2_pos.txt",
                            "./datasets/"+dataset+"/test2_neg.txt")
    print('########################### fabaceae_independent_test loaded')

    pos3, neg3 = readpeptides("./datasets/"+dataset+"/test3_pos.txt",
                            "./datasets/"+dataset+"/test3_neg.txt")
    print('########################### hybirdspecies_independent_test loaded')

    run_training(pos, neg, pos1, neg1, pos2, neg2, pos3, neg3, dataset, all_feature)


    # for fea in all_feature:
    #     for one_model in all_model:
    #         print(one_model)
    #         run_training(pos, neg, dataset, fea, one_model)
    # for fea in all_feature:
    #     run_training(pos, neg, dataset, fea)
