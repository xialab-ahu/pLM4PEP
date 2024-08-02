import sys
import numpy as np
import warnings
import os

import sklearn
import torch

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from evaluate import load
def readpeptides(posfile, negfile):  # return the peptides from input peptide list file
    posdata = open(posfile, 'r')
    pos = []
    for l in posdata.readlines():
        if l[0] == '>':
            continue
        else:
            pos.append(l.strip('\t0\n'))
    posdata.close()
    negdata = open(negfile, 'r')
    neg = []
    for l in negdata.readlines():
        if l[0] == '>':
            continue
        else:
            neg.append(l.strip('\t0\n'))
    negdata.close()
    return pos, neg

model_checkpoint = "facebook/esm2_t12_35M_UR50D"

dataset = 'FRL_data'
pos, neg = readpeptides("../NeuroPpred-SVM-main/datasets/" + dataset + "/train_pos.txt",
                        "../NeuroPpred-SVM-main/datasets/" + dataset + "/train_neg.txt")



pos1, neg1 = readpeptides("../NeuroPpred-SVM-main/datasets/" + dataset + "/test_pos.txt",
                        "../NeuroPpred-SVM-main/datasets/" + dataset + "/test_neg.txt")
print('########################### ath_independent_test')

pos2, neg2 = readpeptides("../NeuroPpred-SVM-main/datasets/" + dataset + "/test2_pos.txt",
                        "../NeuroPpred-SVM-main/datasets/" + dataset + "/test2_neg.txt")
print('########################### fabaceae_independent_test')

pos3, neg3 = readpeptides("../NeuroPpred-SVM-main/datasets/" + dataset + "/test3_pos.txt",
                        "../NeuroPpred-SVM-main/datasets/" + dataset + "/test3_neg.txt")
print('########################### hybirdspecies_independent_test')


allpos=pos+pos1+pos2+pos3
allneg=neg+neg1+neg2+neg3
pep_combined = allpos + allneg
target = [1] * len(allpos) + [0] * len(allneg)
train_sequences, test_sequences, train_labels, test_labels = train_test_split(pep_combined, target, test_size=0.2, shuffle=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokenizer(train_sequences[0]))

train_tokenized = tokenizer(train_sequences)
test_tokenized = tokenizer(test_sequences)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)
print('train_dataset',train_dataset)

num_labels = max(train_labels + test_labels) + 1  # Add 1 since 0 can be a label
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model_name = model_checkpoint.split("/")[-1]
batch_size = 8

args = TrainingArguments(
    f"{model_name}-finetuned-alldata-localization",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
