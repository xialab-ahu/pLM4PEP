import torch
import numpy as np
# a=torch.load('examples/data/some_proteins_emb_esm2/independent_test_dataset3_1018/ath-identified-miPEP165a_positive_test.pt')
a=torch.load('examples/data/some_proteins_emb_esm2/train_dataset_500_pertok.pt')
b=np.array(a)
torch.load('examples/data/some_proteins_emb_esm2/independent_test_dataset3_1018/ath-identified-miPEP165a_positive_test.pt')