import torch

PATH='esm2_t12_35M_UR50D-finetuned-alldata-localization\checkpoint-822\pytorch_model.bin'
model=torch.load(PATH)
PATH2='esm2_t12_35M_UR50D.pt'
model2=torch.load(PATH2)
model2['model']=model

torch.save(model2,"pytorch_model_finetuning_replace.pt")
# torch.save(model,"pytorch_model.pt")