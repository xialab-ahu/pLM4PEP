python extract.py esm2_t12_35M_UR50D.pt examples/data/some_proteins.fasta examples/data/some_proteins_emb_esm2 --include mean per_tok

python extract.py esm2_t12_35M_UR50D.pt examples/data/independent_test_dataset3_1018.txt examples/data/some_proteins_emb_esm2/independent_test_dataset3_1018 --include mean per_tok

python extract.py esm2_t12_35M_UR50D.pt examples/data/independent_test_dataset2_612.txt examples/data/some_proteins_emb_esm2/independent_test_dataset2_612 --include mean per_tok

python extract.py esm2_t12_35M_UR50D.pt examples/data/independent_test_dataset1_607.txt examples/data/some_proteins_emb_esm2/independent_test_dataset1_607 --include mean per_tok
python extract.py esm2_t12_35M_UR50D.pt examples/data/independent_test_dataset1_607_shift.txt examples/data/some_proteins_emb_esm2/independent_test_dataset1_607_shift --include mean per_tok

python extract.py esm2_t12_35M_UR50D.pt examples/data/train_dataset_500.txt examples/data/some_proteins_emb_esm2/train_dataset_500 --include mean
python extract.py esm2_t12_35M_UR50D.pt examples/data/train_dataset_500_shift.txt examples/data/some_proteins_emb_esm2/train_dataset_500_shift --include mean per_tok
12层 transformer 35M 参数
#finetuning
python extract.py esm2_t12_35M_UR50D-finetuned-alldata-localization\checkpoint-822\pytorch_model.bin examples/data/train_dataset_500.txt examples/data/some_proteins_emb_esm2/train_dataset_500 --include mean
python extract.py pytorch_model_finetuning_replace.pt examples/data/train_dataset_500.txt examples/data/some_proteins_emb_esm2/train_dataset_500 --include mean

esm2_t12_35M_UR50D-finetuned-alldata-localization\checkpoint-822\pytorch_model.bin


t6_8M
python extract.py esm2_t12_35M_UR50D.pt examples/data/train_dataset_500.txt examples/data/some_proteins_emb_esm2/train_dataset_500_pertok --include per_tok
python extract.py esm2_t6_8M_UR50D.pt examples/data/train_dataset_500_shift.txt examples/data/some_proteins_emb_esm2/train_dataset_500_shift_t6_8M --include mean
python extract.py esm2_t6_8M_UR50D.pt examples/data/independent_test_dataset1_607_shift.txt examples/data/some_proteins_emb_esm2/independent_test_dataset1_607_shift_t6_8M --include mean
python extract.py esm2_t6_8M_UR50D.pt examples/data/independent_test_dataset2_612.txt examples/data/some_proteins_emb_esm2/independent_test_dataset2_612_t6_8M --include mean
python extract.py esm2_t6_8M_UR50D.pt examples/data/independent_test_dataset3_1018.txt examples/data/some_proteins_emb_esm2/independent_test_dataset3_1018_t6_8M --include mean
python extract.py esm2_t6_8M_UR50D.pt examples/data/train_dataset_500.txt examples/data/some_proteins_emb_esm2/train_dataset_500_t6_8M --include mean
python extract.py esm2_t6_8M_UR50D.pt examples/data/independent_test_dataset1_607.txt examples/data/some_proteins_emb_esm2/independent_test_dataset1_607_t6_8M --include mean

PredNeuroP
python extract.py esm2_t12_35M_UR50D.pt examples/data/PredNeuroP/train_all.txt examples/data/PredNeuroP/train_all --include mean
python extract.py esm2_t12_35M_UR50D.pt examples/data/PredNeuroP/test_all.txt examples/data/PredNeuroP/test_all --include mean

PredAPP
python extract.py esm2_t12_35M_UR50D.pt examples/data/PredAPP/Training.fasta examples/data/PredAPP/train_all --include mean
python extract.py esm2_t12_35M_UR50D.pt examples/data/PredAPP/Test.fasta examples/data/PredAPP/test_all --include mean

PLM4AC
python extract.py esm2_t12_35M_UR50D.pt examples/data/PLM4AC/all_esm2.txt examples/data/PLM4AC/data_all --include mean


BBPpred
python extract.py esm2_t12_35M_UR50D.pt examples/data/BBPpred/Training_format.fasta examples/data/BBPpred/train_all --include mean
python extract.py esm2_t12_35M_UR50D.pt examples/data/BBPpred/Test_format.fasta examples/data/BBPpred/test_all --include mean


FRL-train+test2+test3
python extract.py esm2_t12_35M_UR50D.pt examples/data/train_test1_test3.txt examples/data/some_proteins_emb_esm2/train_test1_test3 --include mean





