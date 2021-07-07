main=/home/cse/btech/cs1180393/scratch
data=$main/fairness/DATA 
outdir=$main/models/strong_baselines/data_point

python train.py --dataset COCOStuff --nclasses 171 --model data_point --outdir $outdir \
  --labels_train $data/labels_train_80.pkl --labels_val $data/labels_train_20.pkl \
  --nepoch 50 --lr 0.1 --drop 60 --wd 0 --momentum 0.9 \
      --biased_classes_mapped $data/biased_classes_mapped.pkl  \
    --unbiased_classes_mapped $data/unbiased_classes_mapped.pkl \
    --humanlabels_to_onehot $data/humanlabels_to_onehot.pkl \
    --lambda_both 1 --lambda_b 1 --lambda_c 1 --train_batchsize 200
