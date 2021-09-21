main=/home/cse/btech/cs1180393/scratch
data=$main/fairness/DATA 
outdir=$main/Proposals/
modelpath=/home/cse/btech/cs1180393/scratch/models/standard/model_62.pth

python train.py --dataset COCOStuff --nclasses 171 --model layer_cam --outdir $outdir \
  --labels_train $data/labels_train_80.pkl --labels_val $data/labels_train_20.pkl \
  --nepoch 10 --lr 0.1 --drop 60 --wd 0 --momentum 0.9 \
      --biased_classes_mapped $data/biased_classes_mapped.pkl  \
    --unbiased_classes_mapped $data/unbiased_classes_mapped.pkl \
    --humanlabels_to_onehot $data/humanlabels_to_onehot.pkl \
    --modelpath $modelpath --train_batchsize 100 --cam_lambda2 $1 \
    --pretrainedpath $modelpath 
