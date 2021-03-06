HOME=/home/tbui
EXPDIR=$PWD

SCRIPTS=${HOME}/mosesdecoder/scripts
DETRUECASER=${SCRIPTS}/recaser/detruecase.perl

src=$1
tgt=$2
GPUS=$3
MODEL_NAME=$4
MODEL=$PWD/models/${MODEL_NAME}/checkpoint_best.pt

DATASET=$PWD/data
BPE_DATA=$DATASET/tmp/bpe-data
BIN_DATA=$DATASET/tmp/bin-data


########################## Validation dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $BPE_DATA/valid.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/valid_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/valid_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/valid_trans.${tgt}
cat ${PWD}/results/${MODEL_NAME}/valid_trans.${tgt} | sed -r 's/(@@ )|(@@ ?$)//g' > ${PWD}/results/${MODEL_NAME}/valid_rmvbpe.${tgt}

# detruecase
$DETRUECASER < ${PWD}/results/${MODEL_NAME}/valid_rmvbpe.${tgt} > ${PWD}/results/${MODEL_NAME}/valid_detruecase.${tgt}

# detokenize
python3.6 $PWD/postprocess/detokenize.py ${PWD}/results/${MODEL_NAME}/valid_detruecase.${tgt} ${PWD}/results/${MODEL_NAME}/valid.${tgt}
python3.6 $PWD/preprocess/normalize.py ${PWD}/results/${MODEL_NAME}/valid.${tgt} ${PWD}/results/${MODEL_NAME}/valid_normalize.${tgt}
echo "VALID" >> ${PWD}/results/${MODEL_NAME}/valid_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/normalized/valid.${tgt} < ${PWD}/results/${MODEL_NAME}/valid_normalize.${tgt} >> ${PWD}/results/${MODEL_NAME}/valid_result.txt

########################## Test dataset #########################################

CUDA_VISIBLE_DEVICES=$GPUS env LC_ALL=en_US.UTF-8 fairseq-interactive $BIN_DATA \
            --input $BPE_DATA/test.${src} \
            --path $MODEL \
            --beam 5 | tee ${PWD}/results/${MODEL_NAME}/test_trans_result.${tgt}

grep ^H ${PWD}/results/${MODEL_NAME}/test_trans_result.${tgt} | cut -f3 > ${PWD}/results/${MODEL_NAME}/test_trans.${tgt}
cat ${PWD}/results/${MODEL_NAME}/test_trans.${tgt} | sed -r 's/(@@ )|(@@ ?$)//g' > ${PWD}/results/${MODEL_NAME}/test_rmvbpe.${tgt}

# detruecase
$DETRUECASER < ${PWD}/results/${MODEL_NAME}/test_rmvbpe.${tgt} > ${PWD}/results/${MODEL_NAME}/test_detruecase.${tgt}

# detokenize
python3.6 $PWD/postprocess/detokenize.py ${PWD}/results/${MODEL_NAME}/test_detruecase.${tgt} ${PWD}/results/${MODEL_NAME}/test.${tgt}
python3.6 $PWD/preprocess/normalize.py ${PWD}/results/${MODEL_NAME}/test.${tgt} ${PWD}/results/${MODEL_NAME}/test_normalize.${tgt}
echo "TEST" >> ${PWD}/results/${MODEL_NAME}/test_result.txt
env LC_ALL=en_US.UTF-8 perl $PWD/multi-bleu.pl $PWD/data/tmp/normalized/test.${tgt} < ${PWD}/results/${MODEL_NAME}/test_normalize.${tgt} >> ${PWD}/results/${MODEL_NAME}/test_result.txt

