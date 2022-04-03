env LC_ALL=en_US.UTF-8 
HOME=/home/tbui
EXPDIR=$PWD

SCRIPTS=${HOME}/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
TRUECASER_TRAIN=$SCRIPTS/recaser/train-truecaser.perl
TRUECASER=$SCRIPTS/recaser/truecase.perl
BPEROOT=$HOME/subword-nmt/subword_nmt
BPE_TOKENS=7000

DATASET=$PWD/data
DATASET_NAME="train valid test"
NORMALIZED_DATA=$DATASET/tmp/normalized
TOKENIZED_DATA=$DATASET/tmp/tok
TRUECASED_DATA=$DATASET/tmp/truecased
BPE_DATA=$DATASET/tmp/bpe-data
BIN_DATA=$DATASET/tmp/bin-data

# Making directories
if [ ! -d $DATASET/tmp ]; then
    mkdir -p $DATASET/tmp
fi

if [ ! -d $NORMALIZED_DATA ]; then
    mkdir -p $NORMALIZED_DATA
fi

if [ ! -d $TOKENIZED_DATA ]; then
    mkdir -p $TOKENIZED_DATA
fi

if [ ! -d $TRUECASED_DATA ]; then
    mkdir -p $TRUECASED_DATA
fi

if [ ! -d $BPE_DATA ]; then
    mkdir -p $BPE_DATA
fi

if [ ! -d $BIN_DATA ]; then
    mkdir -p $BIN_DATA
fi

src=en
tgt=vi

# Normalization
echo "=> Normalizing...."
for lang in $src $tgt; do 
    for set in $DATASET_NAME; do
        python3.6 ${EXPDIR}/preprocess/normalize.py ${DATA}/${set}.${lang} \
                                        ${NORMALIZED_DATA}/${set}.${lang}
    done
done

# Tokenization
echo "=> Tokenizing...."
for SET in $DATASET_NAME; do
    $TOKENIZER -l en < ${NORMALIZED_DATA}/${SET}.en > ${TOKENIZED_DATA}/${SET}.en
    python3.6 {EXPDIR}/preprocess/tokenize-vi.py ${NORMALIZED_DATA}/${SET}.vi ${TOKENIZED_DATA}/${SET}.vi
done

# Truecaser
echo "=>  Truecasing...."
echo "Traning for english..."
$TRUECASER_TRAIN --model truecase-model.en --corpus ${TOKENIZED_DATA}/train.en

echo "Traning for vietnamese..."
$TRUECASER_TRAIN --model truecase-model.vi --corpus ${TOKENIZED_DATA}/train.vi

for lang in $src $tgt; do
    for set in $DATA_NAME; do
        $TRUECASER --model truecase-model.${lang} < ${TOKENIZED_DATA}/${set}.${lang} > ${TRUECASED_DATA}/${set}.${lang}
    done
done

cat ${TOKENIZED_DATA}/train.en ${TOKENIZED_DATA}/train.vi > $DATASET/tmp/train.en-vi

# learn bpe model with training data
echo "=> LEARNING BPE MODEL...."

subword-nmt learn-bpe -s ${BPE_TOKENS} < $DATASET/tmp/train.en-vi > $DATASET/tmp/en-vi.bpe.${BPE_TOKENS}.model

for SET in $DATASET_NAME; do
    for lang in $src $tgt; do
        subword-nmt apply-bpe -c $DATASET/tmp/en-vi.bpe.${BPE_TOKENS}.model < ${TOKENIZED_DATA}/${SET}.${lang} > $BPE_DATA/${SET}.${lang}
    done
done

fairseq-preprocess -s $src -t $tgt \
			--destdir $BIN_DATA \
			--trainpref $BPE_DATA/train \
			--validpref $BPE_DATA/valid \
			--testpref $BPE_DATA/test \
			--workers 32 \
            2>&1 | tee $EXPDIR/logs/preprocess