src=en
tgt=vi

GPUS=$1
MODEL_NAME=$2
EPOCHS=$3

EXPDIR=$PWD
DATASET=$PWD/data
BIN_DATA=$DATASET/tmp/bin-data

CUDA_VISIBLE_DEVICES=$GPUS fairseq-train $BIN_DATA -s ${SRC} -t ${TGT} \
		            --log-interval 100 \
					--log-format json \
					--max-epoch ${EPOCHS} \
		    		--optimizer adam --lr 0.0001 \
					--clip-norm 0.0 \
					--max-tokens 4000 \
					--no-progress-bar \
					--log-interval 100 \
					--min-lr '1e-09' \
					--weight-decay 0.0001 \
					--criterion label_smoothed_cross_entropy \
					--label-smoothing 0.1 \
					--lr-scheduler inverse_sqrt \
					--warmup-updates 4000 \
					--warmup-init-lr '1e-08' \
					--adam-betas '(0.9, 0.98)' \
					--arch transformer_iwslt_de_en \
					--dropout 0.1 \
					--attention-dropout 0.1 \
					--save-dir $EXPDIR/models/$MODEL_NAME \
					2>&1 | tee $LOG/${MODEL_NAME}

