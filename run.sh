# # !/usr/bin/env sh
# the above line is the shebang. It must be the first line, or will be interpreted as a comment
echo "Executing script: $0"
# to make the script executable, run: chmod 755 script.sh; then run it as ./script.sh


# Flag for CUDA_VISIBLE_DEVICE:
CUDA_VD=$1 # feed this parameter when running the script

DPATH='../DATA/ACDC'
RES_DIR='.'
DSET_NAME='acdc'
TABLE_NAME='Experiments'
DB_NAME="test_results.db"
EPOCHS=1200

EXP_TYPE='semi'
MODALITY=''

N_T_STEPS=-1


for PERC in 'perc25' '1p5T_to_3T'
    do for SPLIT in 'split0'
        do

        # ---------------------------------------------------------------------------------------------------------
        WARM_UP_TIME=100
        # WARM_UP_TIME=0

        # ----------------
        RUN_ID_AND_PATH='AdversarialTTT exp_gan_ttt'
        # The first part of the variable (i.e.: AdversarialTTT) is an ID for the experiment, while the second part
        # (i.e.: exp_gan_ttt) is the name of the file that will be run for the experiment

        # shellcheck disable=SC2086
        set -- ${RUN_ID_AND_PATH}
        RUN_ID=$1
        PATH=$2

        R_ID="${RUN_ID}"_${PERC}_${SPLIT}
        echo "${R_ID}"
        python -m train \
               --RUN_ID="${R_ID}" --sql_db_name="${DB_NAME}" \
               --n_epochs=${EPOCHS} --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${DPATH} \
               --experiment="${PATH}" --warm_up_period="${WARM_UP_TIME}"\
               --dataset_name=${DSET_NAME} --notify=n --verbose=y --n_sup_vols=${PERC} \
               --split_number=${SPLIT} --table_name=${TABLE_NAME} --results_dir=${RES_DIR} \
               --normalization_type='BN' --gan='lsgan' --ttt_steps=${N_T_STEPS} --modality=${MODALITY} \
               --validation_offset=${WARM_UP_TIME} \
               --instance_noise=n \
               --label_flipping=n \
               --one_sided_label_smoothing=n \
               --use_fake_anchors=n \
               --do_test_augmentation=y

        # ----------------
        N_T_STEPS_MAX=2000
        RUN_ID_AND_PATH='AdversarialTTT&&AdversarialTTT exp_post_ttt'
        # Notice the '&&' to define the run-id for segmentor (left) and discriminator (right): in this case, we use the
        # same checkpoint

        # shellcheck disable=SC2086
        set -- ${RUN_ID_AND_PATH}
        RUN_ID=$1
        PATH=$2

        R_ID="${RUN_ID}"_${PERC}_${SPLIT}
        echo "${R_ID}"
        python -m test \
               --RUN_ID="${R_ID}" --sql_db_name="${DB_NAME}" \
               --n_epochs=${EPOCHS} --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${DPATH} \
               --experiment="${PATH}" --warm_up_period="${WARM_UP_TIME}"\
               --dataset_name=${DSET_NAME} --notify=n --verbose=y --n_sup_vols=${PERC} \
               --split_number=${SPLIT} --table_name=${TABLE_NAME} --results_dir=${RES_DIR} \
               --normalization_type='BN' --gan='lsgan' --ttt_steps=${N_T_STEPS_MAX} --modality=${MODALITY} \
               --validation_offset=${WARM_UP_TIME} \
               --instance_noise=n \
               --label_flipping=n \
               --one_sided_label_smoothing=n \
               --use_fake_anchors=n \
               --do_test_augmentation=y # --visuals_only=y



        # ----------------
        N_T_STEPS_MAX=2000
        RUN_ID_AND_PATH='AdversarialTTT&&AdversarialTTT exp_post_ttt_continual'
        # Notice the '&&' to define the run-id for segmentor (left) and discriminator (right): in this case, we use the
        # same checkpoint

        # shellcheck disable=SC2086
        set -- ${RUN_ID_AND_PATH}
        RUN_ID=$1
        PATH=$2

        R_ID="${RUN_ID}"_${PERC}_${SPLIT}
        echo "${R_ID}"
        python -m test \
               --RUN_ID="${R_ID}" --sql_db_name="${DB_NAME}" \
               --n_epochs=${EPOCHS} --CUDA_VISIBLE_DEVICE=${CUDA_VD} --data_path=${DPATH} \
               --experiment_type=${EXP_TYPE} --experiment="${PATH}" --warm_up_period="${WARM_UP_TIME}"\
               --dataset_name=${DSET_NAME} --notify=n --verbose=y --n_sup_vols=${PERC} \
               --split_number=${SPLIT} --table_name=${TABLE_NAME} --results_dir=${RES_DIR} \
               --normalization_type='BN' --gan='lsgan' --ttt_steps=${N_T_STEPS_MAX} --modality=${MODALITY} \
               --validation_offset=${WARM_UP_TIME} \
               --instance_noise=n \
               --label_flipping=n \
               --one_sided_label_smoothing=n \
               --use_fake_anchors=n \
               --do_test_augmentation=y # --visuals_only=y
    done
done

python -m notify --message="Train finished on GPU ${CUDA_VD} (SEMI on ACDC)."

printf "\nScript executed with %s errors.\n" "$?"  # > 0 means there is an error
#if [ condition ]
#then command
#fi