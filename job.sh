#!/bin/bash
#SBATCH -p batch_ce_ugrad 
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=25G
#SBATCH --time=4-00:00:0
#SBATCH --out %j.out
#SBATCH --err %j.err
#SBATCH --gres=gpu:8


OUTDIR=/data/jong980812/project/cil/CODA-Prompt/result/mini-ssv2/144_10task_l2p_0.3
# bash experiments/cifar-100.sh
# experiment settings


# save directory

# hard coded inputs
GPUID='0 1 2 3 4 5 6 7'
CONFIG=configs/ssv2_prompt.yaml
# CONFIG=configs/ucf-101_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################
# process inputs
mkdir -p $OUTDIR

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P_video \
#     --prompt_param 50 8 -1 \
#     --log_dir ${OUTDIR} \
#     --anno_path /data/jong980812/project/cil/CODA-Prompt/anno_list/SSV2/ssv2_data_tasks_109_2.pkl
    # --anno_path /data/jong980812/project/cil/CODA-Prompt/anno_list/UCF101/20.pkl

# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
# #    arg 3 = g-prompt pool length
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 20 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt


# # CODA-P
# #
# # prompt parameter args:
# #    arg 1 = prompt component pool size
# #    arg 2 = prompt length
# # #    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODA_video \
    --prompt_param 200 8 0.0 \
    --log_dir ${OUTDIR} \
    --anno_path /data/jong980812/project/cil/CODA-Prompt/anno_list/SSV2/ssv2_data_tasks_109_2_0.1.pkl
    # --anno_path /data/jong980812/project/cil/CODA-Prompt/anno_list/UCF101/20.pkl
