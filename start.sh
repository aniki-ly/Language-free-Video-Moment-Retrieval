CUDA_VISIBLE_DEVICES=$1 python train.py --model CrossModalityTwostageAttention --config configs/cha_simple_model/simplemodel_cha_BS256_two-stage_attention.yml -w $2
