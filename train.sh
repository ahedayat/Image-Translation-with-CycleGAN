# Hardware
num_workers=3

# CNN BackBone
num_residuals=9

# Input Size
image_size=128

# Data Path
data_df="./datasets/Horse2Zebra/preprocessed.csv"
# - Train
train_base_dir="./datasets/Horse2Zebra"
# - Validation
val_base_dir="./datasets/Horse2Zebra"
# - Test
test_base_dir="./datasets/Horse2Zebra"

# Loss Function
lambda_cycle=10
lambda_identity=5

# Optimizer and its hyper-parameters
learning_rate=2e-4
b1=0.5
b2=0.999
# Learning Rate Scheduler
lr_start_decay=10

# Training Hyper-Parameter
epoch=20
batch_size=1

# Saving Paths
# Generated Images
generated_images="./generated_images"
sample_interval=10

# Reports Path
report_path="./reports"

# Checkpoints
ckpt_path="./checkpoints"
ckpt_prefix="ckpt"
model_saving_freq=50

python train.py \
        --num-workers $num_workers \
        --gpu \
        --num-residuals $num_residuals \
        --image-size $image_size \
        --data-df $data_df \
        --train-base-dir $train_base_dir \
        --val-base-dir $val_base_dir \
        --test-base-dir $test_base_dir \
        --lambda-cycle  $lambda_cycle \
        --lambda-identity  $lambda_identity \
        --learning-rate $learning_rate \
        --b1 $b1 \
        --b2 $b2 \
        --lr-start-decay $lr_start_decay \
        --epoch $epoch \
        --batch-size $batch_size \
        --generated-images $generated_images \
        --report-path $report_path \
        --sample-interval $sample_interval \
        --ckpt-path $ckpt_path \
        --ckpt-prefix $ckpt_prefix \
        --model-saving-freq $model_saving_freq
