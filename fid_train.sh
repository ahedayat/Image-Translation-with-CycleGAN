# Hardware
num_workers=2

# CNN BackBone
num_residuals=9

# Input Size
image_size=128

# Data Path
data_df="./datasets/Horse2Zebra/preprocessed.csv"
# - Test
val_base_dir="./datasets/Horse2Zebra"

# Training Hyper-Parameter
batch_size=8

# Checkpoints
gen_x2y="./checkpoints"
gen_y2x="./checkpoints"

python fid_train.py \
        --num-workers $num_workers \
        --gpu \
        --num-residuals $num_residuals \
        --image-size $image_size \
        --data-df $data_df \
        --val-base-dir $val_base_dir \
        --batch-size $batch_size \
        --gen-x2y $gen_x2y \
        --gen-y2x $gen_y2x
