
PYTHON_MODULES_PATH=../../src
export PYTHONPATH=$PYTHON_MODULES_PATH

# /work/train_data_folder/
SELECT_FROM_FOLDER="$1"
# /work/train_data_to_review/
OUTPUT_TO_FOLDER_TO_REVIEW="$2"
# 20
K_UNSURE="$3"
MODEL_FOLDER_PATH="$4"

# 1. PREDICT USING MODEL
python ../models/predict_model.py \
-i $SELECT_FROM_FOLDER \
--available_gpus 1 \
--model_file_path /content/drive/MyDrive/vork/ML/trees/tree-counting-and-classification-in-images/models/model1/deepforest_p5933_r6815 \
--write_csvs --write_xmls
# [--model_folder_path $MODEL_FOLDER_PATH]

# 2. SELECT DATA TO BE SENT TO THE ORACLE (=labeller)
python ../data/data_selection.py \
    --input_folder $SELECT_FROM_FOLDER \
    --output_folder $OUTPUT_TO_FOLDER_TO_REVIEW \
    --k_unsure $K_UNSURE

# 3. SEND DATA TO BE LABELED TO/BY ORACLE

# 4. RETRAIN ON NEWLY LABELED DATA
# TODO
