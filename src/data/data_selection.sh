PYTHON_MODULES_PATH=../../src
export PYTHONPATH=$PYTHON_MODULES_PATH

python data_selection.py \
    -i /work/train_data_folder/ \
    -o /work/train_data_to_review/ \
    -k 20