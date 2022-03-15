PYTHON_MODULES_PATH=../../src
export PYTHONPATH=$PYTHON_MODULES_PATH

python3 train_model.py \
  -t $PYTHON_MODULES_PATH/data/train/labels.csv \
  -v $PYTHON_MODULES_PATH/data/valid/labels.csv \
  -c $PYTHON_MODULES_PATH/data/config.yaml \
  -o ../../models