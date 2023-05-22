## BERT+KNN online clustering

### Minimal setup

1. Create your encoder: `python create_model.py` (change `create_model.py` to fit your specific needs)
2. Run online evaluation on `data/dev-dataset-task2022-04_preprocessed.json`: `python solution.py`

### ONNX optimizations

If you would like to shrink the size of your model, the script `optimize.py` will create optimized version of BERT encoder for you. Refer to `python optimize.py --help`:
```bash
usage: optimize.py [-h] --model MODEL [--prune PRUNE] [--prune_iterations PRUNE_ITERATIONS] [--batch_size BATCH_SIZE] [--dataset_dir DATASET_DIR] [--onnx_dir ONNX_DIR] [--fuse [FUSE]] [--quantize [QUANTIZE]]
                   [--opset_version OPSET_VERSION] [--do_constant_folding [DO_CONSTANT_FOLDING]]

options:
  -h, --help            show this help message and exit
  --model MODEL         Huggingface model to optimize (default: None)
  --prune PRUNE         Fraction of all heads to prune. (default: 0.0)
  --prune_iterations PRUNE_ITERATIONS
                        Pruning iterations (the higher the better) (default: 5)
  --batch_size BATCH_SIZE
                        Batch size for head importances evaluation. (default: 1)
  --dataset_dir DATASET_DIR
                        Dataset to use for pruning. (default: data)
  --onnx_dir ONNX_DIR   Path to directory where to store ONNX models. (default: onnx)
  --fuse [FUSE]         Fuse some elements of the model (is not supported with quantization) (default: False)
  --quantize [QUANTIZE]
                        Quantize the model. (default: False)
  --opset_version OPSET_VERSION
                        ONNX opset version: 11, 12 or 13. (default: 13)
  --do_constant_folding [DO_CONSTANT_FOLDING]
                        Fold constants during ONNX conversion. (default: False)

```