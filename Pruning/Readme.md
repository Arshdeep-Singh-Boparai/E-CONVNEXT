# Pruning convnext_tiny_1k_224_ema model

## sorted index generation of all layers in unpruned ConvNeXt
```
 python alllayers_sorted_index_generation.py \
  --state_dict_path /path/to/convnext_tiny_1k_224_ema.pth \
  --output_dir ./artifacts/indexes
```

## Obtaining pruned model
```
 python pruned_convnext.py \
  --state_dict_path /path/to/convnext_tiny_1k_224_ema.pth \
  --sorted_index_path ./artifacts/indexes/sorted_index_convnext_op.pkl \
  --dims 96 192 384 256 \
  --num_classes 1000 \
  --save_dir ./artifacts/pruned
```

