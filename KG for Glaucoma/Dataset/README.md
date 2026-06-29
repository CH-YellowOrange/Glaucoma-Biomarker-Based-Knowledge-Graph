---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: int64
  - name: filename
    dtype: string
  - name: annotation
    dtype: string
  - name: description
    dtype: string
  splits:
  - name: train
    num_bytes: 345103034.0
    num_examples: 589
  - name: test
    num_bytes: 71637072.0
    num_examples: 100
  download_size: 415491685
  dataset_size: 416740106.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
