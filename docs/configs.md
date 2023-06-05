# Configuration items

### data_cfg
* Data configurations
>
>  * Args
>     * dataset_name: support `CASIA-B`, `OUMVLP`, `GREW` and `Gait3D` now.
>     * dataset_root: The path of storing your dataset.
>     * num_workers: The number of workers to collect data.
>     * dataset_partition: The path of storing your dataset partition file. It splits the dataset to two parts, including train set and test set.
>     * cache: If `True`, load all data to memory during building dataset.
>     * test_dataset_name: The name of test dataset. 
>     * frame_threshold: The threshold of the sequence length, default: 0
----

### loss_cfg
* Loss function
>  * Args
>     * type: Loss function type, support `TripletLoss` ,`SupConLoss`,`SupConLoss_LP` .
>     * loss_term_weight: loss weight.
>     * log_prefix: the prefix of loss log.

----
### optimizer_cfg
* Optimizer
>  * Args
>     * solver: Optimizer type, example: `SGD`, `Adam`.
>     * **others**: Please refer to `torch.optim`.


### scheduler_cfg
* Learning rate scheduler
>  * Args
>     * scheduler : Learning rate scheduler, example: `OneCycleLR`.
>     * **others** : Please refer to `torch.optim.lr_scheduler`.
----
### model_cfg
* Model to be trained
>  * Args
>     * model : Model type, please refer to [Model Library](../fastposegait/modeling/models) for the supported values.
>     * **others** : Please refer to the [Training Configuration File of Corresponding Model](../configs).
----
### evaluator_cfg
* Evaluator configuration
>  * Args
>     * enable_float16: If `true`, enable the auto mixed precision mode, default: false.
>     * restore_ckpt_strict: If `true`, check whether the checkpoint is the same as the defined model.
>     * restore_hint: `int` value indicates the iteration number of restored checkpoint, `str` value indicates the path of restored checkpoint.
>     * save_name: The name of the experiment.
>     * eval_func: The function name of evaluation. 
>     * sampler:
>       - type: The name of sampler. Choose `InferenceSampler`.
>       - sample_type: In general, we use `all_ordered` to input all frames by the natural order, which makes sure the tests are consistent.
>       - batch_size: `int` values.
>       - **others**: Please refer to [data.sampler](../fastposegait/data/sampler.py) and [data.collate_fn](../fastposegait/data/collate_fn.py)
>     * transform: Support pose data transform. `GaitTR_MultiInput`, `SkeletonInput`, `GaitGraph_MultiInput`
>     * metric: `euc` or `cos`.

----
### trainer_cfg
* Trainer configuration
>  * Args
>     * restore_hint: `int` value indicates the iteration number of restored checkpoint, `str` value indicates the path to restored checkpoint. The option is often used to finetune on new dataset or restore the interrupted training process.
>     * fix_BN: If `True`, we fix the weight of all `BatchNorm` layers.
>     * log_iter: Log the information per `log_iter` iterations.
>     * save_iter: Save the checkpoint per `save_iter` iterations.
>     * with_test: If `True`, we test the model every `save_iter` iterations. (*Disable in Default*)
>     * optimizer_reset: If `True` and `restore_hint!=0`, reset the optimizer while restoring the model.
>     * scheduler_reset: If `True` and `restore_hint!=0`, reset the scheduler while restoring the model.
>     * sync_BN: If `True`, applies Batch Normalization synchronously.
>     * total_iter: The total training iterations, `int` values.
>     * sampler:
>       - type: The name of sampler. Support `TripletSampler` and  `CommonSampler`
>       - sample_type: `[all, fixed, unfixed]` indicates the number of frames used to test, while `[unordered, ordered]` means whether input sequence by the natural order. Example: `fixed_unordered` means selecting fixed number of frames randomly.
>       - batch_size: *[P,K]* for TripletSampler where `P` denotes the subjects in one training batch while `K` represents the sequences every subject owns. For CommonSampler, *[B]*
>       - **others**: Please refer to [data.sampler](../fastposegait/data/sampler.py) and [data.collate_fn](../fastposegait/data/collate_fn.py).
>     * **others**: Please refer to `evaluator_cfg`.
---
**Note**: 
- All the config items will be merged into [default.yaml](../configs/default.yaml), and the current config is preferable.
- The output directory, which includes the log, checkpoint and summary files, depends on the defined `dataset_name`, `model` and `save_name` settings, like `output/${dataset_name}/${model}/${save_name}`.
# Example

```yaml
data_cfg:
  dataset_name: CASIA-B
  dataset_root: your_path
  num_workers: 1
  dataset_partition: ./datasets/CASIA-B/CASIA-B.json
  remove_no_gallery: false
  cache: true
  frame_threshold: 0
  test_dataset_name: CASIA-B

evaluator_cfg:
  enable_float16: false
  restore_ckpt_strict: true
  restore_hint: 80000
  save_name: tmp
  eval_func: identification
  sampler:
    batch_size: 4
    sample_type: all_ordered
    type: InferenceSampler
  transform:
    - type: SkeletonInput

loss_cfg:
  loss_term_weight: 1.0
  margin: 0.2
  type: TripletLoss
  log_prefix: triplet

model_cfg:
  model: tmp

optimizer_cfg:
  lr: 0.1
  momentum: 0.9
  solver: SGD
  weight_decay: 0.0005

scheduler_cfg:
  max_lr: 0.01
  total_steps: 20000
  scheduler: OneCycleLR

trainer_cfg:
  enable_float16: false
  with_test: false
  fix_BN: false
  log_iter: 100
  restore_ckpt_strict: true
  optimizer_reset: false
  scheduler_reset: false
  restore_hint: 0
  save_iter: 2000
  save_name: tmp
  sync_BN: false
  total_iter: 80000
  sampler:
    batch_shuffle: false
    batch_size:
      - 8
      - 16
    frames_num_fixed: 30
    frames_num_max: 50
    frames_num_min: 25
    sample_type: fixed_unordered
    type: TripletSampler
  transform:
    - type: SkeletonInput


```
