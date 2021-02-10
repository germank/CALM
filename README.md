# CALM Task

This library implements the task described at the paper [_Evaluating Online Continual Learning with CALM_](https://arxiv.org/abs/2004.03340).

## Installation 

Download the required corpora by going into the data directory and running
```
pip install -r requirements.txt
python create_datasets.py
```

## Example Usage

```python
# model definition
import random
def get_loss(x, y):
    return random.random()
def train():
    pass

import calm
import numpy as np; np.random.seed(42) # different seeds will generate different segmentations
MultiLingual_dev_10k = calm.CALM("data/news_dev", "char", switch_frequency=10000, nswitches=100, window_size=20, batch_size=10, cuda=False)
results_fn = "loss_log.jsonl"
with open(results_fn, "w") as flog:
    for x, y in MultiLingual_dev_10k:
        loss = get_loss(x, y)
        train()
        calm.log_loss(flog, MultiLingual_dev_10k, loss)
print(calm.compute_metrics(results_fn))
```

## Important methods and objects

The module `calm.py` contains the following methods and objects:

* `CALM`: creates a data iterator for the CALM tasks. It implicitly depends on the 
numpy random generator to randomly switch between classes. It takes
    * `corpus_path`: path to the root of a CALM corpus (e.g. data/news_dev).
    * `model_level`: one of "char" or "word". (str)
    * `switch_frequency`: mean sequence length in number of tokens. (int)
    * `nswitches`: number of switches occurring within a same class. (int)
    * `window_size`: number of consecutive tokens taken per batch. (int)
    * `batch_size`: number of concurrent text streams. (int)
    * `cuda`: send data to GPU (bool)
* `log_loss`: takes a file handler, a CALM iterator object and the loss, and logs all information to a file.
* `compute_metrics`: gets a log filename and computes the following metrics, also disaggregated by class:
    * `loss`: mean loss.
    * `total_pp`: mean overall perplexity.
    * `ppl@sw`: mean perplexity restricted to the first 10 batches after a switch.
    * `rec`: number of batches that the model needs to recover back to the mean loss on the last sequence of a same class.


## Replication of the paper's results

If you want to replicate our own experiments, you can find the code we used [here](https://github.com/germank/lifelonglm).

## Citation

Should you want to cite this work, you can use the following BibTeX entry:

```
@article{Kruszewski:Sorodoc:Mikolov:2020,
  title={Class-Agnostic Continual Learning of Alternating Languages and Domains},
  author={Kruszewski, Germ{\'a}n and Sorodoc, Ionut-Teodor and Mikolov, Tomas},
  journal={arXiv preprint arXiv:2004.03340},
  year={2020}
}
```
