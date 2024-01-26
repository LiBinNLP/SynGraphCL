# SynGraphCL

Code for the paper "Few-Shot Semantic Dependency Parsing via Graph Contrastive Learning"

Due to the large amount of similarity between syntactic dependencies and semantic dependencies, we build a syntax-guided few-shot semantic dependency model.
We have provided two examples which show the syntactic dependency tree and semantic dependency graph to visually illustrate the similarity between syntactic dependence and semantic dependence.

For the English sentence "He wants to teach me English", the example is shown as follows:
![English Example](https://github.com/LiBinNLP/SynGraphCL/blob/main/en_example_syn_sem.png)

For the Chinese sentence "他想教我英语", the example is shown as follows:
![Chines Example](https://github.com/LiBinNLP/SynGraphCL/blob/main/cn_example_syn_sem.png)

As a prerequisite, the following requirements should be satisfied:
* `python`: >= 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.8
* [`transformers`](https://github.com/huggingface/transformers): >= 4.0

## Contrastive Samples Construction
* Prepare the raw data. Put your raw sentences in the file at `./data/contrastive/raw_sentence.txt`
* Generating original samples.
Syntactic Dependency Parsing for raw sentences with stanza, the original data will be generated and saved at `./data/contrastive/original.conllu`:

```py
python stanza_dep_parse.py
```

* Generating positive samples. Positive samples will be generated and saved at `./data/contrastive/positive.conllu`:

```py
python positive_data_gen.py
```

* Generating negative samples. Negative samples will be generated and saved at `./data/contrastive/negative.conllu`:
```py
python negative_data_gen.py
```


## Contrastive Pre-training
Run the following command, the pre-trained GNN model will be saved at `/SynGraphCL/output/pretrain/ggnn/`
```py
python supar/cmds/gnn_pretrain.py
train
-b
-d 0
-c ../../gnn-pretrain.ini
```

## Fine-tuning
Run the following command, the fine-tuned SDP model will be saved at `/SynGraphCL/output/sdp/ggnn/`
```py
python supar/cmds/gnn_finetune_sdp.py 
train
-b
-d 0
-c ../../gnn-pretrain.ini
```

## Evaluating
Run the following command to load the trained model and evaluate it:
```py
python supar/cmds/gnn_finetune_sdp.py 
evaluate
-d 0
-c ../../gnn-pretrain.ini
```
