# CREATE
Authors: Manya Wadhwa, Tiasa Singha Roy, Harvey Lederman,  Junyi Jessy Li, Greg Durrett

[Benchmark🤗](https://huggingface.co/datasets/wadhma/CREATE) | [Leaderboard](https://manyawadhwa.github.io/projects/create/) |

## Overview
CREATE is a benchmark designed to measure associative reasoning in models. This benchmark evaluates whether models can construct valid, diverse, and insightful paths that connect two concepts through intermediate entities or relationships.


We introduce creative utility, a unified metric that captures both the quality and diversity of generated connections. Creative utility includes a patience parameter (p), which controls how utility is distributed across the ranked list of responses.


Example query:

>“What are different ways to connect Dakota Johnson to people who starred in fantasy or science-fiction movies?”

We want the model to generate paths like:

>Dakota Johnson co-stars with Chris Evans in Materialists; Chris Evans played Captain America in The Avengers.

>Dakota Johnson is the stepdaughter of Antonio Banderas, who voiced Puss in Boots in Shrek.

These responses illustrate associative creativity: each path is coherent, factually grounded, and offers a distinct conceptual route between the two endpoints.



## 🛠️ Installation

First, install the required dependencies: 
```bash
python3 -m venv create_env 
source create_env/bin/activate
pip install -r requirements.txt 
```


## 🚀 Running CREATE

Our benchmark (https://huggingface.co/datasets/wadhma/CREATE) is available on huggingface! The following code snippet shows how to access the benchmark. 

```python 
from datasets import load_dataset
data = load_dataset('wadhma/CREATE')['train'].to_pandas() 
print(data['query']) ## the benchmark questions
```

The question answering prompt we used for all models is included in [prompt.py](prompt.py).


## 📊 Output Evaluation Instruction

Once your model predictions on the benchmark are ready, we provide an evaluation script to compute the creative utility of model answers. In the paper we use `gpt-oss-120b' to run evaluations since they were at a larger scale. In the script you have the option of using any evaluator, as longas you specify the model name. 

Note the format and the required fields in your input file/HF repo in [evaluate_creative_utility.py](evaluate_creative_utility.py) [COMING SOON!]