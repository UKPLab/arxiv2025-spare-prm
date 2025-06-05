<!-- <p  align="center">
  <img src='logo.png' width='200'>
</p> -->

# SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/UKPLab/Llama-3-8b-spare-prm-math)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
<!-- [![CI](https://github.com/UKPLab/arxiv2025-repa-prm/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/arxiv2025-repa-prm/actions/workflows/main.yml) -->

## Description:

This repository includes the training, inference and evaluation code used in our Arxiv 2025 paper - [SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling](). 

We introduced a principled framework for a single-pass alignment and step-annotation for automatic process supervision. Process Reward Models (SPARE-PRMs) trained based on the proposed annotation scheme outperform baselines such as Self-Consistency and ORM-weighted aggregation on four datasets across mathematical, question-answering and spatial reasoning datasets. The annotation scheme is also competitive while being computationally efficient compared to tree-search based annotation methods.

<!-- This is the official template for new Python projects at UKP Lab. It was adapted for the needs of UKP Lab from the excellent [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno).

It should help you start your project and give you continuous status updates on the development through [GitHub Actions](https://docs.github.com/en/actions). -->

> **Abstract:** Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce **S**ingle-**P**ass **A**nnotation with **R**eference-Guided **E**valuation (**SPARE**), a novel structured framework that enables single-pass, per-step annotation by aligning each solution step to one or multiple steps in a reference solution, accompanied by explicit reasoning for evaluation. We show that reference-guided step-level evaluation effectively facilitates process supervision on four datasets spanning three domains: mathematical reasoning, multi-hop compositional question answering, and spatial reasoning. We demonstrate that *SPARE*, when compared to baselines, improves reasoning performance when used for: (1) fine-tuning models in an offline RL setup for inference-time greedy-decoding, and (2) training reward models for ranking/aggregating multiple LLM-generated outputs. Additionally, *SPARE* achieves competitive performance on challenging mathematical datasets while offering 2.6 times greater efficiency, requiring only 38% of the runtime, compared to tree search-based automatic annotation.

## Installation

Create a `conda` / `mamba` / `venv` virtual environment and install the dependencies in `requirements.txt`. E.g.:

```bash
mamba create -n spare
mamba activate spare
pip install -r requirements.txt
```

## Running the experiments

The parameters of the experiments are specified in their respecive `config` files:

```bash
config/
├── eval-config.yaml
├── infer-config.yaml         # infer config for solution generation
├── infer-rm-config.yaml      # infer config for solution scoring using Reward Model (RM)
├── private-config.yaml
├── train-po-config.yaml      # train config for preference-optimization (PO)
├── train-sft-config.yaml     # train config for simple supervised fine-tuning (SFT)
└── train-tc-rm-config.yaml   # train config for Token Classification (TC) based Reward Model (RM)
```

The private api keys such as for using OpenAI models or logging through Neptune API can be provided in the `private-config.yaml` file.

To run a desired task e.g. token classification based reward model (`tc-rm`), execute the following command:

```bash
python train_rm.py # to use the default location of the train-tc-rm-config 
# OR alternatively
python train_rm.py --config my-train-tc-rm-config.yaml
```

A trained SPARE-PRM model based on Llama-3-8b is provided for direct-use at [![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/UKPLab/Llama-3-8b-spare-prm-math). A sample code to use it is given below:

```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

incorrect_token = "-"
correct_token = "+"
step_tag = " ки" # space in the beginning required for correct Llama tokenization

tokenizer = AutoTokenizer.from_pretrained("UKPLab/Llama-3-8b-spare-prm-math")

step_target_ids = tokenizer.convert_tokens_to_ids([incorrect_token, correct_token])
step_tag_id = tokenizer.encode(step_tag)[-1] 

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained("UKPLab/Llama-3-8b-spare-prm-math").to(device).eval()

# include this instruction as it was left as it is during the PRM training.
instruction = "You are an expert at solving challenging math problems spanning across various categories and difficulties such as Algebra, Number Theory, Geometry, Counting and Probability, Precalculus etc. For a given math problem, your task is to generate a step-by-step reasoning-based solution providing an answer to the question. Identify the correct concepts, formulas and heuristics that needs to be applied and then derive the contents of the reasoning steps from the given contexts and accurate calculations from the previous reasoning steps."
question = "Yann and Camille go to a restaurant. </S>\nIf there are 10 items on the menu, and each orders one dish, how many different combinations of meals can Yann and Camille order if they refuse to order the same dish? (It does matter who orders what---Yann ordering chicken and Camille ordering fish is different from Yann ordering fish and Camille ordering chicken.)"
correct_generation = "Let's think step by step.\nYann can order 1 of the 10 dishes. ки\nWhen he picks a dish, there are 9 left for Camille to choose from. ки\nThus, there are $10\\cdot 9=\\boxed{90}$ possible combinations.\nHence, the answer is 90. ки\n"
incorrect_generation = "Let's think step by step.\nWithout any restrictions, Yann and Camille could both order the same dish out of the 10 options, for a total of $10 \\cdot 9$ dishes. ки\nHowever, since Yann orders one of the 9 dishes that Camille didn't order (and vice versa), the number of possible combinations becomes $10 \\cdot 9 - 8 = \\boxed{72}$.\nHence, the answer is 72. ки\n"

for generation in (correct_generation, incorrect_generation):
    message = [
        dict(role="system", content=instruction),
        dict(role="user", content=question),
        dict(role="user", content=generation),
    ]

    input_ids = tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(input_ids).logits[:,:,step_target_ids]
        scores = logits.softmax(dim=-1)[:,:,1] # correct_token at index 1 in the step_target_ids  
        step_scores = scores[input_ids == step_tag_id]
        print(step_scores)
        
# tensor([0.9561, 0.9496, 0.9527]) - correct_generation
# tensor([0.6638, 0.6755]) - incorrect_generation
```

Contact person: [Md Imbesat Hassan Rizvi](mailto:imbesat.rizvi@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


<!-- ## Getting Started

> **DO NOT CLONE OR FORK**

If you want to set up this template:

1. Request a repository on UKP Lab's GitHub by following the standard procedure on the wiki. It will install the template directly. Alternatively, set it up in your personal GitHub account by clicking **[Use this template](https://github.com/rochacbruno/python-project-template/generate)**.
2. Wait until the first run of CI finishes. Github Actions will commit to your new repo with a "✅ Ready to clone and code" message.
3. Delete optional files: 
    - If you don't need automatic documentation generation, you can delete folder `docs`, file `.github\workflows\docs.yml` and `mkdocs.yml`
    - If you don't want automatic testing, you can delete folder `tests` and file `.github\workflows\tests.yml`
    - If you do not wish to have a project page, delete folder `static` and files `.nojekyll`, `index.html`
4. Read the file [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md)  for more information about development. -->

## Cite

If you use this repository, our trained SPARE-PRM model or our work, please cite:

```
@misc{rizvi2024spare,
      title={SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling}, 
      author={Md Imbesat Hassan Rizvi and Xiaodan Zhu and Iryna Gurevych},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/}, 
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
