<p  align="center">
  <img src='logo.png' width='200'>
</p>

# arxiv2025_repa_prm
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/UKPLab/Llama-3-8b-spare-prm-math)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/arxiv2025-repa-prm/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/arxiv2025-repa-prm/actions/workflows/main.yml)

This is the official template for new Python projects at UKP Lab. It was adapted for the needs of UKP Lab from the excellent [python-project-template](https://github.com/rochacbruno/python-project-template/) by [rochacbruno](https://github.com/rochacbruno).

It should help you start your project and give you continuous status updates on the development through [GitHub Actions](https://docs.github.com/en/actions).

> **Abstract:** Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce **S**ingle-**P**ass **A**nnotation with **R**eference-Guided **E**valuation (**SPARE**), a novel structured framework that enables single-pass, per-step annotation by aligning each solution step to one or multiple steps in a reference solution, accompanied by explicit reasoning for evaluation. We show that reference-guided step-level evaluation effectively facilitates process supervision on four datasets spanning three domains: mathematical reasoning, multi-hop compositional question answering, and spatial reasoning. We demonstrate that *SPARE*, when compared to baselines, improves reasoning performance when used for: (1) fine-tuning models in an offline RL setup for inference-time greedy-decoding, and (2) training reward models for ranking/aggregating multiple LLM-generated outputs. Additionally, *SPARE* achieves competitive performance on challenging mathematical datasets while offering 2.6 times greater efficiency, requiring only 38% of the runtime, compared to tree search-based automatic annotation.

Contact person: [Md Imbesat Hassan Rizvi](mailto:imbesat.rizvi@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


## Getting Started

> **DO NOT CLONE OR FORK**

If you want to set up this template:

1. Request a repository on UKP Lab's GitHub by following the standard procedure on the wiki. It will install the template directly. Alternatively, set it up in your personal GitHub account by clicking **[Use this template](https://github.com/rochacbruno/python-project-template/generate)**.
2. Wait until the first run of CI finishes. Github Actions will commit to your new repo with a "✅ Ready to clone and code" message.
3. Delete optional files: 
    - If you don't need automatic documentation generation, you can delete folder `docs`, file `.github\workflows\docs.yml` and `mkdocs.yml`
    - If you don't want automatic testing, you can delete folder `tests` and file `.github\workflows\tests.yml`
    - If you do not wish to have a project page, delete folder `static` and files `.nojekyll`, `index.html`
4. Prepare a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install .
pip install -r requirements-dev.txt # Only needed for development
```
5. Adapt anything else (for example this file) to your project. 

6. Read the file [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md)  for more information about development.

## Usage

### Using the classes

To import classes/methods of `arxiv2025_repa_prm` from inside the package itself you can use relative imports: 

```py
from .base import BaseClass # Notice how I omit the package name

BaseClass().something()
```

To import classes/methods from outside the package (e.g. when you want to use the package in some other project) you can instead refer to the package name:

```py
from arxiv2025_repa_prm import BaseClass # Notice how I omit the file name
from arxiv2025_repa_prm.subpackage import SubPackageClass # Here it's necessary because it's a subpackage

BaseClass().something()
SubPackageClass().something()
```

### Using scripts

This is how you can use `arxiv2025_repa_prm` from command line:

```bash
$ python -m arxiv2025_repa_prm
```

### Expected results

After running the experiments, you should expect the following results:

(Feel free to describe your expected results here...)

### Parameter description

* `x, --xxxx`: This parameter does something nice

* ...

* `z, --zzzz`: This parameter does something even nicer

## Development

Read the FAQs in [ABOUT_THIS_TEMPLATE.md](ABOUT_THIS_TEMPLATE.md) to learn more about how this template works and where you should put your classes & methods. Make sure you've correctly installed `requirements-dev.txt` dependencies

## Cite

Please use the following citation:

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
