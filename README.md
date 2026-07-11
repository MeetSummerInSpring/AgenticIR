# An Intelligent Agentic System for Complex Image Restoration Problems
Kaiwen Zhu<sup>\*</sup>, [Jinjin Gu](https://www.jasongt.com/)<sup>\*</sup>, [Zhiyuan You](https://zhiyuanyou.github.io/), [Yu Qiao](https://scholar.google.com/citations?hl=en&user=gFtI-8QAAAAJ), [Chao Dong](http://xpixel.group/)


ICLR 2025


[Paper](https://arxiv.org/abs/2410.17809) | [Project Page](https://kaiwen-zhu.github.io/research/agenticir)


## Overview
### Learning from exploration

![exploration](assets/exploration.png)

### Workflow
![workflow](assets/workflow.png)

## Examples
### Restoration of real-world images
Restore a UDC image (from [this work](https://yzhouas.github.io/projects/UDC/udc.html)) by motion deblurring, defocus deblurring, and low light enhancement.
<div>
<img src="assets/udc10_input.png" width="49%"/>
<img src="assets/udc10_output.png" width="49%"/>
</div>

Restore an underwater image (from [this work](https://ieeexplore.ieee.org/document/9001231)) by defocus deblurring, dehazing, and motion deblurring.
<div>
<img src="assets/9094_input.png" width="49%"/>
<img src="assets/9094_output.png" width="49%"/>
</div>

### Effectiveness of planning with experience
![exp](assets/exp.png)

### Effectiveness of workflow designs
![ref](assets/ref.png)
![rb](assets/rb.png)


## Installation
Please refer to [INSTALL.md](installation/INSTALL.md).

## Usage
### Fine-tuning DepictQA
Please refer to [this](installation/FINE-TUNE.md).

### Setup
+ Fill in the API key in `config.yml`.
+ Run `python src/app_eval.py` and `python src/app_comp.py` in the directory `DepictQA`.

### Data preparation
To generate complexly degraded images, run `python -m dataset.synthesize`. You should place clean images in `dataset/HQ/` and corresponding depth maps in `dataset/depth/`. In the paper we use the [MiO100](https://github.com/Xiangtaokong/MiOIR?tab=readme-ov-file#setp-2-download-the-testsets) dataset. The degradation combinations are listed in `dataset/degradations.txt`. You can customize combinations in `dataset/degradations.txt` or degradation types in `dataset/add_single_degradation.py`.


The data used in the paper can be downloaded from [this link](https://drive.google.com/drive/folders/1DnMeA1GAoxlP0DET3BrRkbFhyTg4Evda?usp=sharing).


### Learning
To let the agent learn from exploration, run 
+ `python -m exploration.exhaust_seq` to generate images to explore;
+ `python -m exploration.explore` to accumulate experience by evaluating images;
+ `python -m exploration.distill` to summarize the experience and distill knowledge.

### Inference
Run `python -m pipeline.infer` to restore an image (path specified in `pipeline/infer.py`).

## BibTex
```
@inproceedings{agenticir,
      title={An Intelligent Agentic System for Complex Image Restoration Problems},
      author={Kaiwen Zhu and Jinjin Gu and Zhiyuan You and Yu Qiao and Chao Dong},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=3RLxccFPHz}
}
```