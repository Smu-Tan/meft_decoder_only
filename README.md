# MEFT for Decoder-only Model
The repository of the Decoder-only Model Part for paper **[Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning](https://arxiv.org/abs/2306.00477)**

Code for the other tasks such as Encoder-Only models can be found at [here](https://github.com/BaohaoLiao/mefts).

## Features
- [x] OPT on Question-Answering

## Installation
```bash
conda create -n mefts python=3.8
conda activate mefts
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Fine-Tuning
#### Run Experiments with OPT
- Edit the **#TODO** places in [scripts/run.sh](/scripts/run.sh)
- Run as
  ```bash
  bash scripts/run.sh
  ```
- openbookqa_test.sh is an example of how to run the openbook_qa task.
  
## Citation
If you find our work or code useful, please cite as:
  ``` bibtex
  @misc{liao2023make,
      title={Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning}, 
      author={Baohao Liao and Shaomu Tan and Christof Monz},
      year={2023},
      eprint={2306.00477},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
  ```
