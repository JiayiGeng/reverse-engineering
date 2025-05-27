# reverse-engineering

This repository contains the code for paper [Are Large Language Models Reliable AI Scientists? Assessing Reverse-Engineering of Black-Box Systems](https://arxiv.org/pdf/2505.17968).

## News
- **[2025/05/27]** Initial release.


## Dependencies
```bash
pip install -r requirements.txt
```

## LLM models
```bash
# OpenAI
export OPENAI_API_KEY="LLM api key"
azure="none"

# Azure
export AZURE_OPENAI_API_KEY="LLM api key"
export OPENAI_API_VERSION="LLM API base url"
export AZURE_OPENAI_ENDPOINT="LLM API endpoint"
azure="azure"

# Claude
export ANTHROPIC_API_KEY="LLM api key"

# Deepseek
export TOGETHER_API_KEY="LLM api key"
```


## Run experiments
Run the following command for the passive observation experiments:
```bash
./scripts/obs.sh
```

Run the following command for the intervention experiments:
```bash
./scripts/intv.sh
```



## Question and Issue
Please contact Jiayi Geng at `jiayig@princeton.edu` for any questions or issues.


## Citation
```bibtex
@article{geng2025are,
   title={Are Large Language Models Reliable AI Scientists? Assessing Reverse-Engineering of Black-Box Systems},
   author={Geng, Jiayi and Chen, Howard and Arumugam, Dilip and Griffiths, Thomas L},
   journal={arXiv preprint arXiv: 2505.17968},
   year={2025}
}
```
