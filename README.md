## environment
'''
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy rouge_score fire openai sentencepiece wandb
pip install transformers==4.28.1
pip install accelerate==0.33.0
'''
You can also create automatically by using auto.yaml
'''
conda env create -f auto.yaml
'''
## Datasets  
Download the training ([GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)) dataset, and store it under the `./data` directory.   

## Training models with poisoned data
'''
For gpt2-xl
    bash run-gptxl.sh
For gpt-j
    bash run-gptj.sh
For llama
    bash run-llama.sh
For llama2
    bash run-llama2.sh
'''
