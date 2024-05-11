# smollama
fucked script to create a tiny llm for phone use because i dont know how to use pytorch

option 1:
 - i learn pytorch
 - i tweak hparams to get a llama model class that is tiny w/ random weights
 - i then use llama.cpp conversion script

option 2:
 - i cook a model file directly with gguf-py
 - hopefully it works?

this project is _option 2_

```sh
git clone ...
cd ...
wget 'https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GPTQ/raw/main/tokenizer.json'
# outputs to /tmp/example.gguf because its faster (since tmpfs is ram)
LLAMACPP_PATH=path/to/llama.cpp python3 ./random_llama.py
path/to/llama.cpp/main -m /tmp/example.gguf
```
