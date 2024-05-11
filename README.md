# smollama
fucked script to create a tiny llm for phone use because i dont know how to use pytorch

```sh
git clone ...
cd ...
wget 'https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GPTQ/raw/main/tokenizer.json'
# outputs to /tmp/example.gguf because its faster (since tmpfs is ram)
python3 ./random_llama.py
path/to/llama.cpp/main -m /tmp/example.gguf
```
