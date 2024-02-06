# picoGPT - Rust

## Usage
1. download gpt-2 weights & convert them to a more usable format
```sh
python python/download_model.py <model_size 124M | 355M | 774M | 1558M> <model-dir>
```
2. run
```sh
cargo run -r -- --model-dir <model-dir> --prompt "Alan Turing theorized that computers would one day become" --num-tokens 8 --model-dir <model-dir>

# output = " the most powerful machines on the planet."
```
This reproduces the expected picoGPT output.
