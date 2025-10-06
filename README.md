# VDVAE hallucination code for the paper: "The oneirogen hypothesis: modeling the hallucinatory effects of classical psychedelics in terms of replay-dependent plasticity mechanisms"
By Colin Bredenberg, Fabrice Normandin, Blake Richards, and Guillaume Lajoie

This codebase is adapted from https://github.com/openai/vdvae, and uses pretrained VDVAE networks from the paper, "Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images." Code adapted from this repository is stored in the "pretrain" folder.

# Setup
Several additional packages are required to run this code, which we manage with `uv` (available here: https://docs.astral.sh/uv/getting-started/installation/)
Running `uv sync`, followed by `uv sync --all-extras` will create a virtual environment `vdvae` that includes all required dependencies.

Also, if you want to run the FFHQ-256 pretrained model, you will have to download the data:
```
./pretrain/setup_ffhq256.sh
```

# Restoring saved models
We generate hallucination visualizations for networks trained on the ImageNet 64 and FFHQ-256 datasets. To download these models follow the instructions below:

### ImageNet 64
```bash
# 125M parameter model, trained for 1.6M iters (about 2.5 weeks on 32 V100)
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
```

### FFHQ-256
```bash
# 115M parameters, trained for 1.7M iterations (or about 2.5 weeks) on 32 V100
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq256-iter-1700000-opt.th
```

# Reproducing paper figures

To reproduce the results of our paper pertaining to the VDVAE models (Figure 6, S7, S8), run the halluc.ipynb file.
