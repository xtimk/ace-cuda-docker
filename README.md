# Ace dockerfile using NVIDIA cuda
Ace Step dockerfile that enables GPU accelerated processing.

## Build commands
build for RTX 2000+ (Ada generation and later)
```bash
docker build ^
    -t tizm/ace-cuda:1.1.0-cu12.8.1-py3.11-u22.04 .
```

The dockerfile is structured accordingly so that you can specify as arguments the versions of cuda, pytorch, torchaudio, torchvision that you want to use for building the image.

If you dont specify any of them it will just use the default values for ace step.

I did this in order to make it run also on older gpu.

## On GTX 1070
I tried several approaches, but didnt manage to get it to work

Approach 1
```bash
docker build ^
    --build-arg PYTORCH_VERSION=2.7.1 ^
    --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu128 ^
    --build-arg TORCHAUDIO_VERSION=2.7.1 ^
    --build-arg TORCHVISION_VERSION=0.22.1 ^
    -t tizm/ace-cuda:2.0.0-cu12.8.1-py3.11-u22.04 .
```

Approach 2
```bash
docker build ^
    --build-arg PYTORCH_VERSION=2.7.1 ^
    --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 ^
    --build-arg TORCHAUDIO_VERSION=2.7.1 ^
    --build-arg TORCHVISION_VERSION=0.22.1 ^
    -t tizm/ace-cuda:2.0.0-cu12.8.1-py3.11-u22.04 .
```

## Run the container
To run the container locally
```bash
docker run --rm --runtime=nvidia --gpus all -p 127.0.0.1:7860:7860 --volume C:\\path\\to\\workspace:/workspace tizm/ace-cuda:2.0.0-cu12.8.1-py3.11-u22.04
```

This mounts a bind volume that persist in your host os, so that the models downloaded remains there, and you dont have to re-download each time you restart the container.

You can set the ACESTEP_CONFIG_PATH env variable to choose the model to use
Defaults to acestep-v15-turbo

For example, to use the XL model
```
ACESTEP_CONFIG_PATH=acestep-v15-xl-base
```
