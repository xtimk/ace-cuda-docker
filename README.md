# Ace dockerfile using NVIDIA cuda
Ace Step dockerfile that enables GPU accelerated processing.

I created this repo just for testing out the nvidia docker images integrations and how it works.

The dockerfile is structured accordingly so that you can specify as arguments the versions of cuda, pytorch, torchaudio, torchvision that you want to use for building the image.

You can aswell specify environment variables to customize the configuration of ace step (like the model to use). 

If you dont specify any of them it will just use the default values for ace step.

I did this in order to make it run also on older gpu s that may support lower versions of pytorch etc..

## Build commands
### RTX 2000+ (Ada generation and later)
```bash
docker build ^
    -t tizm/ace-cuda:1.1.0-cu12.8.1-py3.11-u22.04 .
```

### GTX 1070 (Pascal generation)
Since I got an old PC with a GTX 1070 I wanted to give it a try also with this GPU.

I tried several approaches, but didnt manage to get it to work

Approach 1
```bash
docker build ^
    --build-arg PYTORCH_VERSION=2.7.1 ^
    --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu128 ^
    --build-arg TORCHAUDIO_VERSION=2.7.1 ^
    --build-arg TORCHVISION_VERSION=0.22.1 ^
    -t tizm/ace-cuda:test-2.0.0-cu12.8.1-py3.11-u22.04 .
```

Approach 2
```bash
docker build ^
    --build-arg PYTORCH_VERSION=2.7.1 ^
    --build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu121 ^
    --build-arg TORCHAUDIO_VERSION=2.7.1 ^
    --build-arg TORCHVISION_VERSION=0.22.1 ^
    -t tizm/ace-cuda:test-2.0.0-cu12.8.1-py3.11-u22.04 .
```

using pytorch versions like the 2.1.2 or 2.2.2 it's giving errors on ace step application side.

Still working on this, since running ace step natively on the os (windows 10) works.

The pytorch, torchaudio, torchvision version choosed are the one that ace step is actually using on my PC when running natively.


## Run the container
To run the container locally
```bash
docker run --rm --runtime=nvidia --gpus all -p 127.0.0.1:7860:7860 --volume C:\\path\\to\\workspace:/workspace tizm/ace-cuda:test-2.0.0-cu12.8.1-py3.11-u22.04
```

This mounts a bind volume that persist in your host os, so that the models downloaded remains there, and you dont have to re-download each time you restart the container.

You can set the ACESTEP_CONFIG_PATH env variable to choose the model to use
Defaults to acestep-v15-turbo

For example, to use the XL model
```
ACESTEP_CONFIG_PATH=acestep-v15-xl-base
```


## Images available
I pushed into docker hub these images. (https://hub.docker.com/repository/docker/tizm/ace-cuda/)

Tags:

 - 1.0.0-cu13.1.2: testing version with cuda 13 (probably not working)
 - 1.1.0-cu12.8.1-py3.11-u22.04: the build for Ada and later generations
 - 1.2.0-cu12.8.1-py3.11-u22.04: the build for Ada and later gens, that can be also deployed on vast ai / runpod.
