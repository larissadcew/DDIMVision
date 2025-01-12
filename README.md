# Denoising Diffusion Implicit Models (DDIM)

This is a PyTorch implementation of DDIM, based on the original paper available at: https://arxiv.org/abs/2010.02502

## Usage Guide

### Configuration and Training
1. All configurable parameters are in the `config.yml` file
2. Adjust parameters as needed
3. Run `train.py` to start training

### Image Generation
Run `generate.py` with the following parameters:

Main Parameters:
- `-cp`: Checkpoint path
- `--sampler`: Sampling method ('ddpm' or 'ddim')
- `-bs`: Number of images generated simultaneously (default: 16)
- `--steps`: DDIM sampling steps (default: 100)

Visualization Parameters:
- `--result_only`: Show only final results (default: False)
- `--interval`: Interval between extracted images (default: 50)
- `--nrow`: Images per row in display (default: 4)
- `--show`: Display the resulting image (default: False)
- `-sp`: Path to save the image

Technical Parameters:
- `--device`: Device used ('cuda' or 'cpu')
- `--eta`: DDIM η parameter (default: 0.0)
- `--method`: Sampling method ('linear' or 'quadratic')
- `--to_grayscale`: Convert to grayscale (default: False)

## Practical Examples

### MNIST
[Download checkpoint](https://drive.google.com/file/d/1gwhczBWOjUtw4Fz_y2PidyKnrUsMSN8t/view?usp=drive_link)

To generate sampling process:
```bash
python generate.py -cp "checkpoint/mnist.pth" -bs 16 --interval 3 --show -sp "data/result/mnist_sampler.png" --sampler "ddim" --steps 50
```

To generate multiple images:
```bash
python generate.py -cp "checkpoint/mnist.pth" -bs 256 --show -sp "data/result/mnist_result.png" --nrow 16 --result_only --sampler "ddim" --steps 50
```

### CIFAR10
[Download checkpoint](https://drive.google.com/file/d/1GRVfLSfjGtEPJzxg52k4wj4w2TKk-utO/view?usp=drive_link)

To generate sampling process:
```bash
python generate.py -cp "checkpoint/cifar10.pth" -bs 16 --interval 10 --show -sp "data/result/cifar10_sampler.png" --sampler "ddim" --steps 200 --method "quadratic"
```

To generate multiple images:
```bash
python generate.py -cp "checkpoint/cifar10.pth" -bs 256 --show -sp "data/result/cifar10_result.png" --nrow 16 --result_only --sampler "ddim" --steps 200 --method "quadratic"
```

Example of generated image:

![](data/result/cifar10_result.png)