FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
RUN pip install --no-cache-dir yacs==0.1.8 pandas==1.4.0 iopath==0.1.9 pytorch-lightning==1.5.10 torchmetrics==0.7.1 wandb==0.12.10 rich==11.1.0
