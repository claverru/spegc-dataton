FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install pytorch-lightning albumentations
RUN pip install pandas timm iterative-stratification 

WORKDIR /workspaces

CMD bash