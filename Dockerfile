FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git

RUN pip install pytorch-lightning albumentations
RUN pip install pandas timm iterative-stratification
RUN pip install git+https://github.com/openai/CLIP.git

WORKDIR /workspaces

ENV PYTHONPATH=/workspaces/spegc-dataton

CMD bash
