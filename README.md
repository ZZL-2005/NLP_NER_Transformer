# NLP_NER_Transformer

## 环境配置

conda create -n nlp-ner-transformer python=3.10 -y
conda activate nlp-ner-transformer
pip install jupyter ipykernel matplotlib
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

## 项目架构说明



## 训练指令

python train.py \
  --exp_name transformer \
  --model_name transformer \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 50 \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 4 \
  --max_len 5000 \
  --dropout 0.1 \
  --device cuda


python train.py \
  --exp_name attnres_transformer \
  --model_name attnres_transformer \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 50 \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 4 \
  --max_len 5000 \
  --dropout 0.1 \
  --device cuda