# VALL-E 的非官方 PyTorch 实现
语言 : 🇨🇳 | [🇺🇸](./README.md) 

这是一份 VALL-E（[Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)）非官方 PyTorch 实现的说明文档。

我们可以在单个 GPU 上训练 VALL-E 模型。

![模型](./docs/images/Overview.jpg)

## 演示

* [官方演示](https://valle-demo.github.io/)
* [复现演示](https://lifeiteng.github.io/valle/index.html)

<img src="./docs/images/vallf.png" width="500" height="400">


## 更广泛的影响

> 由于 VALL-E 能够合成保持说话者身份的语音，它可能在错误使用模型时带来潜在风险，如欺骗声音识别或冒充特定说话者。

为避免滥用，不会提供经过良好训练的模型和服务。

## 安装依赖

按照以下步骤快速启动：

```
# PyTorch
pip install torch==1.13.1 torchmetrics==0.11.1 librosa==0.8.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple

# phonemizer pypinyin
apt-get install espeak-ng -i https://pypi.tuna.tsinghua.edu.cn/simple

## OSX: brew install espeak
pip install phonemizer==3.2.1 pypinyin==0.48.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# lhotse 更新到最新版本
# https://github.com/lhotse-speech/lhotse/pull/956
# https://github.com/lhotse-speech/lhotse/pull/960
pip uninstall lhotse
pip uninstall lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# k2
# 在 https://huggingface.co/csukuangfj/k2 中找到正确的版本
# cuda 116 或 cuda 117
pip install https://hf-mirror.com/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
cd -
source ~/.zshrc

# valle
git clone https://github.com/lifeiteng/valle.git
cd valle
pip install -e .
```


## 训练和推理
* #### 英文示例 [examples/libritts/README.md](egs/libritts/README.md)
* #### 中文示例 [examples/aishell1/README.md](egs/aishell1/README.md)
* ### NAR 解码器的前缀模式 0 1 2 4
  **论文第5.1章** "LibriLight 中波形的平均长度为60秒。在训练过程中，我们随机裁剪波形到10秒到20秒之间的随机长度。对

于NAR声学提示令牌，我们从同一句话中选择一个3秒的随机片段波形。"
  * **0**: 没有声学提示令牌
  * **1**: 当前批次话语的随机前缀 **（推荐使用）**
  * **2**: 当前批次话语的随机片段
  * **4**: 如论文所述（由于他们随机裁剪长波形到多个话语，所以同一话语意味着同一长波形中的前一个或后一个话语。）

#### [LibriTTS 演示](https://lifeiteng.github.io/valle/index.html) 在一个具有24G内存的GPU上训练

```
cd examples/libritts

# 第一步 准备数据集
bash prepare.sh --stage -1 --stop-stage 3

# 第二步 在一个具有24GB内存的GPU上训练模型
exp_dir=exp/valle

## 训练 AR 模型
python3 bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir}

## 训练 NAR 模型
cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt  # --start-epoch 3=2+1
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir}

# 第三步 推理
python3 bin/infer.py --output-dir infer/demos \
    --checkpoint=${exp_dir}/best-valid-loss.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \

# 演示推理
https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/run.sh#L68
```
![训练](./docs/images/train.png)

#### 故障排除

* **SummaryWriter 段错误 (核心转储)**
   * 行 `tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")`
   * 修复  [https://github.com/tensorflow/tensorboard/pull/6135/files](https://github.com/tensorflow/tensorboard/pull/6135/files)
   ```
   file=`python  -c 'import site; print(f"{site.getsitepackages()[0]}/tensorboard/summary/writer/event_file_writer.py")'`
   sed -i 's/import tf/import tensorflow_stub as tf/g' $file
   ```

#### 在自定义数据集上训练？
* 准备数据集到 `lhotse manifests`
  * 这里有很多参考 [lhotse/recipes](https://github.com/lhotse-speech/lhotse/tree/master/lhotse/recipes)
* `python3 bin/tokenizer.py ...`
* `python3 bin/trainer.py ...`

## 贡献

* 在多GPU上并行化

 bin/tokenizer.py


## 引用

```bibtex
@article{VALL-E,
  title     = {Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author    = {Chengyi Wang, Sanyuan Chen, Yu Wu,
               Ziqiang Zhang, Long Zhou, Shujie Liu,
               Zhuo Chen, Yanqing Liu, Huaming Wang,
               Jinyu Li, Lei He, Sheng Zhao, Furu Wei},
  year      = {2023},
  eprint    = {2301.02111},
  archivePrefix = {arXiv},
  volume    = {abs/2301.02111},
  url       = {http://arxiv.org/abs/2301.02111},
}
```
