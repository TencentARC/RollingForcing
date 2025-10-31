<p align="center">
<h1 align="center">Rolling Forcing</h1>
<h3 align="center">Autoregressive Long Video Diffusion in Real Time</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://kunhao-liu.github.io/">Kunhao Liu</a><sup>1</sup>
    ·
    <a href="https://wbhu.github.io/">Wenbo Hu</a><sup>2</sup>
    ·
    <a href="https://bluestyle97.github.io/">Jiale Xu</a><sup>2</sup>
    ·
    <a href="http://www.linkedin.com/in/YingShanProfile">Ying Shan</a><sup>2</sup>
    ·
    <a href="https://personal.ntu.edu.sg/shijian.lu/">Shijian Lu</a><sup>1</sup><br>
    <sup>1</sup>Nanyang Technological University <sup>2</sup>ARC Lab, Tencent PCG
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2509.25161"><img src="https://img.shields.io/badge/ArXiv-Paper-brown"></a> <a href="https://kunhao-liu.github.io/Rolling_Forcing_Webpage/"><img src="https://img.shields.io/badge/Project-Webpage-bron"></a> <a href="https://github.com/TencentARC/RollingForcing"><img src="https://img.shields.io/badge/GitHub-Code-blue"></a> <a href="https://huggingface.co/TencentARC/RollingForcing"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow"></a> <a href="https://huggingface.co/spaces/TencentARC/RollingForcing"><img src="https://img.shields.io/badge/HuggingFace-Demo-yellow"></a></h3>
</p>


## 💡 TL;DR: ***REAL-TIME*** streaming generation of ***MULTI-MINUTE*** videos!

https://github.com/user-attachments/assets/7b43ded2-7f29-41a1-8244-a1fc49c418e5

- 🚀 **Real-Time at 16 FPS**:​​ Stream high-quality video directly from text on a ​single GPU.
- 🎬 **Minute-Long Videos**:​​ Generate coherent, multi-minute sequences with ​dramatically reduced drift.
- ​⚙️ **Rolling-Window Strategy**:​​ Denoise frames together in a rolling window for mutual refinement, ​breaking the chain of error accumulation.
- ​🧠 **Long-Term Memory**:​​ The novel ​Attention Sink​ anchors your video, preserving global context over thousands of frames.
- ​🥇 **State-of-the-Art Performance**:​​ Outperforms all comparable open-source models in quality and consistency.


## 🛠️ Installation
Create a conda environment and install dependencies:
```
conda create -n rolling_forcing python=3.10 -y
conda activate rolling_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 🚀 Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt --local-dir .
```

### CLI inference
Example inference script:
```
python inference.py \
    --config_path configs/rolling_forcing_dmd.yaml \
    --output_folder videos/rolling_forcing_dmd \
    --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
    --data_path prompts/example_prompts.txt \
    --num_output_frames 126 \
    --use_ema
```

### Gradio demo (minimal UI)
Run a local web demo that takes a text prompt and shows the generated video.

1) Ensure the Wan base model and checkpoint above are downloaded.
2) Launch the app:
```
python app.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt
```
Then open the printed local URL in your browser.

## 📈 Training
### Download training prompts, ODE-initialized checkpoint, and teacher model
```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B
```

### Train Rolling Forcing on a single machine with 8 GPUs
```
torchrun --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint 127.0.0.1:29500 \
  train.py \
  -- \
  --config_path configs/rolling_forcing_dmd.yaml \
  --logdir logs/rolling_forcing_dmd
```

## 🔖 Citation
If you find this codebase useful for your research, please kindly cite our paper and consider giving this repo a ⭐️.
```bibtex
@article{liu2025rolling,
  title={Rolling Forcing: Autoregressive Long Video Diffusion in Real Time},
  author={Liu, Kunhao and Hu, Wenbo and Xu, Jiale and Shan, Ying and Lu, Shijian},
  journal={arXiv preprint arXiv:2509.25161},
  year={2025}
}
```

## 🙏 Acknowledgements
- [Self Forcing](https://github.com/guandeh17/Self-Forcing): the codebase and algorithm we built upon. Thanks for their wonderful work.
- [Wan](https://github.com/Wan-Video/Wan2.1): the base model we built upon. Thanks for their wonderful work.
