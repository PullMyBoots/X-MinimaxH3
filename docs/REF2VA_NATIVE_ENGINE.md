# Ref2VA 独立引擎（v0.5.0）

## 发布边界

`scripts/start-reference.sh` 启动固定的 Ref2VA 服务。它和高保真、Turbo 启动器
共享 Web/API、Qwen3-VL、Video VAE、Audio VAE、采样器、SM89 ConvRot kernel、
Block Offload 与封装器，但加载独立的 Ref2VA DiT 权重。运行热路径不启动或调用
ComfyUI。

支持1至9张静态参考图和1至3段参考视频；视频单段2–15秒且总时长不超过15秒。
参考视频的内嵌音轨明确丢弃；独立参考音频已接入，最多三段。

## 权重与来源

- 文件：`models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- 仓库：`Comfy-Org/MiniMax-H3`
- revision：`014cd40f7e177756c6b2473c0d93b1c89a790dd2`
- 大小：20,970,379,616 bytes
- SHA-256：`9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779`

开发工作区允许 `models/` 中的软链接指向已下载的同一文件；发行下载器可用：

```bash
python scripts/download_models.py --accept-model-license --reference-only
```

## 条件协议

1. 图片保持宽高比，只在面积超过目标画布时缩小，宽高舍入到32的倍数。
2. 视频只解码画面，重采样到24fps，裁到输出帧数并向下对齐`17n+5`网格。
3. 同一份预处理媒体同时进入Qwen3-VL和Video VAE，避免重复解码和语义错位。
4. Qwen图片使用`<Picture i>:`；视频每12帧采样一次并以成对帧构成带时间戳的`<Video i>:`块；独立音频在presentation中使用`<Audio i>:`标签，但原始波形不送入Qwen。
5. 独立参考音频经PyAV重采样到32kHz双声道，再由Audio-VAE编码为`[1,32,2,T]`条件latent。
6. DiT顺序是`presentation → reference images/videos → reference audio → target audio → target video`。
7. 图片推进一个旋转时间单位，视频按完整latent时间跨度推进，音频按latent帧数推进；所有参考token均使用不可更新mask。
7. 参考条件噪声跟随请求seed；目标音视频仍按所选H3调度生成。

## 速度/质量控制

Ref2VA使用原始权重路线的20点调度，产品默认9次真实计算、11次forecast；高级模式
仍可设置真实步数。20/0保留为最完整的质量参考点。FL2VA的9/11人工验收不能直接
外推到Ref2VA，因此复杂多参考任务仍需专门的成片审核。

## 真实烟测（RTX 4090）

2026-08-13使用1张参考图、864×480、73帧、3.042秒、20/0、seed 82431运行成功：

- 服务冷启动/模型装配：68.393秒（compact内存档）；
- 热服务中的首次请求：148.665秒；
- 输出：H.264 24fps + AAC 32kHz双声道；
- 六帧门控：参考人物、服装、石室、地图与雨夜连续保留，动作从入室过渡到指图，
  未见彩色斑块、散光重影、结构崩坏或VAE接缝。

该烟测只证明单参考图端到端链路与基本视觉正确性。参考视频本轮只完成合成媒体的
解码、重采样、Qwen时间块、Video VAE接口、DiT布局和API测试，按要求没有执行耗时
的真实H3生成；因此参考视频仍属于“已接入、待真实成片终审”。
