# X-MinimaxH3

[English](README.md) · **简体中文**

X-MinimaxH3 是面向单张 NVIDIA SM89 GPU 优化的 MiniMax H3 本地视频生成服务。
中英文 Web 控制台与 REST API 覆盖单视频创作、长视频创作和任务中心，并提供
FL2VA/Ref2VA、Base/LoRA、SelfLift 渐进生成、保留 latent 的全片终采和自动人脸修复。

> 本源码仓库不分发模型权重、用户上传文件、中间 latent 或生成视频。

## 主要能力

- 统一的双端点采样轨迹：一采/终采步数与分辨率使用同一根轴，两个阶段的加速力度
  可以分别调整。
- Base 联合调度器统一安排真实 DiT 计算、预测计算以及逐步逐层 Attention 预算。
- 四个公开模型入口：W4A8/INT8 × FL2VA/Ref2VA；内部8GB、16GB、24GB显存
  执行器由后端自动选择。
- 自动完成资源路由，执行计划在加载模型时编译，不进入 DiT 逐步热循环。
- SelfLift 渐进生成：前段在较小画布运行，以 H3 学习式 3D latent 放大器升维，
  再在高分辨率画布完成尾步；INT8 档位最高可到 1440P。
- 长视频在线创作只生成累计低清预览并保留干净终采分支；JSON 一键创作跳过
  无用的中间预览解码，直接生成高清成片。Ref2VA JSON 每个窗口可声明一套完整且
  独立的 Picture/Audio 参考集；省略时继承上一窗口，服务端本机路径不会公开返回。
- 全片终采沿一条连续的低分辨率 latent 时间轴运行，可选 3–8 秒滑窗、重叠融合、
  音频 token 固定、终采 Sigma 调节和一次统一解码。
- FL2VA 自动人脸修复会对欠清晰的人脸轨迹排序，将目标区域排入方形 Atlas，
  使用四步 H3 Turbo 完成批量修复。
- FL2VA 支持纯文本、首帧、尾帧和首尾帧约束。
- Ref2VA 支持参考图片、参考视频和独立参考音频。
- 内置 Larry Turbo 与三套任务型 LightX2V LoRA 配置。
- 可配置 1–4 步分叉预览，预览支路不会修改保留的正式采样状态。
- 串行 GPU 队列、任务取消、历史记录和每秒硬件监控。
- 可选 ComfyUI HTTP 连接器，不会在 ComfyUI 中重复加载一套 H3。
- 控制台和主要用户文档均支持英文与简体中文。

## AI 提示词写作规范

[`ai-prompt-guides/`](ai-prompt-guides/) 内含六份可独立上传给 ChatGPT 或其他 AI 的中文
TXT 规范，覆盖 FL2VA 与 Ref2VA 的单视频、长视频在线逐窗和长视频 JSON 一键生成。
每份文件都规定了 AI 应询问的信息、H3 写作规则以及可以直接粘贴进工作台的输出格式。
使用时只需上传与当前任务对应的一份文件。

## 视频教程

<p align="center">
  <a href="https://www.bilibili.com/video/BV12PYd6XELW/?spm_id_from=333.1387.upload.video_card.click&amp;vd_source=3e73d78daf8cfe638fe517477471061e">
    <img src="assets/tutorial/x-minimaxh3-tutorial-zh.png" width="860" alt="MiniMax H3 无限生成长视频——X-MinimaxH3 中文教程">
  </a>
</p>

<p align="center">
  <strong>▶ MiniMax H3 无限生成长视频</strong><br>
  <a href="https://www.bilibili.com/video/BV12PYd6XELW/?spm_id_from=333.1387.upload.video_card.click&amp;vd_source=3e73d78daf8cfe638fe517477471061e">前往哔哩哔哩观看完整中文教程</a>
</p>

## 效果实测对比

以下本机实测对比展示了同一素材的 720P 视频直出与 1440P H3 原生二次采样效果。
分割线会在 5 秒、10 秒和 15 秒样例中横向滑动。测试配置为 Intel Core i9（14代）、
128GB 内存、RTX 4090 24GB，使用 INT8 FL2VA。

<p align="center">
  <video controls muted loop playsinline width="860" src="assets/demos/effect-comparison-zh.mp4">
    浏览器不支持内嵌视频播放。
  </video>
</p>

<p align="center">
  <a href="assets/demos/effect-comparison-zh.mp4">▶ 在线观看或下载中文效果对比视频</a>
</p>

## 交流与反馈

国际用户可在 [GitHub Discussions](https://github.com/PullMyBoots/X-MinimaxH3/discussions)
交流安装、硬件兼容性、性能测试、API 接入和生成作品。实时交流可加入
[Telegram 公开群](https://t.me/XMinimaxH3Community)。中文用户也欢迎加入微信群，
或添加作者微信直接反馈。

| 加入 Telegram 社群 | 添加作者 | 加入微信群 |
|:---:|:---:|:---:|
| <a href="https://t.me/XMinimaxH3Community"><img src="assets/community/telegram-community.png" width="260" alt="X-MinimaxH3 Telegram 社群二维码"></a> | <img src="assets/community/wechat-contact.jpg" width="260" alt="作者微信二维码"> | <img src="assets/community/wechat-group.jpg" width="260" alt="X-MinimaxH3 微信交流群二维码"> |
| [打开公开群](https://t.me/XMinimaxH3Community) | 请备注 `X-MinimaxH3` | 群二维码过期后会在这里更新 |

对于能够复现的 Bug 和功能建议，请优先提交到
[GitHub Issues](https://github.com/PullMyBoots/X-MinimaxH3/issues)，方便长期检索问题和解决方案。

## 已验证平台

| 组件 | 已验证配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，SM89 |
| 系统 | Linux x86-64 / WSL2 |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA Runtime | 13.0 |
| 服务编译工具链 | CUDA 13.3 |
| 主机内存 | 建议 64GB 或以上；运行时自动选择常驻策略 |

其他 GPU 架构尚未作为发布平台验收。逻辑 8GB/16GB 路线是在 SM89 上通过
显存硬上限测试的；同容量物理显卡仍需要单独进行设备级验证。

## 快速部署

### 全新安装

下面的命令会创建运行环境、检出固定版本的上游源码，并下载
`models/manifest.json` 声明的全部权重：

```bash
git clone https://github.com/PullMyBoots/X-MinimaxH3.git
cd X-MinimaxH3
./setup.sh --download-models --accept-model-license
./run.sh
```

`--accept-model-license` 仅表示你确认已经阅读并接受各权重发布者的许可证，
不会修改或替代模型本身的许可条款。

### 复用现有环境与权重

```bash
./setup.sh \
  --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/vendor \
  --sparse-build-dir /path/to/compiled/sparge
./run.sh
```

`vendor` 目录必须包含 `MiniMax-H3/` 和 `LightX2V/`。如果兼容的稀疏算子
编译产物已经位于 `vendor` 同级的标准 `extensions/` 目录，可以省略
`--sparse-build-dir`。

浏览器打开 <http://127.0.0.1:8090>。停止服务：

```bash
./stop.sh
```

在 WSL2 中，`./run.sh` 会自动把热运行源码与缓存同步到 Linux 文件系统，避免
服务反复通过 `/mnt/c` 导入大量 Python 文件和访问元数据。

## 安装与发布验收

快速检查、完整权重/源码版本检查和回归测试入口分别是：

```bash
./doctor.sh
./doctor.sh --full
./test.sh
```

当前源码、Web UI、API 契约、长视频/SelfLift、人脸修复、发布包边界和 ComfyUI
连接器均纳入发布回归测试。精确测试数量和解压后复测结果记录在
[VALIDATION.md](VALIDATION.md)；模型哈希与此前 RTX 4090 真实生成矩阵继续作为
历史硬件证据保留。

## 自动资源执行

控制台只显示`W4A8 · FL2VA`、`W4A8 · Ref2VA`、`INT8 · FL2VA`和
`INT8 · Ref2VA`四个入口。加载权重前自动检测显存：

| 检测显存 | 内部执行器 | 可用权重 | 原生首遍生成 | H3原生二次采样 |
|---|---|---|---|---|
| 8–15GB | 8GB | W4A8 | 单个原生窗最高720p × 15秒；支持无感长时请求 | 最高1080p |
| 16–23GB | 16GB | W4A8或INT8 | 两种权重均实验性开放最高1080p原生窗；支持无感长时请求 | W4A8最高1080p；INT8最高1440p |
| 24GB及以上 | 24GB | W4A8或INT8 | 两种权重均开放最高1080p原生窗；支持无感长时请求 | W4A8最高1080p；INT8最高1440p |

Web控制台、REST API和生成节点接受1–300秒。超过单个物理原生窗的请求会自动
使用39帧干净联合音视频前缀和一次最终解码。当前真实成片发布门覆盖480P×30秒、
Base 20步、加速75；更高分辨率和更长时长仍应在目标部署上验收。机制与逐帧验收
证据见[`docs/TRANSPARENT_LONG_HORIZON_2026_09_01.md`](docs/TRANSPARENT_LONG_HORIZON_2026_09_01.md)。

运行时在内部管理 H3、Qwen、VAE 和子进程的常驻关系。资源计划只在模型加载时
完成编译，不会在每个 DiT step 运行 Python 路由。

9月1日资源门以少步高加速完整跑通23/23行：覆盖8/16/24GB矩阵的全部FL2VA
分辨率，并为每个后端补充一条真实单图Ref2VA边界任务。详见[VALIDATION.md](VALIDATION.md)。

超出当前后端能力边界的任务会被明确拒绝，不会静默切换到其他后端。运行服务
返回的分辨率、时长和参考媒体限制才是当前档位的最终有效边界。

设置页可开启 3–8 秒的终采时间滑窗。较短窗口降低单窗延迟和峰值显存，较长窗口
保留更多动作上下文；时间相位对齐、重叠融合、音频 token 固定和显存安全缩窗由
后端自动处理。

## LoRA 配置

| LoRA | 任务族 | 标定步数 |
|---|---|---:|
| Larry Turbo v4-600 EMA | FL2VA / Ref2VA | 4–8，默认6 |
| LightX2V FL2VA Turbo v1.1 768p | FL2VA | 4 |
| LightX2V FL2VA Turbo v1.0 768p | FL2VA | 8 |
| LightX2V Ref2VA Turbo v0.1 | Ref2VA | 4 |

LightX2V 的 FL2VA 与 Ref2VA LoRA 属于不同任务族，不能互换。设置页会递归扫描
当前模型仓库的 `loras/` 目录，并只开放通过兼容性检查的权重。

## ComfyUI

完整说明见[中文 ComfyUI 指南](integrations/comfyui/README.md)或
[English ComfyUI guide](integrations/comfyui/README.en.md)。

先启动X-MinimaxH3并在控制台选择四个模型入口之一，然后执行：

```bash
./integrations/comfyui/start_comfyui.sh
```

打开 <http://127.0.0.1:8188>。示例工作流位于
`integrations/comfyui/example_workflows/`，同时提供中文和英文版本。连接器只调用
同一个 8090 HTTP 服务，
不会在 ComfyUI 进程中额外占用一套 H3 模型显存。

## 目录结构

```text
h3serve/                 Web/API、队列、调度器与 H3 原生运行时
backends/                SM89 算子和经过审计的窄化二进制运行时
static/                  中英文 Web 控制台
ai-prompt-guides/        六种 AI 提示词写作与输出规范
integrations/comfyui/    可选连接器与示例工作流
models/manifest.json     权重来源、大小和 SHA-256 契约
scripts/                 安装、启动、验收和研究工具
tests/                   单元、契约与运行时回归测试
docs/                    用户、部署与架构文档
```

## 详细文档

- [中文用户指南](docs/USER_GUIDE.zh-CN.md)
- [中文部署指南](docs/DEPLOYMENT.zh-CN.md)
- [English user guide](docs/USER_GUIDE.en.md)
- [English deployment guide](docs/DEPLOYMENT.en.md)
- [原生引擎架构](docs/NATIVE_ENGINE_ARCHITECTURE.md)
- [SelfLift 渐进生成](docs/SELFLIFT_PROGRESSIVE_GENERATION.md)
- [长视频创作台 v3](docs/INFINITE_CREATION_STUDIO_V3.md)
- [自动显存路由与主机内存硬预算](docs/AUTOMATIC_RESOURCE_BUDGET_2026_08_31.md)
- [第三方组件声明](THIRD_PARTY_NOTICES.md)
- [发布验收记录](VALIDATION.md)

## 安全说明

服务默认只监听 `127.0.0.1`。绑定到非回环地址前必须设置足够强的
`H3_SERVE_API_KEY`。服务自身不提供 TLS 或多租户隔离；网络部署应使用可信的
反向代理。详见 [SECURITY.md](SECURITY.md)。

## 致谢

X-MinimaxH3 建立在 MiniMax H3 社区多项重要工作的基础上，特别感谢：

- [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)：
  本项目的原生二次采样规划器改造了其时间/空间分块、Overlap 与融合拼接设计，
  并在此基础上实现了脱离 ComfyUI 的运行时、整幅画布准入、自动资源路由、H3
  时间相位对齐和条件缓存复用。
- [Comfyui Minimax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：
  提供学习式 3D latent 放大网络的架构和公开权重，用于构造 H3 二次采样的高分辨率
  初始 latent。
- [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)：其渐进式干净端点
  升维机制为本项目原生 H3 双分辨率轨迹提供了参考；本仓库没有嵌入其运行时代码。
- [SageAttention](https://github.com/thu-ml/SageAttention)：提供量化稠密
  Attention 算子及其实现基础。本项目围绕该基础进一步完成了 H3 专用布局、量化、
  长序列稳定性与统一调度器集成。
- [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum)：
  公开并验证了遮罩式联合音视频前缀机制，为本项目原生无感长时执行器提供了关键参考。
- [ComfyUI-MiniMax-H3-LongMedia](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia)：
  提供长媒体系统实现参考。本项目没有嵌入其ComfyUI猴子补丁运行时，而是独立吸收了
  单模型生命周期、延迟解码和提示词局部时间轴等兼容原则。

以上上游项目不隶属于 X-MinimaxH3，也不对本项目负责；其原始许可证和声明继续
有效，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

当前版本属于**公开可查看源码**，并不自动构成开源许可证授权。项目原创代码目前
保留所有权利；第三方软件和模型工件继续遵循各自许可证。使用或分发前请阅读
[LICENSE](LICENSE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和
[models/manifest.json](models/manifest.json)。
