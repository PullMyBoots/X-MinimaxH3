# X-MinimaxH3 部署指南

本文面向 Linux x86-64 与 WSL2。校准平台是 RTX 4090（SM89）、Python 3.10、
PyTorch 2.13.0+cu130 与 CUDA Toolkit 13.3。权重不随源码仓库发布。

## 1. 硬件与磁盘

- NVIDIA SM89 GPU；其他架构尚未作为发布平台验收。
- W4A8要求H3服务至少16GiB额度，INT8要求至少32GiB；控制台另为操作系统
  固定保留6GiB。
- 完整 INT8、W4A8、VAE、文本编码器、二采模型与 LoRA 约需 100GB 磁盘空间。
- WSL2 用户应把热运行镜像和编译缓存保存在 Linux 文件系统；源码可以留在
  Windows 挂载盘，`./run.sh` 会自动同步到 `/root/x-minimaxh3-runtime`。

## 2. 全新安装

先确认显卡驱动可被 WSL/Linux 访问：

```bash
nvidia-smi
git clone <你的仓库地址> X-MinimaxH3
cd X-MinimaxH3
./setup.sh --download-models --accept-model-license
```

`--accept-model-license` 只表示你确认已经阅读并接受各权重发布者的许可；它不会
改变任何模型许可证。安装器会创建隔离环境、固定上游源码版本、下载清单中的
权重，并生成不入库的 `.env.local`。

## 3. 复用现有环境与权重

已有符合要求的环境时无需重复下载：

```bash
./setup.sh \
  --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/vendor \
  --sparse-build-dir /path/to/compiled/sparge
```

其中 `vendor` 必须包含 `MiniMax-H3/` 与 `LightX2V/`。若编译产物位于
`vendor` 同级的标准 `extensions/` 目录，稀疏路径会被自动发现。模型目录结构和哈希由
`models/manifest.json` 定义。快速检查只核对结构；正式发布前建议运行完整检查：

```bash
./doctor.sh
./doctor.sh --full
```

## 4. 启动与停止

```bash
./run.sh
# 浏览器打开 http://127.0.0.1:8090
./stop.sh
```

服务默认仅监听本机。需要局域网访问时，先在 `.env.local` 设置访问密钥，再改变
监听地址：

```bash
export H3_SERVE_API_KEY='请替换为足够长的随机字符串'
export H3_SERVE_HOST=0.0.0.0
```

不要把 `.env.local`、模型权重、用户输入或生成视频提交到 Git。

## 5. 自动资源执行

控制台只提供W4A8/INT8 × FL2VA/Ref2VA四个模型入口。加载权重前自动检测显存，
并在内部选择8GB、16GB或24GB执行器。W4A8最低需要8GB显存，原生生成最高
720p、二采最高1080p；INT8最低需要16GB显存，支持已开放的1080p生成范围与
最高1440p二采。

模型页还要求设置H3服务可用的主机内存硬上限，滑块上限始终为系统保留6GiB。
Linux使用cgroup v2 `memory.max`覆盖服务与所有子进程；若系统没有可写或已委托的
cgroup v2 memory控制器，服务会明确报错，不会伪装成已经限额。超出能力边界的
任务会被拒绝，不会静默更换权重或资源契约。

## 6. LoRA 与二次采样

设置页会扫描模型目录的 `loras/`。FL2VA 与 Ref2VA LoRA 不可互换；系统根据
注册元数据限制任务族。内置清单包含 Larry Turbo 与三个 LightX2V 任务型 LoRA。

H3 二次采样使用 Base 权重、SA Solver、固定调度基础参数，并在同一任务条件上
重新去噪。建议先用低分辨率抽卡，再在任务历史中选择二次采样。用户只调节
1–8 步、四档重绘强度和 0–100 加速力度；时间/空间分块由显存策略自动决定。

## 7. ComfyUI

本项目的 ComfyUI 节点是轻量 HTTP 连接器，不会在 ComfyUI 进程中再次加载 H3。
先启动 8090 服务，再执行：

```bash
./integrations/comfyui/start_comfyui.sh
```

打开 `http://127.0.0.1:8188`，示例工作流位于
`integrations/comfyui/example_workflows/`。断点任务会暂停等待“继续生成”或
“放弃生成”；它不是故障。若不需要人工断点，把节点中的预览模式设为关闭。

## 8. 常见问题

- **首次启动慢**：首次需要加载/映射大权重和编译缓存；把热运行目录与模型放在
  Linux 文件系统能显著减少 WSL 挂载盘元数据开销。
- **8090 被占用**：运行 `./stop.sh`；脚本只终止属于本发布目录的服务，不会误杀
  其他 Python 进程。
- **显存满但功率低**：通常处于权重/激活搬运或 CPU 编码、VAE 阶段。持续发生时
  查看任务卡片的实际阶段并运行 `./doctor.sh`。
- **模型缺失**：检查 `.env.local` 的 `H3_SERVE_MODEL_DIR`，再运行完整 preflight。
- **服务异常退出**：查看 `data/service.log` 与任务错误引用号；提交 Issue 时移除
  提示词、媒体和本机绝对路径。

## 9. 发布验收

```bash
bash -n setup.sh run.sh stop.sh doctor.sh scripts/*.sh
python -m compileall -q h3serve scripts tests
./test.sh
./doctor.sh --full
./scripts/build_release.sh
```

本仓库的已验证环境与测试结果记录在 `VALIDATION.md`。
