# X-MinimaxH3 用户指南

## 1. 启动与进入控制台

完成部署后执行：

```bash
./run.sh
```

浏览器打开 `http://127.0.0.1:8090`。首次进入时服务只启动轻量控制台，不会立刻
把所有 H3 权重装入内存。选择一个模型后才开始装配；页面会显示真实加载阶段
和百分比。需要停止服务时执行 `./stop.sh`，不要直接关闭浏览器代替停止后端。

## 2. 四个模型入口与自动资源路由

用户只选择W4A8/INT8与FL2VA/Ref2VA，内部显存执行器不再作为选项：

| 权重 | 最低显存 | 首遍生成 | H3二采 |
| --- | ---: | --- | --- |
| W4A8 | 8GB | 最高720p×15秒 | 最高1080p |
| INT8 | 16GB | 最高1080p×15秒 | 最高1440p |

同一页面设置完整H3进程树最多能用的主机内存：W4A8可实验性下探到12GiB，INT8
可下探到24GiB；已验证下限仍分别为16GiB和32GiB。滑块上限为操作系统保留6GiB。
Linux通过cgroup v2执行硬限制；后端在模型
加载时一次性推导Pinned覆盖和GPU常驻Block，不会在每个DiT step运行资源路由。

FL2VA 支持纯文本、首帧、尾帧和首尾帧约束。Ref2VA 支持最多9张参考图、3段参考
视频和3段独立参考音频。参考视频总时长不超过15秒；其中的内嵌音轨不会作为音色参考。

切换服务族、权重或内存额度前必须让运行队列为空。Base与当前LoRA装入同一热会话，
可以逐任务切换，不会为 LoRA 再启动一个模型服务。

## 3. 创建任务

控制台提供模块化分镜和自由提示词两种输入方式。自由提示词会逐字传给 H3，不会在
后台自动调用外部大模型。推荐的 H3 文本结构为：

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: N/A
```

有台词时可用 `<d>[Chinese]台词</d>` 或 `<d>[English]dialogue</d>`。Ref2VA 提示词
使用 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 与上传端口编号对应。

## 4. 步数与加速力度

- `sampling_steps`：完整采样轨迹的总长度。Base 可设置5–30步，LoRA可设置4–10步。
- `acceleration`：0–100连续计算预算。0是Dense参考端；75是当前人工审阅的发布质量
  拐点；75以上进入明确允许质量风险的激进区域。

Base 路线会在内部联合安排真实DiT步、预测步和逐步逐层Attention预算。LoRA短轨迹
不会插入未经验证的预测步，加速力度只改变逐步逐层Attention计算分配。构图锚点、
因果交互、声音保护和末端细节保护属于不可关闭的内部约束。

## 5. LoRA

右上角“设置 → LoRA 权重版本”会递归扫描 `models/loras/`。当前内置配置包括：

| LoRA | 任务族 | 推荐步数 |
| --- | --- | ---: |
| Larry Turbo v4-600 EMA | FL2VA / Ref2VA | 4–8，默认6 |
| LightX2V FL2VA Turbo v1.1 768p | FL2VA | 4 |
| LightX2V FL2VA Turbo v1.0 768p | FL2VA | 8 |
| LightX2V Ref2VA Turbo v0.1 | Ref2VA | 4 |

FL2VA 与 Ref2VA 的 LightX2V LoRA 不能互换。切换 LoRA 要求队列为空并重建当前热
会话；控制台会自动把步数移动到该权重的标定点。未知社区 LoRA 只有在原生命名格式
通过头部检查时才会开放，使用前仍需自行确认基础模型、任务族和许可。

## 6. 断点预览

启用断点任务后，只需要选择在正式轨迹的第几步暂停。预览分辨率和预览步数在设置页
统一配置，默认360p、4步。后端在断点处原子保存正式latent和调度历史，然后释放GPU
执行权。预览只是可丢弃的快速分支：

- “继续生成”从同一个正式断点恢复，不重跑前面的步骤；
- “放弃生成”删除断点和预览；
- 清理latent缓存不会删除已经生成的MP4，但会让相关任务失去恢复/二采能力。

## 7. H3 二次采样

先用360p或480p低成本抽卡，满意后在完成任务卡点击“H3 二次采样”。二采使用源任务
保留的干净AV latent、原提示词和原参考条件，而不是把MP4当普通超分输入。

可设置：

- 目标分辨率：由当前资源档位限制；
- 真实DiT步数：1–8；
- 重绘强度：保真、标准、增强、强修复；
- 加速力度：继续使用本项目的Attention预算调度。

重绘越强，允许模型修改身份、细节和运动的幅度越大；它不保证单调提升质量。人脸、
嘴型或精确物体结构敏感时优先使用“保真/标准”和更多步骤，而不是强修复加少步数。

## 8. 任务管理

任务页每秒刷新CPU、主机内存、GPU利用率、显存与功率。等待任务可以排序；运行任务
可以取消；成功任务可以播放、下载、二采或删除。删除记录会一并删除该任务的上传文件、
latent和输出，无法恢复。

## 9. ComfyUI

连接器只调用8090服务，不在ComfyUI内部加载H3，因此不会产生第二份显存占用：

```bash
runtime/venv/bin/python integrations/comfyui/install_local.py /path/to/ComfyUI
integrations/comfyui/start_comfyui.sh /path/to/ComfyUI
```

先启动X-MinimaxH3并在控制台选择服务族，再打开对应示例工作流：

- `integrations/comfyui/example_workflows/H3_Serve_FL2VA_First_Last.json`
- `integrations/comfyui/example_workflows/H3_Serve_Ref2VA_Multi_Reference.json`

## 10. REST API

机器可读契约位于 `GET /openapi.json`。常用接口：

- `GET /healthz`：进程存活；
- `GET /readyz`：当前引擎是否就绪；
- `PUT/DELETE /api/v1/engine`：进入或退出启动器；
- `POST /api/v1/generations`：提交生成；
- `GET /api/v1/jobs`：任务列表；
- `DELETE /api/v1/jobs/{id}`：取消；
- `POST /api/v1/jobs/{id}/resume`：恢复断点；
- `POST /api/v1/jobs/{id}/second-sampling`：二次采样；
- `DELETE /api/v1/cache/latents`：清理latent缓存。

监听非回环地址时必须设置 `H3_SERVE_API_KEY`。服务本身不提供TLS和多租户隔离。

## 11. 常见问题

- 页面一直加载：查看启动终端的模型阶段；运行 `./doctor.sh`。
- WSL读取很慢：保持源码/任务在Windows盘，但让 `./run.sh` 自动使用Linux执行镜像，
  权重优先放到Linux原生盘。
- GPU显存满但功率低：模型可能处于CPU→GPU搬运、Qwen编码或VAE阶段；持续很久才是
  异常。检查任务阶段而不是只看单个瞬时功率。
- 取消没有反应：新版取消在DiT层边界生效；若端口被旧服务占用，先运行 `./stop.sh`。
- 完整模型哈希检查很慢：`./doctor.sh`是快速检查；`./doctor.sh --full`才读取全部权重。
