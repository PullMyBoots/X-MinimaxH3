# 两服务族与分叉预览设计

## 产品边界

发行版只保留两个需要装卸大模型的服务族：

1. `first_last`：FL2VA，支持文本、首帧、尾帧、首尾帧条件；
2. `reference`：Ref2VA，支持多图、参考视频和独立参考音频。

每个服务族的CPU热会话同时装配基础INT8 ConvRot图和Larry LoRA低秩参数。请求通过
`model_variant=base|lora`控制259个驻留低秩更新是否参与计算。开关是Python执行策略，
不会复制权重、不会重建模型图，也不会触发服务族切换。Base关闭LoRA时直接返回量化
基础线性层结果；CPU测试要求与未应用LoRA的基础路径逐元素完全一致。

只有`first_last`与`reference`互换时才释放当前热会话并加载另一份基础checkpoint。

## 分叉预览状态机

`preview_mode=pause`的任务状态为：

```text
queued -> running -> awaiting_preview --continue--> running -> succeeded
                                \--discard-----> cancelled
```

在指定实际计算步之后，运行时克隆当前视频latent、音频latent和sigma位置。预览分支用
1–3个真实去噪步从当前sigma快速走到0，再单独执行VAE解码与AV封装。它不复用或修改
正式轨迹的forecast历史、采样器状态或latents。

预览完成后，DiT权重已回到CPU热态；正式latents保留在GPU。选择`continue`时重新激活
同一个Transformer，沿原采样schedule继续；选择`discard`则取消任务。人的等待时间会
计入任务墙钟时间，但不属于模型推理耗时。

`preview_mode=auto`使用同一隔离分支生成预览，但不等待决定；适合API批处理留样。

## 公共接口

- 生成：`POST /api/v1/generations`
- 预览：`GET /api/v1/jobs/{id}/preview`
- 继续：`POST /api/v1/jobs/{id}/preview/continue`
- 放弃：`POST /api/v1/jobs/{id}/preview/discard`

Web控制台直接显示预览和决定按钮。ComfyUI节点暴露`推理路线`和`分叉预览`；选择
“在Web控制台抽卡”时，ComfyUI运行保持等待，由8090控制台作决定。

## 质量与性能边界

分叉预览是用于判断主体、构图、动作方向和大致叙事的低成本代理成片，不承诺最终清晰度、
口型或音色完全一致。`preview_branch_steps`越小，预览越快但拖影越明显。正式输出仍使用
用户所选质量档和原schedule；预览分支不得成为正式轨迹的输入。

发布验收至少覆盖：同服务族Base→LoRA→Base不重建session；关闭LoRA的Base路径精确
一致；pause的continue和discard状态机；预览文件可下载；正式成片在启用/关闭预览时
使用同seed得到一致结果。

## 2026-08-16 RTX 4090 实测

环境为单卡RTX 4090 24GB、`fullspeed`主机内存模式、FL2VA热会话。联合Base+LoRA
session冷装配为59.801秒。之后连续执行：

| 请求 | 规格 | 结果 | 生成耗时 |
|---|---|---|---:|
| Base + pause预览 + continue | 640×352、22帧、9/11 | 成功；先进入`awaiting_preview` | 28.286秒（包含人工等待） |
| LoRA | 同规格、6步 | 成功 | 8.066秒 |
| Base | 同规格、9/11 | 成功 | 11.644秒 |

三次请求后`warm_state.startup_seconds`仍为59.801秒，证明Base→LoRA→Base没有重建
session。预览MP4为0.917秒，四帧视觉检查无色斑、坍缩或直接中间latent式重影。

为检查正式轨迹隔离，又以相同prompt、seed 81601和规格关闭预览重跑。启用预览后继续
与关闭预览的最终MP4 SHA-256均为：

```text
7d332f5f736334d06f610d74a71794187949991994acbd19fcaf3e8ac00e020e
```

文件逐字节`cmp`相等，直接证明本次实现的预览分支没有污染正式轨迹。实测成片位于
`output/5addaea1-ca86-472a-95a9-61464bd957fd.mp4`，预览位于同名
`.preview.mp4`；无预览对照为`output/cae41f16-7e8c-427f-aa64-2d3e48ccea4a.mp4`。

Ref2VA联合session冷装配为58.531秒；随后以同一`<Picture 1>`连续执行Base和LoRA，
分别为20.972秒与7.831秒。两次生成后`warm_state`仍为同一个`reference` session，
四帧视觉检查均保持参考女孩身份且无色斑/坍缩。对应成片：

- `output/e5e62ef2-879f-403e-844f-2c751851e2fe.mp4`（Base）；
- `output/c3b6ebe2-1597-4ef9-9db5-5ebb75505ab6.mp4`（LoRA）。
