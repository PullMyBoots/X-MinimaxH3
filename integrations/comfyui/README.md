# H3 Serve Connector for ComfyUI

[English](README.en.md) · **简体中文**

这是 H3 Video Service 正式发行包内的可选 ComfyUI 客户端集成，源码位于
`integrations/comfyui/`。它不会在 ComfyUI 内加载 H3 权重，而是调用
独立运行的 H3 Serve HTTP API，因此服务端模型热态、任务队列、4090优化和超分策略均会保留。

## 本地安装

```bash
python install_local.py /path/to/ComfyUI
```

正式启动顺序固定为：

1. 在H3 Serve目录执行`./scripts/start.sh`，只启动8090控制台。
2. 打开8090，选择FL2VA或Ref2VA服务族，等待模型加载并显示“服务已就绪”。
3. 再打开ComfyUI并运行工作流。

ComfyUI不负责切换FL2VA/Ref2VA服务族；但每个生成节点可以选择原始权重或LoRA，
该选择只是常驻会话内的任务级热开关。未选择服务族、正在加载或未就绪时，连接节点
会明确拒绝提交任务。

重启 ComfyUI 后搜索 `H3 Serve`。最简单的工作流是：

1. `H3 Serve · 连接服务`：默认地址 `http://127.0.0.1:8090`；如服务设置了密钥则填写。
2. `H3 Serve · 简单生成`：连接提示词与可选首/尾帧，选择分辨率、时长、步数和加速力度。
3. 运行生成节点；它会直接保存并预览服务端成片。
4. 如需放大细化，把生成节点的`最终视频`接到`H3 Serve · H3二次采样`。该小节点与控制台一致，只暴露目标分辨率、1–8步、四档重绘强度和加速力度。

示例按输入协议拆成两套，避免把互斥端口放在同一个初学者模板里：

- 中文 `example_workflows/H3_Serve_FL2VA_First_Last.json`：文本、首帧、尾帧或首尾帧生成。
- 中文 `example_workflows/H3_Serve_Ref2VA_Multi_Reference.json`：多图、参考视频与参考音频生成。
- English `example_workflows/H3_Serve_FL2VA_First_Last_EN.json`：英文节点与选项的 FL2VA 工作流。
- English `example_workflows/H3_Serve_Ref2VA_Multi_Reference_EN.json`：英文节点与选项的 Ref2VA 工作流。

中英文工作流调用完全相同的H3 Serve API、推理后端和数值参数；差别仅限节点标题、
输入/输出名称、枚举选项和默认提示词语言，不会改变生成结果或性能。

生成和二采节点都会把服务端成片下载到 ComfyUI 的 `output/h3_serve/` 并直接显示预览，
无需再连接 ComfyUI 的 `SaveVideo` 节点。H3二采要求源任务保留干净AV latent，普通外部MP4不能替代它。
二采固定使用Base权重、Simple调度和SA Solver；不会提供LoRA二采选项。

原始权重与LoRA共用对应服务族的工作流和热会话，在生成节点的`推理路线`中选择。
FL2VA生成节点只暴露`first_frame`/`last_frame`；Ref2VA生成节点直接暴露编号明确的
`Picture 1`～`Picture 9`、`Video 1`～`Video 3`和`Audio 1`～`Audio 3`，两套互斥输入不会出现在同一个节点上。将 ComfyUI 的 `Load Video` 输出直接接到 `Video N`；单段视频为2–15秒、最多三段且合计不超过15秒。其内嵌音轨不参与参考，音色请连接独立的 `Load Audio` 到 `Audio N`。

如果这个ComfyUI只用于调用H3服务，推荐使用随包启动器，使ComfyUI保持CPU模式、不占用
4090显存：

```bash
./start_comfyui.sh /path/to/ComfyUI
```

Ref2VA不再使用素材集合串联节点。将ComfyUI的`Load Image`输出直接接到生成节点的
`Picture N`，将`Load Audio`输出直接接到`Audio N`。编号同时决定提示词中对应的
`<Picture N>`、`<Video N>`与`<Audio N>`。编号决定提示词中对应的引用；例如“保持<Video 1>的骑行轨迹和镜头运动，只将人物服装改为 OL 工装”。

Ref2VA创建工作流不再重复提供`参考图片分辨率`与`参考视频分辨率`；它直接使用控制台保存的
服务端全局设置（初始分别为720P和360P）。
该设置由H3 Serve统一执行：只等比降低像素分辨率，
不裁切、不拉伸、不改变完整构图或视频时长，小素材也不会被放大。Web、REST API和
ComfyUI走的是同一个预处理实现。

FL2VA与Ref2VA都把`prompt`作为普通`STRING`输入端口，不做分镜编译、模板包装或
BGM字段追加。示例工作流用内置`Text (Multiline)`提供一整段最终提示词，连接器逐字
原样交给H3 Serve。若提示词需要H3结构、声音字段或引用标签，由用户全部写在这一个
字符串中。

高级节点只暴露两个推理控制量：`sampling_steps` 是总采样轨迹长度（原始权重5–30，
LoRA 4–10），`acceleration` 是0–100连续加速档位。0表示全部真实步和Dense
Attention；越高表示越小的计算预算。服务自动安排真实/预测位置和逐步逐层Attention，
构图、因果交互、声音与末端细节保护不能在ComfyUI里关闭。LoRA当前只自动调度
Attention，不插入未经人工验收的预测步。连接器轮询服务进度并同步到ComfyUI进度条；中断ComfyUI任务时，
它也会请求服务端取消对应任务。成片下载到 `ComfyUI/output/h3_serve/`。

生成节点的总时长支持1–60秒。超过当前画布单个原生计算窗的请求由服务端自动拆成
受保护的联合音视频latent续写窗，普通H3的`[Shot N] At MM:SS`和中文/英文时间范围
会重映射到局部时间；ComfyUI不需要另一套长视频工作流。窗口间不解码，只在全部
latent完成后解码一次。超过15秒时中途断点/预览暂不开放，节点应保持`预览=关闭`。

FL2VA与Ref2VA生成节点的预览都只有`关闭`/`开启`两个状态。开启后只需设置`预览位置`；
预览分辨率与预览步数使用8090控制台设置里的全局默认。正式轨迹运行到预览位置后会保存断点并停止，
不会继续计算剩余正式步；此时任务释放GPU执行权，快速预览显示在生成节点下方和
`预览视频`输出端口。节点随后显示两个按钮：

- `继续生成`：把同一正式轨迹重新排队，从保留的断点继续到最终视频；
- `放弃生成`：删除服务端任务、预览和断点，并将节点复位为可提交新任务的状态。

关闭预览时不创建断点，节点直接运行到最终视频。原来的独立“断点任务/查看断点预览/
恢复断点任务”节点仍保留，用于需要自行编排任务ID的高级工作流；普通工作流不再需要它们。

断点节点中的数字是正式σ采样位置。LoRA正式任务的全部指定步均为真实Turbo步，
加速档位只调整逐步逐层Attention配额，不插入预测步。`保留中间状态`决定能否恢复，`输出预览`只建立
可丢弃LoRA分支；两者互不替代。断点文件保存在H3 Serve的数据目录，不复制到ComfyUI。

## API兼容性

连接器使用：

- `GET /healthz`
- `GET /api/v1/options`
- `POST /api/v1/generations`
- `GET/DELETE /api/v1/jobs/{id}`
- `DELETE /api/v1/jobs/{id}/record`
- `GET /api/v1/jobs/{id}/preview`
- `POST /api/v1/jobs/{id}/preview/{continue|discard}`
- `POST /api/v1/jobs/{id}/resume`
- `POST /api/v1/jobs/{id}/second-sampling`
- `GET /api/v1/jobs/{id}/video`

H3 Serve 的机器可读契约位于 `http://服务地址/openapi.json`。
