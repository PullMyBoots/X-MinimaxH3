# 长视频创作台 v3

v3 共用一条逐窗口低分辨率 latent 轨迹。在线模式在分叉点复制状态，用少量低分辨率尾步
生成累计预览，同时保留干净的分叉 latent；用户确认全片后，后端把各窗状态组成完整时间轴，
统一执行学习式 latent 放大、重叠时序高清尾步和一次解码。JSON 模式不进入预览支路，也不
解码中间窗口；所有低分辨率窗口完成后自动运行同一个全片高清分支。两种模式的一采与最终
分辨率都必须在项目创建时确定，因为它们共同定义 SelfLift 转换状态。

## 新建项目

先填写项目名称，再选择一种工作方式：

1. **在线创作**：进入工作区后逐窗编辑和提交。
2. **JSON 一键创作**：创建项目后立即把完整脚本提交给持久化后台队列；队列不解码
   中间预览，完成全部低分辨率窗口后自动运行一次全片高清分支。

创建时固定以下轨迹字段：预览与最终分辨率、画面比例、Base/LoRA、总采样步数和高清分支
步数。分叉点等于“总步数 − 高清分支步数”。加速强度不属于锁定轨迹：在线项目的每个
窗口可独立设置自己的加速力度；确认全片后，再为最终高清分支设置一次加速力度。窗口还可
修改新增时长、前窗承接秒数、视觉/音频记忆容量和视觉记忆分辨率。

## JSON 脚本

每个窗口是一个完整 H3 动作提示词，不把同一窗口拆成镜头、对白和动作等多个变量：

```json
{
  "overview": "全片不变的世界、角色、空间和连续性约束。",
  "overall_soundscape": "全片连续声场。",
  "non_diegetic_music": "全片连续配乐，或 N/A。",
  "final_acceleration": 75,
  "references": {},
  "windows": [
    {"prompt": "[Shot 1] 第一个物理窗口。", "seed": 1001, "acceleration": 50},
    {"prompt": "[Shot 1] 严格续写上一窗结尾。", "seed": 1002, "acceleration": 60}
  ]
}
```

窗口对象还可覆盖 `duration_seconds`、`overlap_seconds`、`acceleration`、
`visual_memory_capacity`、`audio_memory_capacity` 和 `visual_memory_resolution`。
顶层 `final_acceleration` 只控制 JSON 队列最后的全片高清分支。

Ref2VA 的首窗参考素材使用本机路径映射：

```json
{
  "references": {
    "Picture 1": {"path": "/absolute/path/scene.png"},
    "Picture 2": "/absolute/path/character.png",
    "Audio 1": {"path": "/absolute/path/voice.wav"}
  },
  "windows": [
    {"prompt": "[Shot 1] Use <Picture 1> and <Picture 2> as fixed references."}
  ]
}
```

图片 ID 必须从 `Picture 1` 连续编号，最多 9 张；音频 ID 必须从 `Audio 1` 连续编号，
最多 3 段。映射只在首窗上传并由后续窗口继承。公开项目数据只返回 ID，不暴露路径。

## 运行规则

- 后端只在前一个窗口成功后提交下一窗。在线窗口成功表示分叉预览与正式断点已保存；
  JSON 窗口成功表示完整正式 SelfLift 轨迹已完成。
- 失败会停止自动队列并保留失败位置。
- 在线模式只允许删除当前尾窗，历史任务卡和已经生成的媒体仍保留。
- 在线模式在工作区顶部播放最近成功的累计预览；每个成功窗口同时拥有低清累计预览
  latent 和该窗口自己的高清 SelfLift 正式断点。
- 在线最终生成恢复所有窗口的正式断点；它不会把累计低清成片重新加噪，也不会重跑低清前缀。
- JSON 模式不生成预览分支，也不解码中间窗口；队列完成低分辨率时间轴后自动执行一次
  全片 SelfLift 升维、重叠时序高清尾步和统一解码。
- 最终分辨率和高清尾步在创建时锁定；在线定稿或 JSON 脚本分别提供高清加速。
- 项目容器可以删除；删除项目不会删除任务中心中的任务记录和媒体。

## REST 入口

| 操作 | 方法与路径 |
|---|---|
| 新建项目 | `POST /api/v1/infinite-projects` |
| 提交在线窗口 | `POST /api/v1/infinite-projects/{id}/windows` |
| 启动 JSON 队列 | `POST /api/v1/infinite-projects/{id}/batch` |
| 停止 JSON 后续提交 | `DELETE /api/v1/infinite-projects/{id}/batch` |
| 删除尾窗 | `DELETE /api/v1/infinite-projects/{id}/windows/last` |
| 在线全片最终采样 / JSON 最终结果兼容查询 | `POST /api/v1/infinite-projects/{id}/final-sampling` |
| 删除项目 | `DELETE /api/v1/infinite-projects/{id}` |

在线 v3 的 `final-sampling` 请求体可使用 `{"acceleration": 75}` 自定义本次高清分支加速。
`resolution` 与 `steps` 仍须和项目轨迹一致；不同值会在进入 GPU 前被拒绝。
JSON v3 不需要调用该接口；旧客户端调用时会以 HTTP 200 返回已经完成的尾窗任务，不会
新增 GPU 计算。v2 旧项目继续使用累计 latent 的脱离式二次采样兼容路径。
