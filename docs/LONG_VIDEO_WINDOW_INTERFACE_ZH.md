# 长视频窗口接口 v2

v2 把长视频创作固定为三层：全片共同设定、分镜、分镜内时间段。每个时间段严格对应一次 H3 推理窗口，服务不会再做隐藏的语义拆分。浏览器入口是 `/static/long-video.html`。

## 用户控制

| 字段 | 范围 | 含义 |
|---|---:|---|
| `overlap_seconds` | 0.25–4 | 后续窗口携带的前一窗口末尾音视频上下文。 |
| `max_window_seconds` | 5–15 | 单次物理窗口上限，包含重叠上下文。 |
| `memory` | 0–100 | 自动长期音视频参考容量；0 关闭长期参考，短期重叠仍保留。 |

时间段的 `duration_seconds` 表示新增内容。后续物理窗口约等于“新增内容＋重叠”，预览会返回 H3 `5 + 17k` 帧网格上的实际长度。

## v2 的状态与相机规则

v2 修正了 v1 的三类权威冲突：

1. `entities.*.identity` 只能描述不随时间变化的身份和外观。位置、持有者、开关和可见性只存在于结构化状态中。
2. `camera_style` 只描述同一分镜不变的镜头规则。每个时间段的 `camera` 只描述本窗口构图、运动和结束机位；续镜窗口不会再次收到完整开场路径。
3. 动作用 `start_seconds` / `end_seconds` 占有明确时间范围，并通过 `state_updates` 事务式修改状态。下一窗口的开始状态由编译器推导，不再依靠互相矛盾的自由文本终态。

```json
{
  "version": 2,
  "overlap_seconds": 1.625,
  "max_window_seconds": 15,
  "memory": 60,
  "story": {
    "overview": "全片稳定风格与共同约束。",
    "entities": {
      "actor": {
        "kind": "character",
        "identity": "人物不随时间变化的长相、服装和声音。"
      },
      "room": {
        "kind": "location",
        "identity": "固定空间身份和布局。"
      },
      "prop": {
        "kind": "prop",
        "identity": "唯一物件不随时间变化的外观。"
      }
    },
    "initial_state": {
      "actor": {"location": "{{room}}", "right_hand": "holds {{prop}}"},
      "room": {"layout": "fixed"},
      "prop": {"location": "{{actor}} right hand", "holder": "{{actor}}"}
    },
    "shots": [
      {
        "id": "shot_1",
        "camera_style": "本分镜不变的镜头、连续性和镜头语言。",
        "overall_soundscape": "环境声和动作声规则。",
        "non_diegetic_music": "N/A",
        "segments": [
          {
            "id": "A",
            "duration_seconds": 10,
            "transition": "opening",
            "camera": {
              "composition": "本窗口开场构图。",
              "motion": "只属于本窗口的运镜。",
              "end": "本窗口结束机位。"
            },
            "beats": [
              {
                "start_seconds": 0,
                "end_seconds": 4,
                "text": "{{actor}} 执行动作。",
                "state_updates": {
                  "prop": {"location": "table", "holder": null}
                }
              }
            ],
            "dialogue": [
              {
                "start_seconds": 4,
                "end_seconds": 6,
                "speaker": "actor",
                "language": "Chinese",
                "text": "原样对白。"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

第一条时间段必须是 `opening`；新分镜首段必须是 `cut`；同一分镜的后续段必须是 `continue`。`cut` 默认保留 0.75 秒建立画面，最少 0.5 秒，动作和对白不能进入这段时间。

编译后的每个窗口都从 `00:00.000` 起算。后续窗口先声明重叠上下文，再把本段所有时间范围加上重叠偏移。历史视觉记忆只拥有身份和场景布局权威；当前结构化状态拥有物体位置、持有者、可见性和开关状态的最高权威。一镜到底的续镜窗口保护完整视觉重叠，但只路由最近状态记忆，避免开场旧机位把相机拉回旧构图。硬切窗口不保护前镜视频 token，完整的不输出重叠区作为可写视觉预热区，使新机位在进入可见时间线前建立完成；音频仍按独立的重叠预热和长期音色规则处理。硬切后仅在不存在可变道具状态冲突时加入开场身份锚；发生状态变化后的切镜只路由最新状态记忆，避免旧画面把物体拉回旧位置。

FL2VA 每窗仍只接收 `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 三项标准字段。Ref2VA 使用六字段参考契约，并继续支持实体身份中的 `<Picture N>` 和 `<Audio N>` 引用。

`memory` 是容量预算，并非质量强度。视觉上限仍为 0–9 帧；单说话人且没有用户声音参考时，FL2VA 自动声音片段为 0.5–3 秒，Ref2VA 为 2–6 秒。FL2VA 的额外媒体记忆仍是实验路径；Ref2VA 的参考布局与训练输入一致。

v1 请求继续可以执行，但公开编辑器和示例已经使用 v2。模板位于 `static/long-video-examples/cafe30.json` 和 `static/long-video-examples/theater30.json`。
