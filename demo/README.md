# AgenticIR 演示模块

本目录只放演示相关代码，不修改或依赖 AgenticIR 的运行入口。

当前提供 `replay`：读取一次已经完成的真实运行，将 `workflow.log`、
`llm_qa.md`、`summary.json` 和中间图片整理为脱敏后的统一事件流，再由
浏览器按真实时间线回放。

## 快速开始

在仓库根目录运行：

```bash
python -m demo.replay build output/099-251207_002922
python -m demo.replay serve --port 8000
```

浏览器访问 <http://127.0.0.1:8000>。

也可以一次完成构建并启动：

```bash
python -m demo.replay all output/099-251207_002922 --port 8000
```

默认生成目录是 `demo/replay/web/data/current/`。它只包含：

- 脱敏后的 `manifest.json`
- 输入、候选结果、中间结果和最终结果图片的副本

原始 Base64、系统提示词、本地绝对路径以及模型配置不会被网页读取。

## 目录结构

```text
demo/
├── README.md
└── replay/
    ├── builder.py       # 历史日志 -> 统一事件协议
    ├── server.py        # 无第三方依赖的本地静态服务
    ├── __main__.py      # build / serve / all 命令
    ├── tests/
    └── web/
        ├── index.html
        ├── styles.css
        ├── app.js
        ├── data-source.js
        └── data/        # 本地生成，不提交
```

## 为实时版预留的边界

页面消费的事件都遵循 `manifest.json` 中的 `events[]` 结构：

```json
{
  "id": "event-005",
  "at": 17.348,
  "source": "depictqa",
  "kind": "tool_assessment",
  "stage": "deraining",
  "title": "DepictQA 评估 MPRNet",
  "summary": "处理后雨退化程度：medium",
  "image": "data/current/assets/01-deraining-mprnet.png",
  "qa_id": "qa-003",
  "data": {}
}
```

后续实时版建议新增 `demo/live/`，由它负责：

1. 启动或连接真实 AgenticIR 任务；
2. 将运行事件转换成同一事件结构；
3. 通过 SSE 的 `/api/events` 按行发送事件；
4. 在 `web/data-source.js` 中增加 `LiveDataSource`。

展示组件无需感知事件来自历史文件还是实时进程。
