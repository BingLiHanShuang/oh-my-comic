# oh-my-comic WebUI

一个基于本地部署的LLM、SDXL/qwen edit画图模型运行的自由创作互动连环画系统，支持手机端输入剧情方向、电脑端展示背景图和角色连环画。

## 功能概述

- **手机端（5001 端口）**：展示故事文本，输入下一段剧情方向，实时接收新故事段落
- **电脑端（5002 端口）**：全屏背景图随剧情切换，底部连环画展示角色图，手机滚动自动同步
- **本地 Qwen 模型**：通过 `llama-server` 暴露 OpenAI-compatible API，生成故事文本和绘图提示词
- **Stable Diffusion**：对接 AUTOMATIC1111 WebUI API，生成角色竖图和背景横图
- **显存互斥**：Qwen 生成完文本后自动关闭 llama-server，再启动 SD 生成图片

---

## 项目结构

```
WebuiDisplaywithPhone/
├── app.py                        # 主程序
├── requirements.txt              # Python 依赖
├── .env.example                  # 配置模板（复制为 .env 后填写）
├── README.md
├── data/
│   ├── story.json                # 故事数据（运行时自动生成）
│   └── raw_llm_response.txt      # LLM 原始输出（调试用）
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── mobile.js
│   │   └── desktop.js
│   └── generated/                # 生成的图片（运行时自动创建）
├── templates/
│   ├── mobile.html               # 手机端页面
│   └── desktop.html              # 电脑端页面
└── prompts/
    ├── story_system_prompt.txt   # 系统 prompt（可自定义）
    └── story_user_template.txt   # 用户 prompt 模板（可自定义）
```

---

## 快速开始

### 1. 安装依赖

```bat
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，然后填写你的路径：

```bat
copy .env.example .env
```

编辑 `.env`，至少填写：

```env
LLAMA_SERVER_EXE=C:\path\to\llama-server.exe
QWEN_MODEL_PATH=C:\path\to\qwen3.6-27b-q4_k_m.gguf
SD_API_URL=http://127.0.0.1:7860
```

### 3. 运行

#### Mock 模式（无需任何模型，立即可用）

```bat
python app.py --mock
```

用于测试手机/电脑同步效果，会自动生成占位故事和占位图片。

#### 真实模式（需要 llama-server 和 Stable Diffusion）

先启动 Stable Diffusion WebUI（需要 API 模式）：

```bat
# 在 A1111 目录下
webui-user.bat --api --listen
```

然后启动本项目：

```bat
python app.py
```

脚本会自动启动 llama-server，生成故事后关闭，再调用 SD 生成图片。

#### 只读模式（展示已有故事，不生成新内容）

```bat
python app.py --serve-only
```

#### 手动管理 llama-server（不自动启动）

```bat
python app.py --no-start-llama
```

适用于你已经手动启动了 llama-server 的情况。

---

## 访问地址

启动后终端会显示：

```
============================================================
  Interactive Story WebUI
============================================================
  Mode       : generate
  Mobile     : http://192.168.1.x:5001  (手机访问)
  Desktop    : http://127.0.0.1:5002    (电脑访问)
============================================================
```

- **手机**：连接同一局域网，浏览器打开 `http://电脑局域网IP:5001`
- **电脑**：浏览器打开 `http://127.0.0.1:5002`

> **注意**：如果手机无法访问，请检查 Windows 防火墙是否允许 Python 通过 5001 端口。

---

## 使用流程

1. 手机打开 `http://电脑IP:5001`
2. 电脑打开 `http://127.0.0.1:5002`
3. 手机端输入故事开始方向，点击「生成」
4. 等待 Qwen 生成故事文本（状态栏显示进度）
5. 故事文本出现在手机端后，SD 开始生成图片
6. 背景图生成完成后，电脑端背景自动切换
7. 角色图生成完成后，电脑端连环画自动更新
8. 手机端上下滑动故事文本，电脑端连环画自动同步到对应段落
9. 继续输入下一段剧情方向，重复上述流程

---

## 电脑端操作

- **右上角按钮**：点击「隐藏连环画」可隐藏底部连环画，只看全屏背景图；再次点击恢复
- **连环画布局**：左侧为上一段角色缩略图，中间为当前段大图（最多 2 张并列），右侧为下一段缩略图
- 隐藏/显示状态会保存在浏览器 localStorage，刷新后保持

---

## 自定义 Prompt

编辑 `prompts/` 目录下的文件：

- `story_system_prompt.txt`：系统角色设定，控制模型的整体行为
- `story_user_template.txt`：用户 prompt 模板，支持以下占位符：
  - `{{HISTORY}}`：最近 N 段故事历史（N 由 `STORY_CONTEXT_SEGMENTS` 控制）
  - `{{USER_DIRECTION}}`：用户输入的剧情方向
  - `{{SEGMENT_ID}}`：当前段落编号

---

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_SERVER_EXE` | `llama-server` | llama-server 可执行文件路径 |
| `QWEN_MODEL_PATH` | `/path/to/model.gguf` | Qwen GGUF 模型文件路径 |
| `LLAMA_HOST` | `127.0.0.1` | llama-server 监听地址 |
| `LLAMA_PORT` | `8080` | llama-server 端口 |
| `LLAMA_CTX_SIZE` | `8192` | 上下文长度（tokens） |
| `LLAMA_GPU_LAYERS` | `99` | GPU 卸载层数 |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible API 地址 |
| `OPENAI_MODEL` | `local-qwen` | 模型名称（llama-server 通常不校验） |
| `SD_API_URL` | `http://127.0.0.1:7860` | Stable Diffusion WebUI API 地址 |
| `SD_CHARACTER_WIDTH` | `768` | 角色图宽度（竖向） |
| `SD_CHARACTER_HEIGHT` | `1344` | 角色图高度（竖向） |
| `SD_BACKGROUND_WIDTH` | `1344` | 背景图宽度（横向） |
| `SD_BACKGROUND_HEIGHT` | `768` | 背景图高度（横向） |
| `SD_STEPS` | `25` | SD 生成步数 |
| `SD_CFG_SCALE` | `7.0` | SD CFG 强度 |
| `SD_NEGATIVE_PROMPT` | `（空）` | 固定负面提示词 |
| `STORY_CONTEXT_SEGMENTS` | `6` | 发给 Qwen 的历史段落数 |
| `MOBILE_PORT` | `5001` | 手机端端口 |
| `DESKTOP_PORT` | `5002` | 电脑端端口 |

---

## 数据格式

故事保存在 `data/story.json`，格式如下：

```json
{
  "title": "oh-my-comic",
  "current_index": 2,
  "segments": [
    {
      "id": 0,
      "text": "故事正文...",
      "character_prompts": [
        { "id": "young_deer", "prompt": "..." }
      ],
      "background_prompt": {
        "id": "moonlit_river", "prompt": "..."
      },
      "character_images": [
        { "id": "young_deer", "status": "done", "url": "/static/generated/...", "file": "..." }
      ],
      "background_image": {
        "id": "moonlit_river", "status": "done", "url": "/static/generated/...", "file": "..."
      },
      "status": "text_ready"
    }
  ]
}
```

---

## 常见问题

**Q: 手机无法访问 5001 端口**

A: 检查 Windows Defender 防火墙，允许 Python 通过 5001 端口的入站连接。或者临时关闭防火墙测试。

**Q: llama-server 启动超时**

A: 27B 模型加载可能需要 1-2 分钟。可以增大等待时间（修改 `app.py` 中 `start_llama_server` 的循环次数），或者手动启动 llama-server 后使用 `--no-start-llama` 参数。

**Q: SD 生成图片失败**

A: 确认 AUTOMATIC1111 已以 `--api` 参数启动，且 `SD_API_URL` 配置正确。

**Q: Qwen 返回的 JSON 解析失败**

A: 原始响应保存在 `data/raw_llm_response.txt`，可以查看模型实际输出内容。可以尝试调整 `prompts/story_user_template.txt` 中的格式要求。

**Q: 如何重置故事**

A: 删除 `data/story.json` 和 `static/generated/` 目录下的图片，重新启动程序即可。