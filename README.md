# oh-my-comic WebUI

一个基于本地部署的 LLM（llama-server）、SDXL 和 Qwen Edit 画图模型运行的自由创作互动连环画系统。

- **手机端（5001 端口）**：展示故事文本和用户消息，输入下一段剧情方向，支持上传图片，支持创作模式开关
- **电脑端（5002 端口）**：全屏背景图随剧情切换，连环画展示角色立绘，手机滚动自动同步

---

## 功能概述

### 普通模式

```
用户输入方向
→ LLM 生成故事 JSON（文本 + 角色 prompt + 背景 prompt）
→ 关闭 LLM，释放显存
→ SDXL 生成背景图和角色图（最多 SDXL_MAX_BATCH_SIZE 张/批）
→ Qwen Edit 生成重复角色图（hybrid 模式，串行）
→ 生图完成后重新预热 LLM
```

### 创作模式

```
开启创作模式（需要 LLM 已就绪）
→ LLM 保持运行
→ 用户可以连续输入多段方向，只生成文本，不生图
→ 关闭创作模式
→ LLM 关闭，释放显存
→ 一次性批量生成所有 pending 图片
→ 生图完成后重新预热 LLM
```

### 图片上传（多模态）

```
用户上传 1 张图片 + 文本
→ LLM（需要 LLAMA_MMPROJ_MODEL）解析图片内容
→ 图片描述 + 用户文本（必须明说当前这段故事只能有图中角色一人）合并为故事输入
→ LLM 生成故事 JSON
→ 只生成背景图，不生成角色图
→ 如果 JSON 中有 character_prompts[0].id，上传图片自动绑定为该角色立绘
```

---

## 项目结构

```
oh-my-comic/
├── app.py                        # 主程序
├── requirements.txt              # Python 依赖
├── .env.example                  # 配置模板（复制为 .env 后填写）
├── README.md
├── qwen_example.py               # LLM / SDXL / Qwen Edit 调用示例
├── data/
│   ├── story.json                # 故事数据（运行时自动生成）
│   ├── raw_llm_response.txt      # LLM 原始输出（调试用）
│   ├── last_error.txt            # 最近一次错误详情（调试用）
│   └── uploads/                  # 用户上传的图片（运行时自动创建）
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

除了 `LLAMA_EXTRA_ARGS` 之外都需要填写。

### 3. 运行

#### Mock 模式（无需任何模型，立即可用）

```bat
python app.py --mock
```

用于测试手机/电脑同步效果，会自动生成占位故事和占位图片。

#### 真实模式

```bat
python app.py
```

程序会自动启动 llama-server，生成故事后关闭，再调用 SDXL 生成图片。

#### 只读模式（展示已有故事，不生成新内容）

```bat
python app.py --serve-only
```

---

## 访问地址

启动后终端会显示：

```
============================================================
  oh-my-comic WebUI
============================================================
  Mode          : generate
  Mobile        : http://192.168.1.x:5001  (手机访问)
  Desktop       : http://127.0.0.1:5002    (电脑访问)
  Prompt rating : general
  Image mode    : sdxl_only
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
4. 状态栏显示 LLM 生成进度（含计时）
5. 故事文本出现在手机端后，SDXL 开始生成图片（状态栏显示批次和计时）
6. 背景图生成完成后，电脑端背景自动切换
7. 角色图生成完成后，电脑端连环画自动更新
8. 手机端上下滑动故事文本，电脑端连环画自动同步到对应段落
9. 继续输入下一段剧情方向，重复上述流程

---

## 手机端功能

### 用户消息显示

每段故事会显示两个气泡：

```
[用户消息气泡（右侧蓝色）]
第 N 段
系统生成的故事正文...
```

### 创作模式开关

手机端顶部有「创作模式」开关：

- **开启条件**：Qwen 语言模型已就绪（状态栏显示"Qwen 语言模型已就绪"）
- **开启后**：LLM 保持运行，用户可以连续写多段文本，不生图
- **关闭后**：LLM 关闭，一次性批量生成所有 pending 图片
- **切换限制**：LLM 生成中或批量生图中时无法切换

### 图片上传

输入栏左侧的 `+` 按钮可以上传一张图片：

- 支持格式：PNG、JPG、JPEG、WebP、GIF
- 上传后会显示预览，可以点击 `✕` 移除
- 提交时图片会和文本一起发送，文本中必须明说当前这段故事只能有图中角色一人
- 需要在 `.env` 中配置 `LLAMA_MMPROJ_MODEL` 才能解析图片内容
- 上传图片的这一轮只生成背景图，不生成角色图
- 如果 LLM 输出了角色 ID，上传图片会自动绑定为该角色的立绘

---

## 电脑端操作

- **右上角按钮**：点击「隐藏连环画」可隐藏连环画，只看全屏背景图；再次点击恢复
- **连环画布局**：
  - 左侧：上一段角色缩略图（叠放，鼠标移上去展开并列）
  - 中间：当前段大图（最多 2 张并列，高度尽可能最大）
  - 右侧：下一段角色缩略图（叠放，鼠标移上去展开并列）
- 隐藏/显示状态会保存在浏览器 localStorage，刷新后保持

---

## 自定义 Prompt

编辑 `prompts/` 目录下的文件：

- `story_system_prompt.txt`：系统角色设定
- `story_user_template.txt`：用户 prompt 模板，支持以下占位符：
  - `{{HISTORY}}`：最近 N 段故事历史（N 由 `STORY_CONTEXT_SEGMENTS` 控制）
  - `{{USER_DIRECTION}}`：用户输入的剧情方向
  - `{{SEGMENT_ID}}`：当前段落编号
  - `{{PROMPT_RATING}}`：图片 rating tag（由 `.env` 中 `PROMPT_RATING` 控制）

---

## 配置说明（`.env`）

### LLM（llama-server）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLAMA_SERVER_EXE` | `""` | llama-server.exe 完整路径 |
| `QWEN_GGUF_MODEL` | `""` | Qwen GGUF 模型文件路径 |
| `LLAMA_HOST` | `127.0.0.1` | llama-server 监听地址 |
| `LLAMA_PORT` | `8080` | llama-server 端口 |
| `LLAMA_CTX_SIZE` | `131072` | 上下文长度（tokens） |
| `LLAMA_SPEC_TYPE` | `ngram-mod` | 推测解码类型 |
| `LLAMA_SPEC_NGRAM_SIZE_N` | `12` | 推测解码 N |
| `LLAMA_SPEC_NGRAM_SIZE_M` | `48` | 推测解码 M |
| `LLAMA_KV_UNIFIED` | `true` | 统一 KV cache |
| `LLAMA_KV_OFFLOAD` | `true` | KV cache offload |
| `LLAMA_MLOCK` | `true` | 锁定内存 |
| `LLAMA_NO_MMAP` | `true` | 禁用 mmap |
| `LLAMA_FLASH_ATTN` | `on` | Flash Attention（`on`/`off`/空） |
| `LLAMA_MMPROJ_MODEL` | `""` | 多模态投影模型路径（图片上传功能需要） |
| `LLAMA_EXTRA_ARGS` | `""` | 额外参数（空格分隔） |

### OpenAI 客户端

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | API 地址 |
| `OPENAI_API_KEY` | `llama-cpp-local` | API Key（本地随意填） |
| `OPENAI_MODEL` | `qwen3.6-35b` | 模型名称 |
| `PROMPT_RATING` | `general` | 图片 rating tag（`general`/`sensitive`/`questionable`/`explicit`） |

### SDXL

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SDXL_MODEL_PATH` | `""` | SDXL .safetensors 模型路径 |
| `SDXL_DTYPE` | `bfloat16` | 精度（`bfloat16`/`float16`/`float32`） |
| `SDXL_MAX_BATCH_SIZE` | `4` | 每批最多同时生成张数 |
| `SDXL_STEPS` | `30` | 生成步数 |
| `SDXL_GUIDANCE_SCALE` | `6.0` | CFG 强度 |
| `SDXL_NEGATIVE_PROMPT` | （内置） | 固定负面提示词 |
| `SDXL_CHARACTER_WIDTH` | `768` | 角色图宽度 |
| `SDXL_CHARACTER_HEIGHT` | `1024` | 角色图高度 |
| `SDXL_BACKGROUND_WIDTH` | `1920` | 背景图宽度 |
| `SDXL_BACKGROUND_HEIGHT` | `1280` | 背景图高度 |

### Qwen Edit（角色一致性）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_EDIT_DIFFUSION_MODEL_PATH` | `""` | 扩散模型路径 |
| `QWEN_EDIT_LLM_PATH` | `""` | LLM 路径 |
| `QWEN_EDIT_VAE_PATH` | `""` | VAE 路径 |
| `QWEN_EDIT_CLIP_VISION_PATH` | `""` | CLIP Vision 路径 |
| `QWEN_EDIT_F2P_LORA_PATH` | `""` | F2P 一致性 LORA 所在目录的路径（只允许放F2P.safetensors一个文件） |
| `QWEN_EDIT_WIDTH` | `768` | 输出宽度 |
| `QWEN_EDIT_HEIGHT` | `1024` | 输出高度 |
| `QWEN_EDIT_CFG_SCALE` | `1` | CFG 强度 |
| `QWEN_EDIT_SAMPLE_STEPS` | `8` | 采样步数 |
| `QWEN_EDIT_SAMPLE_METHOD` | `euler_a` | 采样方法 |
| `QWEN_EDIT_SCHEDULER` | `simple` | 调度器 |
| `QWEN_EDIT_SEED` | `-1` | 随机种子（-1 随机） |

### 图片生成模式

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IMAGE_GENERATION_MODE` | `sdxl_only` | `sdxl_only`：全部用 SDXL；`hybrid`：首次角色 SDXL，重复角色 Qwen Edit |

### 故事生成

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STORY_CONTEXT_SEGMENTS` | `6` | 发给 LLM 的历史段落数 |

### 端口

| 变量 | 默认值 | 说明 |
|------|--------|------|
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
      "user_text": "用户输入的剧情方向",
      "text": "故事正文...",
      "character_prompts": [
        { "id": "amiya", "prompt": "1girl, full_body, ..." }
      ],
      "background_prompt": {
        "id": "forest", "prompt": "general, forest, ..."
      },
      "character_images": [
        { "id": "amiya", "status": "done", "url": "/static/generated/...", "file": "..." }
      ],
      "background_image": {
        "id": "forest", "status": "done", "url": "/static/generated/...", "file": "..."
      },
      "status": "text_ready"
    }
  ]
}
```

---

## 如何重置故事

删除以下文件，然后重启程序：

```bat
del data\story.json
del data\raw_llm_response.txt
del data\last_error.txt
del /q static\generated\*.png
del /q static\generated\*.jpg
del /q static\generated\*.webp
del /q data\uploads\*
```

---

## 常见问题

**Q: 手机无法访问 5001 端口**

A: 检查 Windows Defender 防火墙，允许 Python 通过 5001 端口的入站连接。

**Q: llama-server 启动超时**

A: 27B 模型加载可能需要 1-2 分钟。状态栏会显示等待计时。可以增大 `LLAMA_CTX_SIZE` 以外的参数来加快加载。

**Q: Qwen 返回的 JSON 解析失败**

A: 原始响应保存在 `data/raw_llm_response.txt`，错误详情在 `data/last_error.txt`。

**Q: 图片上传后提示"图片解析不可用"**

A: 需要在 `.env` 中配置 `LLAMA_MMPROJ_MODEL`，并且你的 llama-server 版本需要支持多模态。

**Q: 创作模式开关是灰色的**

A: 需要等待 Qwen 语言模型预热完成（状态栏显示"Qwen 语言模型已就绪"）后才能开启。

**Q: 上传图片后，该角色后续会保持一致性吗？**

A: 会。上传图片绑定的角色 id 会被加入 `force_qwen_edit_character_ids` 列表，保存在 `data/story.json` 中。即使 `IMAGE_GENERATION_MODE=sdxl_only`，该角色后续出现时也会强制使用 Qwen Edit 进行图生图，保持外观一致。其他角色仍然遵循 `IMAGE_GENERATION_MODE` 设置。

**Q: 如何只用 SDXL 不用 Qwen Edit**

A: 在 `.env` 中设置：
```env
IMAGE_GENERATION_MODE=sdxl_only
```
注意：如果你上传过角色参考图，该角色仍会强制使用 Qwen Edit，不受此设置影响。

**Q: 如何调整图片 rating**

A: 在 `.env` 中设置：
```env
PROMPT_RATING=sensitive
```
支持：`general` / `sensitive` / `questionable` / `explicit`

**Q: 电脑端立绘太小**

A: 立绘高度会自动按浏览器窗口高度的 86% 计算。如果需要调整，修改 `static/js/desktop.js` 中的 `CENTER_HEIGHT_RATIO`。

**Q: 如何切换角色图分辨率（如 1080x1920）**

A: 修改 `.env`：
```env
SDXL_CHARACTER_WIDTH=1080
SDXL_CHARACTER_HEIGHT=1920
```
同时修改 `static/js/desktop.js` 中的：
```js
var CHARACTER_ASPECT = 1080 / 1920;