# EmbodiedForge

离线机器人操作 episode 自动处理流水线：原始 episode → 带 skill、mask、affordance、3D 几何标注的训练样本。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline（编排层）                          │
│  ingest → segment → semantic → ground → mask → heatmap → 3D    │
│                       ↓ qc ↓ viz ↓ export                       │
└────────────────────────┬────────────────────────────────────────┘
                         │ 仅依赖接口
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Adapters（抽象接口）                           │
│  SemanticLabelerAdapter  GroundingAdapter  SegmentationAdapter   │
│  DepthAdapter                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ 具体实现
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Backends（可插拔实现）                               │
│  mock/  ← 无需 CUDA，直接跑通                                    │
│  qwen_vl/  groundingdino/  sam2/  depth_anything/  ← 可选接入  │
└─────────────────────────────────────────────────────────────────┘
```

**核心设计原则**：Pipeline 代码绝不直接 import 模型 SDK，只依赖 adapter 抽象接口。Backend 通过 YAML 配置由 factory 动态加载。

- **没有 CUDA 也能跑** — `mock` backend 产生符合真实 schema 的合成数据，整条链路可端到端验证
- **不改代码切换模型** — 只需编辑 `configs/demo.yaml` 中的 backend 字段
- **Pipeline 逻辑可独立测试** — 不依赖模型是否可用

## 快速开始

### 1. 安装（基础环境）

```bash
# 推荐使用 uv
uv sync

# 或者使用 pip
pip install -e .
```

基础环境仅安装轻量依赖（pydantic、numpy、pillow、opencv-python、click、pyyaml、matplotlib、rich），不需要 CUDA，不需要 PyTorch。

### 2. 生成示例数据

```bash
eforge generate-sample-data --output-dir examples/sample_episode
```

这会创建一个 30 帧的合成 pick-and-place episode，包含：
- `front_rgb/` 和 `wrist_rgb/` 图像（可视化合成场景）
- `meta.json`（任务指令、FPS、相机内参）
- `state.json`、`action.json`、`gripper.json`、`force_torque.json`（模拟机器人信号）

### 3. 运行完整流水线

```bash
eforge run-all --episode-dir examples/sample_episode --config configs/demo.yaml --output-dir artifacts
```

输出示例：
```
[Stage 0] Ingesting episode...
  -> 30 frames, task: 'pick up the red cup from the table'
[Stage 1] Segmenting into stages...
  -> 3 stages found
     [reach] frames 0-8, keyframes: [4]
     [grasp] frames 9-25, keyframes: [14]
     [lift] frames 26-29, keyframes: [26]
...
[Stage 9] Exporting training sample...
  -> Exported to artifacts/export/sample_episode_001/sample.json

Pipeline complete in 0.2s
```

### 4. 查看输出

```bash
# 完整训练样本 JSON
cat artifacts/export/sample_episode_001/sample.json

# 可视化叠加图（bbox + mask + heatmap + point）
ls artifacts/artifacts/viz/

# 各阶段中间产物
ls artifacts/artifacts/segments.json
ls artifacts/artifacts/semantic_annotations.json
ls artifacts/artifacts/grounding_results.json
ls artifacts/artifacts/masks/
ls artifacts/artifacts/heatmaps/
```

### 5. 单独运行各阶段

```bash
eforge segment --episode-dir examples/sample_episode
eforge annotate-semantic --episode-dir examples/sample_episode
eforge ground --episode-dir examples/sample_episode
eforge segment-mask --episode-dir examples/sample_episode
eforge build-heatmap --episode-dir examples/sample_episode
eforge qc --episode-dir examples/sample_episode
```

### 6. 运行测试

```bash
uv run pytest tests/ -v
```

## 流水线各阶段说明

| 阶段 | CLI 命令 | 说明 |
|-------|------------|-------------|
| 0. Ingest | （自动） | 从磁盘加载原始 episode 到 schema |
| 1. Segment | `eforge segment` | 基于 gripper + velocity + force 信号切分阶段（reach/grasp/lift/place） |
| 2. Semantic | `eforge annotate-semantic` | 标注关键帧的 target_object、target_part、affordance_query |
| 3. Ground | `eforge ground` | 文本查询 → 边界框定位 |
| 4. Mask | `eforge segment-mask` | 从 bbox 提示生成分割 mask |
| 5. Heatmap | `eforge build-heatmap` | 基于 point + mask 生成 affordance 热力图 |
| 6. 3D | （在 run-all 中） | 通过深度图将 2D affordance point 转到 3D 相机坐标系 |
| 7. QC | `eforge qc` | 质量检查（point-in-mask、面积、时序一致性） |
| 8. Viz | （在 run-all 中） | 生成叠加可视化图 |
| 9. Export | （在 run-all 中） | 组装并写出 TrainingSample |

## 配置文件

编辑 `configs/demo.yaml` 来切换 backend：

```yaml
semantic:
  backend: mock           # 可选: mock, qwen_vl_local, qwen_vl_remote
grounding:
  backend: mock           # 可选: mock, qwen_vl_remote, groundingdino_local
segmentation:
  backend: mock           # 可选: mock, sam2_local, sam2_remote
depth:
  backend: mock           # 可选: mock, depth_anything_local
```

所有 backend 都必须通过 `mock` 实现来保证无 CUDA 环境可跑通。

## 输出 Schema

导出的 `sample.json` 包含以下核心字段：

```json
{
  "episode_id": "sample_episode_001",
  "task_instruction": "pick up the red cup from the table",
  "stages": [
    {
      "stage_id": 0,
      "skill_id": "reach",
      "start_frame": 0,
      "end_frame": 8,
      "keyframe_indices": [4]
    }
  ],
  "keyframes": [
    {
      "frame_idx": 4,
      "image_path": "...",
      "skill_id": "reach",
      "target_object": "cup",
      "target_part": "center",
      "affordance_query": "approach direction on cup",
      "object_bbox": {"x1": 224, "y1": 168, "x2": 416, "y2": 312, "confidence": 0.78},
      "part_bbox": {"x1": 224, "y1": 168, "x2": 416, "y2": 312, "confidence": 0.78},
      "object_mask_path": ".../mask_000004.png",
      "affordance_point": {"x": 320, "y": 240},
      "affordance_heatmap_path": ".../heatmap_000004.png",
      "keypoints_3d": [{"x": 0.0, "y": 0.0, "z": 0.65, "label": "affordance_point_3d"}],
      "skill_soft": {"reach": 0.8, "grasp": 0.05, "lift": 0.05, "place": 0.05, "insert": 0.05}
    }
  ],
  "qc": {
    "checks": [...],
    "all_passed": true,
    "score": 1.0
  }
}
```

## 工程结构

```
src/embodiedforge/
├── schemas/           # Pydantic 数据模型（episode、segment、semantic、geometry、sample）
├── adapters/          # 抽象接口 + 工厂
│   ├── base.py        # SemanticLabelerAdapter、GroundingAdapter、SegmentationAdapter、DepthAdapter
│   └── factory.py     # create_adapter() 工厂方法 + 注册表
├── backends/          # 具体实现
│   ├── mock/          # Mock 适配器（无需任何 ML 模型）
│   ├── qwen_vl/       # Qwen-VL 占位实现（接口已定义，待接入真实模型）
│   ├── groundingdino/ # GroundingDINO 占位实现
│   ├── sam2/          # SAM2 占位实现
│   └── depth_anything/# Depth Anything 占位实现
├── pipeline/          # 各阶段实现（纯业务逻辑，不直接调用模型 SDK）
│   ├── ingest.py      # 从磁盘加载 episode
│   ├── segment.py     # 时序阶段切分
│   ├── annotate_semantic.py  # 语义标注
│   ├── ground.py      # 视觉定位
│   ├── segment_mask.py       # Mask 生成
│   ├── build_heatmap.py      # Affordance 热力图
│   ├── annotate_3d.py        # 3D 几何
│   └── run_all.py     # 全流水线编排
├── qc/                # 质量检查
├── export/            # LeRobot 格式导出
├── viz/               # 叠加可视化
└── cli/               # Click CLI 命令
configs/
    demo.yaml          # 默认流水线配置（全部 mock backend）
tests/
    test_schemas.py    # Schema 单元测试
    test_pipeline.py   # Pipeline 集成测试
examples/
    sample_episode/    # 生成的示例数据
```

## 当前状态：Mock vs 真实

| 模块 | 状态 | 说明 |
|-----------|--------|-------|
| Schema 定义 | ✅ 真实 | Pydantic 模型，完全验证 |
| Pipeline 编排 | ✅ 真实 | 所有阶段已实现，可运行 |
| Ingest 数据加载 | ✅ 真实 | 读取真实 episode 目录结构 |
| 时序阶段切分 | ✅ 真实 | 基于 gripper + velocity + force 的启发式切分 |
| 语义标注 | 🟡 Mock | 接口已定义；mock 使用规则模板。真实：实现 Qwen-VL adapter |
| 视觉定位 | 🟡 Mock | 接口已定义；mock 返回图像中心 bbox。远程：Qwen-VL API（✅），本地：GroundingDINO（待实现） |
| 分割 Mask | 🟡 Mock | 接口已定义；mock 绘制椭圆 mask。真实：实现 SAM2 adapter |
| Affordance 热力图 | ✅ 真实 | 高斯热力图计算是真实逻辑 |
| 3D 几何 | 🟡 Mock | 接口已定义；mock 返回渐变深度图。真实：实现 Depth Anything adapter |
| QC 质量检查 | ✅ 真实 | 所有检查逻辑是真实实现的 |
| Export 导出 | ✅ 真实 | 完整 LeRobot 格式 sample 导出 |
| Visualization | ✅ 真实 | OpenCV overlay（bbox + mask + heatmap + point） |
| CLI | ✅ 真实 | 所有命令可运行 |
| 测试 | ✅ 真实 | Schema 单元测试 + Pipeline 集成测试 |

## 接入真实模型

如果需要将 mock 替换为真实模型：

### 方式一：Qwen-VL 语义标注

```bash
pip install embodiedforge[vision]
```

修改 `configs/demo.yaml`：
```yaml
semantic:
  backend: qwen_vl_local
  kwargs:
    model_path: Qwen/Qwen2.5-VL-7B-Instruct
    device: cuda
```

然后实现 `src/embodiedforge/backends/qwen_vl/adapters.py` 中的 `LocalQwenVLAdapter.label()` 方法。

远程 API（DashScope / vLLM）已完整实现，开箱即用：

```yaml
semantic:
  backend: qwen_vl_remote
  kwargs:
    provider: dashscope
    model: qwen3-vl-plus
    api_key: sk-xxxx
```

### 方式二：Qwen-VL 视觉定位（推荐 ⭐）

复用同一个 Qwen-VL API 做视觉定位，无需额外模型：

```yaml
grounding:
  backend: qwen_vl_remote
  confidence_threshold: 0.3
  kwargs:
    provider: dashscope
    model: qwen3-vl-plus
    api_key: sk-xxxx
```

与 semantic 共享同一个 API key 和 endpoint。实现位于 `src/embodiedforge/backends/qwen_vl/adapters.py` 中的 `RemoteQwenVLGroundingAdapter`。

### 方式三：GroundingDINO 视觉定位（本地）

```bash
pip install embodiedforge[grounding]
```

```yaml
grounding:
  backend: groundingdino_local
  kwargs:
    model_path: IDEA-Research/groundingdino-base
    device: cuda
```

⚠ 当前为占位实现，`LocalGroundingDINOAdapter.ground()` 待接入真实模型推理。

### 方式四：SAM2 分割

```bash
pip install embodiedforge[sam2]
```

```yaml
segmentation:
  backend: sam2_local
  kwargs:
    model_cfg: sam2_hiera_l
    checkpoint: ./checkpoints/sam2_hiera_large.pt
    device: cuda
```

### 方式五：远程推理服务

对于运行在独立环境（Docker、远程服务器）中的模型：

```yaml
semantic:
  backend: qwen_vl_remote
  kwargs:
    api_url: http://gpu-server:8000/v1/chat/completions
    api_key: your-key
```

## 环境解耦说明

项目设计保证主工程环境轻量，与重模型推理环境解耦：

```
主工程环境 (uv/pip)          模型推理环境 (conda/Docker)
├── pydantic、numpy、        ├── PyTorch + CUDA
│   pillow、opencv、         ├── SAM2
│   click、pyyaml             ├── GroundingDINO
├── schemas、pipeline、      ├── Qwen-VL
│   qc、export、viz、cli     └── Depth Anything
└── adapter 抽象接口
```

- Pipeline 代码仅从 `adapters/base.py` 导入抽象接口
- Backend 通过 factory 动态加载，不直接 import 模型 SDK
- 如果某个 backend 的依赖缺失，factory 会报清晰错误，可回退到 `mock`

## 后续计划

1. **实现真实的 Qwen-VL adapter** 用于语义标注
2. **实现真实的 SAM2 adapter** 用于视频帧间 mask 传播
3. **添加 insert/peg-in-hole skill** 及轴估计
4. **添加 LeRobot HDF5 导出格式** 作为 JSON 的补充
5. **添加批量处理** 支持多个 episode
6. **添加数据增强** pipeline 阶段
7. **接入真实机器人数据**（如 ROS bag 或 Zarr 格式）

## License

MIT
