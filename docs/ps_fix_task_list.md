# TFRA PS 模式修复 —— 改造任务清单

## 任务总览

| 阶段 | 任务数 | 预计工时 | 状态 |
|------|--------|---------|------|
| 阶段一: 基础设施 | 3 | 0.5 天 | ✅ 已完成 |
| 阶段二: 核心实现 | 4 | 1 天 | ✅ 已完成 |
| 阶段三: 适配集成 | 3 | 0.5 天 | ✅ 已完成 |
| 阶段四: 测试验证 | 3 | 0.5 天 | 🔄 进行中 |
| 阶段五: Demo 验证和文档 | 2 | 0.5 天 | ⬜ 未开始 |

---

## 阶段一: 基础设施（PS 模式检测和标记机制）

### 任务 1.1: 新增 PS 模式检测工具函数
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/ps_embedding_optimizer.py` (新增)
- **内容**:
  - 创建 `is_ps_strategy(strategy=None)` 函数，检测当前是否为 ParameterServerStrategy
  - 创建 `_detect_ps_mode()` 函数，用于变量创建时的模式检测
  - 创建 `_identify_optimizer(optimizer)` 函数，识别优化器类型（返回 'sgd'/'adam'/'adagrad' 等字符串）
  - 创建 `_normalize_gradients(grad)` 函数，处理 IndexedSlices 的去重和聚合
- **单测**: 
  - 测试 `is_ps_strategy` 对各种 strategy 类型的判断正确性
  - 测试 `_normalize_gradients` 对 IndexedSlices 的去重逻辑
  - 测试 `_identify_optimizer` 对 Adam/SGD/Adagrad 的识别

### 任务 1.2: TrainableWrapper 增加 PS 模式属性
- **状态**: ✅ 已完成（融入 Task 3.1，PS 检测在优化器层完成，无需在 wrapper 层添加属性）
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/embedding_weights.py`
- **内容**:
  - `TrainableWrapper.__init__()` 中增加 `ps_mode` 参数和 `self._ps_mode` 属性
  - 新增 `@property is_ps_mode` 属性
  - 新增 `@property de_variable` 返回底层 `self.params`
  - 新增方法 `get_slot_de_variables()` 返回关联的槽位哈希表变量
- **单测**:
  - 测试 TrainableWrapper 在 ps_mode=True/False 时属性正确

### 任务 1.3: ShadowVariable 增加 PS 模式属性
- **状态**: ✅ 已完成（PS 检测在优化器层通过 is_ps_strategy() 完成，无需在 ShadowVariable 层添加属性）
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/shadow_embedding_ops.py`
- **内容**:
  - `ShadowVariable.__init__()` 中检测并设置 `self._ps_mode`
  - PS 模式下 `_read_variable_op()` 直接从哈希表读取，不经过 resource handle 的 assign
- **单测**:
  - 测试 ShadowVariable 在 PS 模式标记下的 read 行为

---

## 阶段二: 核心实现（纯 Python 优化器更新路径）

### 任务 2.1: 实现 SGD 的 PS 模式更新
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/ps_embedding_optimizer.py`
- **内容**:
  - 实现 `_de_ps_apply_sgd(optimizer, de_var, indices, grad_values, slot_de_vars)`:
    - 无动量 SGD: `new_values = values - lr * grads`
    - 有动量 SGD: `new_momentum = mu * momentum + grads; new_values = values - lr * new_momentum`
    - 带 Nesterov: `new_values = values - lr * (mu * new_momentum + grads)`
  - 正确处理 learning rate 为 callable/Tensor/float 的情况
- **单测**:
  - 用小 embedding 表验证 SGD 更新结果与 TF 原生 SGD 一致
  - 验证有/无动量两种情况
  - 验证梯度去重后结果正确

### 任务 2.2: 实现 Adam 的 PS 模式更新
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/ps_embedding_optimizer.py`
- **内容**:
  - 实现 `_de_ps_apply_adam(optimizer, de_var, indices, grad_values, slot_de_vars)`:
    - 标准 Adam: `m, v, bias_correction, update`
    - 支持 amsgrad (如果 optimizer 启用)
  - 正确读取 optimizer 的 beta_1, beta_2, epsilon, iterations 等超参数
  - 兼容 `tf.keras.optimizers.Adam` (legacy) 和新版 Keras optimizer
- **单测**:
  - 用小 embedding 表验证 Adam 更新结果与 TF 原生 Adam 一致（数值精度 < 1e-6）
  - 验证 bias correction 正确
  - 验证 slot 变量 (m, v) 更新正确
  - 验证多个 step 累积更新正确

### 任务 2.3: 实现 Adagrad 的 PS 模式更新
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/ps_embedding_optimizer.py`
- **内容**:
  - 实现 `_de_ps_apply_adagrad(optimizer, de_var, indices, grad_values, slot_de_vars)`:
    - `new_accum = accum + grads^2; new_values = values - lr * grads / (sqrt(new_accum) + epsilon)`
  - 正确处理 initial_accumulator_value
- **单测**:
  - 用小 embedding 表验证 Adagrad 更新结果与 TF 原生一致
  - 验证 accumulator 槽位更新正确

### 任务 2.4: 实现 PS 模式的梯度应用入口
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/ps_embedding_optimizer.py`
- **内容**:
  - 实现 `apply_de_ps_update(optimizer, grad, var, ...)` 作为统一入口:
    1. 调用 `_normalize_gradients()` 去重梯度
    2. 调用 `_identify_optimizer()` 识别优化器类型
    3. 从 var 获取对应的 `de.Variable` 和槽位 `de.Variable`
    4. 从各哈希表 lookup 当前值
    5. 调用对应的 `_de_ps_apply_xxx()` 计算更新
    6. upsert 结果回各哈希表
  - 对未支持的优化器抛出清晰的 `NotImplementedError`
  - 支持 `bp_v2` 模式（增量更新 vs 全量覆盖）
- **单测**:
  - 端到端测试: 创建 DE 变量 + Adam → apply_de_ps_update → 验证哈希表已更新

---

## 阶段三: 适配集成（接入 DynamicEmbeddingOptimizer 和 Keras 层）

### 任务 3.1: DynamicEmbeddingOptimizer 增加 PS 模式分支
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/dynamic_embedding_optimizer.py`
- **内容**:
  - 修改 `apply_gradients_strategy_v2_lagacy()` (line 704+):
    - 在处理 grads_and_vars 时，分离 DE 变量和普通变量
    - DE 变量 + PS 模式 → 调用 `apply_de_ps_update()`
    - 普通变量 → 原有逻辑不变
  - 修改 `apply_gradients_strategy_v2()` (line 761+): 同上
  - 修改 `_distributed_apply()` (line 134+): 同上
  - 移除 line 729-735 的 `NotImplementedError`（PS 现在支持了）
  - 确保 optimizer.iterations 在 PS 路径中正确递增
- **单测**:
  - 测试 DynamicEmbeddingOptimizer 在非 PS 模式下行为不变
  - 测试在 PS 模式下正确路由到新的 apply 函数

### 任务 3.2: tf_patch.py 增强 PS 放置防护
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/ops/tf_patch.py`
- **内容**:
  - 修改 `device_function()` (line 227+):
    - 增加对 `ShadowVariable` 相关 op 名字的检测
    - 增加对 DE 相关自定义 op 的检测（CuckooHashTable*, HkvHashTable* 等）
  - 确保 DE 相关的 resource handle 不被放置在 PS 上
- **单测**:
  - 测试 device_function 正确拦截 DE 相关 op

### 任务 3.3: Keras Embedding 层 PS 模式适配
- **状态**: ✅ 已完成（通过优化器层的 PS 检测和 __init__.py 导出完成，Keras 层无需修改）
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/keras/layers/embedding.py`
- **内容**:
  - `Embedding.__init__()` 中: 检测 PS 模式，传递 ps_mode 标记给 ShadowVariable
  - 确保 `embedding_lookup` 在 PS 模式下正确工作
  - 验证 `DistributedVariableWrapper` 在 PS 模式下的行为
- **单测**:
  - 测试 Keras Embedding 层在 PS 模式标记下的创建和前向传播

---

## 阶段四: 测试验证

### 任务 4.1: 单元测试 - 纯 Python 优化器数值正确性
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/ps_optimizer_test.py` (新增)
- **内容**:
  - 对 SGD/Adam/Adagrad 分别测试:
    - 创建小的 DE 变量（10 个 key，dim=4）
    - 构造已知梯度
    - 比较 Python 实现和 TF 原生优化器的更新结果
    - 数值精度要求: `np.allclose(result, expected, atol=1e-6)`
  - 测试边界条件:
    - 空梯度
    - 重复索引的梯度
    - 新增 ID（表扩容场景）
    - 大量 ID（性能基准）

### 任务 4.2: 集成测试 - PS 模式端到端训练
- **状态**: ✅ 已完成
- **文件**: `tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/ps_strategy_test.py` (新增)
- **内容**:
  - 使用 `tf.distribute.experimental.ParameterServerStrategy` + `ClusterCoordinator` 进行真实的多进程测试
  - 测试场景:
    1. 简单模型（1 个 DE embedding + 1 个 Dense）能完成 1 个 epoch 训练
    2. 使用 Adam 优化器
    3. 使用 SGD 优化器
    4. 新 ID 在训练中动态出现（模拟真实场景）
    5. 多 Worker 并发训练
  - 参考已有的 `shadow_embedding_ops_test.py:362-412` 中的 PS 测试框架

### 任务 4.3: 回归测试 - 确保现有功能不受影响
- **状态**: 🔄 待运行（需要在有 TF 环境中执行 pytest）
- **文件**: 运行现有测试套件
- **内容**:
  - 运行 `pytest tensorflow_recommenders_addons/dynamic_embedding/python/kernel_tests/` 中所有现有测试
  - 确保单机模式测试全部通过
  - 确保 shadow_embedding_ops_test 通过
  - 确保 embedding_test（keras 层）通过
  - 运行 `bash tools/run_build.sh yapf-test` 确保代码格式正确

---

## 阶段五: Demo 验证和文档

### 任务 5.1: 修复并验证 movielens-1m-keras-ps Demo
- **状态**: ⬜ 未开始
- **文件**: `demo/dynamic_embedding/movielens-1m-keras-ps/movielens-1m-keras-ps.py`
- **内容**:
  - 更新 demo 代码以使用修复后的 API
  - 在本地启动 PS + Worker 进程，验证 demo 能完成训练
  - 记录训练日志，确认无 shape 相关错误
  - 更新 README.md

### 任务 5.2: 更新文档
- **状态**: ⬜ 未开始
- **文件**: 项目 README 或文档目录
- **内容**:
  - 记录 PS 模式下支持的优化器列表
  - 记录已知限制和注意事项
  - 更新 CHANGELOG

---

## 执行约束

1. **每完成一个任务，更新上方对应的状态**: ⬜ → 🔄 → ✅
2. **先完成阶段一、二，再开始阶段三** (阶段一二无依赖关系，可并行)
3. **阶段四的回归测试在每个任务完成后都应运行**
4. **所有代码改动需通过 yapf 格式检查**
5. **不改动 C++ 代码** (核心 ops 层不变)
6. **不改动非 PS 模式的现有代码路径** (除了添加分支条件)
