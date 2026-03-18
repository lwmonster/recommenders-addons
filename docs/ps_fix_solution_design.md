# TFRA PS 模式修复 —— 详细解决方案设计

## 一、方案对比评估

### 方案 A: 在 PS 模式下绕过原生优化器算子，直接用哈希表操作完成梯度更新

| 维度 | 评估 |
|------|------|
| 可行性 | ✅ 高 — 纯 Python 层面改动，无需 C++ |
| 复杂度 | 中等 — 需为每种优化器实现纯 Python 更新逻辑 |
| 性能 | 略有额外开销（多一次 lookup），但可接受 |
| 风险 | 低 — 仅影响 PS 模式下的 DE 变量 |

### 方案 B: 确保 TrainableWrapper 的 shape 在 step 内一致

| 维度 | 评估 |
|------|------|
| 可行性 | ❌ 低 — PS 异步执行下无法保证时序 |
| 复杂度 | 高 — 需要深入修改 TF 运行时行为 |
| 性能 | 好（如果能工作） |
| 风险 | 极高 — 可能引入更难调试的竞态条件 |

### 方案 C: 自定义 PS 感知的优化器包装器（高层拦截）

| 维度 | 评估 |
|------|------|
| 可行性 | ✅ 高 — 在已有 DynamicEmbeddingOptimizer 基础上扩展 |
| 复杂度 | 中等 |
| 性能 | 与方案 A 相同 |
| 风险 | 低 |

### 方案 D: 将 DE 变量从 TF 的 PS 变量分片中解耦

| 维度 | 评估 |
|------|------|
| 可行性 | ✅ 高 — 作为辅助加固措施 |
| 复杂度 | 低-中 |
| 性能 | 中性/略好 |
| 风险 | 低 |

### 最终选择: 方案 C + A + D 组合

- **方案 C**（主体）: 在 `DynamicEmbeddingOptimizer` 中增加 PS 模式分支
- **方案 A**（核心）: PS 分支中绕过原生算子，用纯 Python 实现优化器更新逻辑
- **方案 D**（加固）: 确保 TrainableWrapper/ShadowVariable 不被 PS strategy 当作普通变量分片

这也是 Oracle 架构分析推荐的方案，与 DeepRec 项目的做法一致（自定义优化器路径 + 自定义变量类型）。

## 二、整体架构设计

### 2.1 PS 模式下的新数据流

```
修复前 (崩溃):
  embedding_lookup → TrainableWrapper → TF 原生 _resource_apply_sparse
                                         → ResourceScatterAdd (PS 上执行)
                                         → ❌ shape 不一致，崩溃

修复后 (正常):
  embedding_lookup → TrainableWrapper → PS 模式检测
                                         → _apply_de_ps_update (Worker 上执行)
                                         → 纯 Python 计算更新值
                                         → de_var.upsert(ids, updated_values) (直接写哈希表)
                                         → ✅ 绕过 ResourceScatterAdd
```

### 2.2 详细数据流（以 Adam 为例）

```
PS 模式下的 Adam 更新流程:

1. 前向传播 (Worker):
   ids = [0, 1, ..., 62]  # 本 batch 的特征 ID
   values = de_var.lookup(ids)  # 从动态哈希表查询
   embeddings = f(values)  # 前向计算
   loss = loss_fn(embeddings, labels)

2. 反向传播 (Worker):
   grads = tape.gradient(loss, trainable_wrapper)
   # grads 是 IndexedSlices: indices=[0,1,...,62], values=[63, dim]

3. PS 感知的梯度应用 (Worker，新增逻辑):
   a. 去重 & 聚合梯度索引:
      unique_indices, idx = tf.unique(grads.indices)
      unique_grads = tf.math.unsorted_segment_sum(grads.values, idx, len(unique_indices))

   b. 从哈希表读取当前参数和槽位状态:
      current_values = de_var.lookup(unique_indices)         # 主表
      current_m = slot_m_de_var.lookup(unique_indices)       # 一阶动量
      current_v = slot_v_de_var.lookup(unique_indices)       # 二阶动量

   c. 纯 Python 计算 Adam 更新 (使用优化器超参数):
      t = optimizer.iterations + 1
      lr = optimizer.learning_rate
      beta_1, beta_2, epsilon = optimizer.beta_1, optimizer.beta_2, optimizer.epsilon
      
      new_m = beta_1 * current_m + (1 - beta_1) * unique_grads
      new_v = beta_2 * current_v + (1 - beta_2) * unique_grads ** 2
      m_hat = new_m / (1 - beta_1 ** t)
      v_hat = new_v / (1 - beta_2 ** t)
      updated_values = current_values - lr * m_hat / (sqrt(v_hat) + epsilon)

   d. 写回哈希表 (原子操作):
      de_var.upsert(unique_indices, updated_values)          # 更新主表
      slot_m_de_var.upsert(unique_indices, new_m)            # 更新一阶动量
      slot_v_de_var.upsert(unique_indices, new_v)            # 更新二阶动量

4. 非 DE 变量 (Dense 层等) 继续使用原生优化器路径，不受影响
```

### 2.3 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                   DynamicEmbeddingOptimizer                       │
│                                                                  │
│  apply_gradients(grads_and_vars)                                 │
│    │                                                             │
│    ├─ 非 DE 变量 ──→ 原生 optimizer.apply_gradients() (不变)     │
│    │                                                             │
│    └─ DE 变量 (TrainableWrapper/ShadowVariable)                  │
│         │                                                        │
│         ├─ 非 PS 模式 ──→ 现有逻辑 (_resource_apply_* + update_op)│
│         │                                                        │
│         └─ PS 模式 ──→ _apply_de_ps_update() [新增]              │
│              │                                                   │
│              ├─ _de_ps_apply_sgd()     [新增]                    │
│              ├─ _de_ps_apply_adam()    [新增]                    │
│              ├─ _de_ps_apply_adagrad() [新增]                    │
│              └─ _de_ps_apply_xxx()    [按需扩展]                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    de.Variable (哈希表)                           │
│  - lookup(ids) → 读取                                            │
│  - upsert(ids, values) → 写入                                   │
│  - 主表 + 槽位表 统一通过哈希表操作更新                            │
│  - 无需经过 ResourceVariable handle → 无 shape 冲突              │
└──────────────────────────────────────────────────────────────────┘
```

## 三、需要修改的文件清单

### 3.1 核心修改文件

| 文件 | 修改内容 | 优先级 |
|------|---------|-------|
| `dynamic_embedding_optimizer.py` | 新增 PS 模式分支、纯 Python 优化器更新函数 | P0 |
| `embedding_weights.py` | 新增 `_is_de_ps_mode` 属性标记；优化 PS 模式下的 read/write 路径 | P0 |
| `shadow_embedding_ops.py` | ShadowVariable 在 PS 模式下的适配 | P0 |
| `tf_patch.py` | 增强 PS 放置防护，从名字启发式改为属性标记 | P1 |

### 3.2 辅助修改文件

| 文件 | 修改内容 | 优先级 |
|------|---------|-------|
| `dynamic_embedding_variable.py` | `embedding_lookup` 中增加 PS 模式标记传递 | P1 |
| `distributed_embedding_variable.py` | 确保 DistributedVariableWrapper 在 PS 模式下正确工作 | P1 |
| `keras/layers/embedding.py` | Keras 层在 PS 模式下的适配 | P1 |

### 3.3 新增文件

| 文件 | 内容 | 优先级 |
|------|------|-------|
| `ps_embedding_optimizer.py` (新增) | PS 模式专用的纯 Python 优化器更新逻辑 | P0 |

### 3.4 测试文件

| 文件 | 内容 | 优先级 |
|------|------|-------|
| `kernel_tests/ps_strategy_test.py` (新增) | PS 模式下的集成测试 | P0 |

## 四、核心代码变更详细设计

### 4.1 新增文件: `ps_embedding_optimizer.py`

```python
# 位置: tensorflow_recommenders_addons/dynamic_embedding/python/ops/ps_embedding_optimizer.py

"""
PS 模式下动态 Embedding 的纯 Python 优化器更新实现。
绕过 TF 原生的 ResourceScatterAdd 等算子，直接操作哈希表。
"""

def is_ps_strategy(strategy):
    """判断当前是否在 ParameterServerStrategy 下运行"""
    ...

def apply_de_ps_update(optimizer, de_var, grad, slot_vars, slot_de_vars):
    """
    PS 模式下的 DE 变量梯度应用入口。
    
    Args:
        optimizer: TF 优化器实例
        de_var: de.Variable, 主 embedding 哈希表
        grad: IndexedSlices 或 dense Tensor, 梯度
        slot_vars: dict, {slot_name: TrainableWrapper}, 优化器槽位的 wrapper
        slot_de_vars: dict, {slot_name: de.Variable}, 优化器槽位的哈希表
    """
    # 1. 标准化梯度 (去重 + 聚合)
    indices, grad_values = _normalize_gradients(grad)
    
    # 2. 从哈希表读取当前状态
    current_values = de_var.lookup(indices)
    current_slots = {name: sv.lookup(indices) for name, sv in slot_de_vars.items()}
    
    # 3. 根据优化器类型计算更新
    optimizer_type = _identify_optimizer(optimizer)
    updated_values, updated_slots = _compute_update(
        optimizer_type, optimizer, current_values, grad_values, current_slots)
    
    # 4. 写回哈希表 (先写槽位，再写主表)
    update_ops = []
    for name, sv in slot_de_vars.items():
        update_ops.append(sv.upsert(indices, updated_slots[name]))
    with tf.control_dependencies(update_ops):
        main_update = de_var.upsert(indices, updated_values)
    
    return main_update

def _normalize_gradients(grad):
    """去重并聚合 IndexedSlices 梯度"""
    if isinstance(grad, tf.IndexedSlices):
        unique_indices, idx = tf.unique(grad.indices)
        unique_values = tf.math.unsorted_segment_sum(
            grad.values, idx, tf.shape(unique_indices)[0])
        return unique_indices, unique_values
    else:
        # Dense gradient 的情况
        return None, grad

def _compute_update(optimizer_type, optimizer, values, grads, slots):
    """根据优化器类型计算参数更新"""
    if optimizer_type == 'sgd':
        return _de_ps_apply_sgd(optimizer, values, grads, slots)
    elif optimizer_type == 'adam':
        return _de_ps_apply_adam(optimizer, values, grads, slots)
    elif optimizer_type == 'adagrad':
        return _de_ps_apply_adagrad(optimizer, values, grads, slots)
    else:
        raise NotImplementedError(
            f"Optimizer '{optimizer_type}' is not yet supported in PS mode "
            f"with dynamic embedding. Supported: sgd, adam, adagrad")

def _de_ps_apply_sgd(optimizer, values, grads, slots):
    """SGD 纯 Python 实现"""
    lr = optimizer.learning_rate
    # SGD with momentum
    if 'momentum' in slots:
        momentum_coeff = optimizer.momentum
        new_momentum = momentum_coeff * slots['momentum'] + grads
        updated_values = values - lr * new_momentum
        return updated_values, {'momentum': new_momentum}
    else:
        updated_values = values - lr * grads
        return updated_values, {}

def _de_ps_apply_adam(optimizer, values, grads, slots):
    """Adam 纯 Python 实现"""
    lr = optimizer.learning_rate
    beta_1 = optimizer.beta_1
    beta_2 = optimizer.beta_2
    epsilon = optimizer.epsilon
    t = tf.cast(optimizer.iterations + 1, values.dtype)
    
    m = slots.get('m', tf.zeros_like(values))
    v = slots.get('v', tf.zeros_like(values))
    
    new_m = beta_1 * m + (1.0 - beta_1) * grads
    new_v = beta_2 * v + (1.0 - beta_2) * tf.square(grads)
    
    m_hat = new_m / (1.0 - tf.pow(beta_1, t))
    v_hat = new_v / (1.0 - tf.pow(beta_2, t))
    
    updated_values = values - lr * m_hat / (tf.sqrt(v_hat) + epsilon)
    
    return updated_values, {'m': new_m, 'v': new_v}

def _de_ps_apply_adagrad(optimizer, values, grads, slots):
    """Adagrad 纯 Python 实现"""
    lr = optimizer.learning_rate
    epsilon = getattr(optimizer, 'epsilon', 1e-7)
    
    accumulator = slots.get('accumulator', tf.zeros_like(values))
    new_accumulator = accumulator + tf.square(grads)
    updated_values = values - lr * grads / (tf.sqrt(new_accumulator) + epsilon)
    
    return updated_values, {'accumulator': new_accumulator}
```

### 4.2 修改: `dynamic_embedding_optimizer.py`

核心变更点:

```python
# 在 DynamicEmbeddingOptimizer() 函数内部

# 新增 PS 模式检测
from .ps_embedding_optimizer import is_ps_strategy, apply_de_ps_update

# 修改 apply_gradients 系列函数中处理 DE 变量的逻辑:
# 原来: 对 DE 变量也调用 _resource_apply_sparse/dense
# 修改后: 如果是 PS 模式 + DE 变量，走 apply_de_ps_update 路径

def _patched_apply_gradients(grads_and_vars, ...):
    strategy = distribute_ctx.get_strategy() if distribute_ctx.has_strategy() else None
    
    de_grads_and_vars = []
    normal_grads_and_vars = []
    
    for grad, var in grads_and_vars:
        if _is_de_variable(var) and is_ps_strategy(strategy):
            de_grads_and_vars.append((grad, var))
        else:
            normal_grads_and_vars.append((grad, var))
    
    # 非 DE 变量: 走原生路径
    if normal_grads_and_vars:
        original_apply_gradients(normal_grads_and_vars, ...)
    
    # DE 变量: 走 PS 专用路径
    for grad, var in de_grads_and_vars:
        apply_de_ps_update(optimizer, var.params, grad,
                          _get_slot_wrappers(var),
                          _get_slot_de_vars(var))
```

### 4.3 修改: `embedding_weights.py` - TrainableWrapper 增强

```python
class TrainableWrapper(ResourceVariable):
    def __init__(self, params, ids, max_norm, *args, **kwargs):
        ...
        # 新增: PS 模式标记
        self._ps_mode = kwargs.pop('ps_mode', False)
        ...
    
    # 新增: 提供直接访问底层 DE 变量的接口
    @property
    def de_variable(self):
        """返回底层的 de.Variable 哈希表"""
        return self.params
    
    @property
    def is_ps_mode(self):
        return self._ps_mode
```

### 4.4 修改: `tf_patch.py` - 增强 PS 放置防护

```python
def device_function(self, op):
    ...
    # 原来: 靠名字匹配 "TrainableWrapper"
    # 修改: 增加属性标记检查
    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    
    is_de_wrapper = ("TrainableWrapper" in node_def.name or 
                     "ShadowVariable" in node_def.name or
                     _is_de_related_op(node_def))
    
    if (not is_de_wrapper and self._ps_tasks
        and self._ps_device and node_def.op in self._ps_ops):
        ...  # 原有 PS 放置逻辑
```

### 4.5 修改: `shadow_embedding_ops.py` - PS 模式适配

```python
class ShadowVariable(EmbeddingWeights, TrainableWrapper):
    def __init__(self, params, ..., **kwargs):
        ...
        # 新增: 检测并标记 PS 模式
        self._ps_mode = _detect_ps_mode()
    
    # 确保在 PS 模式下，embedding_lookup 不触发对 PS handle 的操作
    def _read_variable_op(self, do_prefetch=True, no_copy=False):
        if self._ps_mode:
            # PS 模式下，直接从哈希表读取，不经过 resource handle
            return self.prefetch_values()
        else:
            return super()._read_variable_op(do_prefetch, no_copy)
```

## 五、槽位变量的处理策略

### 5.1 槽位变量的关键属性映射

| 优化器 | 槽位名称 | 作用 | DE 变量 key |
|--------|---------|------|-------------|
| Adam | m | 一阶动量 | `{var_name}/Adam/m` |
| Adam | v | 二阶动量 | `{var_name}/Adam/v` |
| SGD+Momentum | momentum | 动量 | `{var_name}/SGD/momentum` |
| Adagrad | accumulator | 梯度平方累加 | `{var_name}/Adagrad/accumulator` |
| RMSprop | rms | 均方根 | `{var_name}/RMSprop/rms` |
| RMSprop | momentum | 动量 | `{var_name}/RMSprop/momentum` |

### 5.2 槽位创建和管理

现有的 `create_slots()` 函数（`dynamic_embedding_optimizer.py:870`）已经为每个槽位创建了独立的 `de.Variable`（哈希表）。在 PS 模式下，这些哈希表不需要改变，只需要改变 **如何使用它们进行更新**。

原来的流程:
```
slot_wrapper.read_value() → 原生算子更新 slot_wrapper handle → slot_wrapper.update_op()
```

新的 PS 流程:
```
slot_de_var.lookup(ids) → Python 计算新值 → slot_de_var.upsert(ids, new_values)
```

### 5.3 主表-槽位的原子性保障

在 PS 模式下，单个 Worker 对同一批 IDs 的更新序列应该是：

```
1. lookup 主表 + 所有槽位表 (一次性读取)
2. 在 Python 中计算所有更新值
3. upsert 所有槽位表 (先写槽位)
4. upsert 主表 (最后写主表)
```

如果需要更强的原子性保证（多 Worker 并发场景），可以对每个 DE shard 使用 `tf.CriticalSection`：

```python
# 可选的强一致性保障
critical_sections = {}  # 每个 shard 一个 CriticalSection

def atomic_de_update(de_var, slot_de_vars, indices, new_values, new_slots):
    for shard_idx in range(de_var.shard_num):
        cs = critical_sections.setdefault(
            (de_var.name, shard_idx), 
            tf.CriticalSection(name=f"{de_var.name}_cs_{shard_idx}"))
        
        def _update_shard():
            shard_indices = ...  # 属于这个 shard 的 indices
            for slot_name, slot_var in slot_de_vars.items():
                slot_var.upsert(shard_indices, new_slots[slot_name])
            de_var.upsert(shard_indices, new_values)
        
        cs.execute(_update_shard)
```

## 六、PS 模式检测逻辑

```python
def is_ps_strategy(strategy=None):
    """检测当前是否在 ParameterServerStrategy 下运行"""
    if strategy is None:
        if not distribute_ctx.has_strategy():
            return False
        strategy = distribute_ctx.get_strategy()
    
    from tensorflow.python.distribute import parameter_server_strategy
    from tensorflow.python.distribute import parameter_server_strategy_v2
    
    return isinstance(strategy, (
        parameter_server_strategy.ParameterServerStrategyV1,
        parameter_server_strategy_v2.ParameterServerStrategyV2,
    ))

def _detect_ps_mode():
    """在变量创建时检测 PS 模式"""
    if not distribute_ctx.has_strategy():
        return False
    return is_ps_strategy()
```

## 七、向后兼容性保证

1. **PS 模式检测作为门控**: 所有新增逻辑都以 `is_ps_strategy()` 为前提条件，非 PS 模式完全不受影响
2. **非 DE 变量不受影响**: 只有 `TrainableWrapper`/`ShadowVariable` 类型的变量才走新路径
3. **现有接口不变**: `de.keras.layers.Embedding`、`de.DynamicEmbeddingOptimizer` 等公共 API 保持不变
4. **现有测试不应受影响**: 单机/Horovod/Mirrored 的测试应全部通过

## 八、风险和待解决问题

### 8.1 已识别风险

| 风险 | 严重度 | 缓解措施 |
|------|-------|---------|
| 纯 Python 优化器实现与原生 C++ 实现的数值精度差异 | 低 | 使用相同的 TF 数学运算 |
| 未支持的优化器类型 | 中 | 先支持 SGD/Adam/Adagrad，其他抛出明确错误信息 |
| 多 Worker 并发更新的一致性 | 中 | CriticalSection 可选加固 |
| 梯度累积/梯度裁剪等高级特性的兼容性 | 中 | 在 _normalize_gradients 中统一处理 |

### 8.2 后续优化方向

1. **性能优化**: 如果 Python 路径成为瓶颈，可以后续添加 C++ fused op（类似 DeepRec 的做法）
2. **更多优化器支持**: 按需添加 RMSprop、Ftrl、AdamW 等
3. **混合精度训练**: 确保 PS 路径支持 float16/bfloat16
4. **学习率调度**: 确保 PS 路径正确处理 LearningRateSchedule 类型的学习率
