# TFRA 动态 Embedding 在 ParameterServerStrategy 下的根因分析

## 一、问题现象

在使用 TFRA 的 Keras API（`de.keras.layers.Embedding`）构建模型，并通过 `ParameterServerStrategy` 进行分布式训练时，无论使用 Adam、SGD 或其他任何优化器，都会在训练刚开始时触发以下两类崩溃：

| TF 版本 | 报错信息 | 崩溃算子 |
|---------|---------|---------|
| TF 2.16 | `indices[63] = 63 is not in [0, 63)` | `ResourceScatterAdd` |
| TF 2.15 | `Cannot update variable with shape [62,32] using a Tensor with shape [63,32]` | `AssignSubVariableOp` |

**关键事实**: 即使使用无状态的 SGD（无动量槽位），问题依然存在。这证明问题不在优化器槽位同步，而在更底层的变量形状管理机制。

## 二、TFRA 动态 Embedding 的核心架构

### 2.1 数据流概览

```
┌──────────────────────────────────────────────────────────┐
│                    de.Variable (动态哈希表)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ CuckooHash  │  │ CuckooHash  │  │   ...       │      │
│  │  Shard 0    │  │  Shard 1    │  │             │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└──────────────────────────┬───────────────────────────────┘
                           │ lookup / upsert
                           ▼
┌──────────────────────────────────────────────────────────┐
│         TrainableWrapper (ResourceVariable 子类)          │
│  - 作为 TF 优化器眼中的"普通变量"                          │
│  - prefetch_values(): 从哈希表 lookup 填充本地 handle     │
│  - update_op(): 将优化后的值 upsert 回哈希表              │
│  - shape: 由当前 step 的 ids 数量决定（动态变化！）        │
└──────────────────────────┬───────────────────────────────┘
                           │ 被优化器当作普通 Variable 使用
                           ▼
┌──────────────────────────────────────────────────────────┐
│              TF 原生优化器 (Adam/SGD/...)                  │
│  - _resource_apply_sparse → ResourceScatterAdd           │
│  - _resource_apply_dense → AssignSubVariableOp           │
│  这些 C++ 算子假设：变量形状在 step 内静态不变             │
└──────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

| 文件 | 组件 | 职责 |
|-----|------|------|
| `dynamic_embedding_variable.py` | `Variable` | 动态哈希表，支持 lookup/upsert/remove，可跨设备分片 |
| `embedding_weights.py` | `TrainableWrapper` | ResourceVariable 子类，桥接哈希表与 TF 变量系统 |
| `shadow_embedding_ops.py` | `ShadowVariable` | TrainableWrapper 的 eager 模式持久化版本，用于 tf.function/keras |
| `dynamic_embedding_optimizer.py` | `DynamicEmbeddingOptimizer` | 猴子补丁优化器，拦截 apply_gradients |
| `tf_patch.py` | `patch_on_tf()` | 补丁 TF 内部函数（设备放置、槽位创建等） |
| `distributed_embedding_variable.py` | `DistributedVariableWrapper` | 包装多副本 trainable 变量 |

### 2.3 单机模式下的正常工作流程

```
1. 前向传播:
   embedding_lookup(de_var, ids)
     → TrainableWrapper 创建，initial_shape = [len(ids), dim]
     → prefetch_values(): de_var.lookup(ids) → 结果写入 wrapper handle
     → 返回 wrapper 作为可训练张量

2. 反向传播:
   optimizer.apply_gradients([(grad, wrapper)])
     → DynamicEmbeddingOptimizer 拦截
     → v0 = wrapper.read_value()         # 快照当前值
     → s0 = [slot.read_value()]           # 快照槽位值
     → _resource_apply_sparse(grad, wrapper)  # TF 原生算子更新 wrapper handle
     → wrapper.update_op(v0)              # 将更新后的值写回哈希表
     → slot.update_op(s0)                 # 将槽位值写回槽位哈希表

关键: 在单机模式下，上述步骤是严格串行的:
  - step 开始 → lookup（可能插入新 key）→ 表扩容
  - 同一 step 的 backward → wrapper handle 已经是扩容后的大小
  - 原生算子操作的 shape 与 handle 中的数据一致 → 正常工作
```

## 三、根本原因分析

### 3.1 核心矛盾

**TF 原生优化器 C++ 算子的假设**：
> "张量（变量）的形状在一个 step 内是静态不变的"

**TFRA 动态 Embedding 的设计理念**：
> "哈希表可以在任何时刻动态增长（新特征 ID 触发自动插入）"

在单机模式下，这个矛盾被 **执行顺序的串行性** 隐藏了。在 PS 模式下，这个矛盾被 **图切分和异步执行** 彻底暴露。

### 3.2 PS 模式下的具体崩溃链路

```
时序图 (ParameterServerStrategy):

Worker                                    PS
  │                                        │
  │  1. 前向传播                            │
  │  embedding_lookup(ids=[0,1,...,62])     │
  │  → 新 ID 触发 table.insert()           │
  │  → 哈希表: 62 → 63 条                   │
  │  → TrainableWrapper shape: [63, dim]   │
  │                                        │
  │  2. 计算梯度                            │
  │  grad = IndexedSlices(                  │
  │    indices=[0,1,...,62],   ← 63 个索引  │
  │    values=[63, dim])                   │
  │                                        │
  │  3. 发送梯度到 PS ──────────────────────→│
  │                                        │  4. PS 上的 ResourceScatterAdd
  │                                        │     尝试: var[62] += grad[62]
  │                                        │     但 PS 上的 var 还是 [62, dim]!
  │                                        │     ══════════════════════════
  │                                        │     ❌ indices[62]=62 不在 [0,62) 中
  │                                        │     ══════════════════════════
  │                                        │
```

### 3.3 为什么 TrainableWrapper handle 在 PS 上形状不对？

这是问题的核心技术细节：

1. **TrainableWrapper 是 ResourceVariable 的子类**。在 PS 模式下，`ParameterServerStrategy` 会自动将变量（包括 TrainableWrapper）的 resource handle 放置在 PS 节点上。

2. **TrainableWrapper 的 handle 创建**（`embedding_weights.py:327-333`）：
   ```python
   handle = resource_variable_ops.eager_safe_variable_handle(
       initial_value=initial_value,
       shape=None,  # 注意: shape 传的是 None
       shared_name=shared_name,
       ...
   )
   ```
   虽然 handle 创建时 shape=None，但 `initial_value` 的 shape 会被 TF 运行时记录。

3. **PS 的变量分片机制**：PS strategy 的 `_create_variable` 方法会对变量按第一个维度进行分片（如果配置了 `variable_partitioner`）。即使没有显式分片，变量的 resource handle 也会被放置在 PS 上，而 TF C++ runtime 会根据 handle 关联的张量 shape 做边界检查。

4. **前向传播中的 assign**：`_read_variable_op()` 在 train 模式下会先 `assign_variable_op(handle, prefetch_values())`，这个 assign 确实会更新 PS 上的 handle 中的张量。但是:
   - 在 PS 的异步执行模型下，assign 和后续的 scatter_add 可能不在同一个执行序列中
   - Worker 计算出的梯度 indices 是基于 **扩容后** 的表大小，但 PS 上的 scatter_add 可能看到的是 **扩容前** 的变量状态

5. **tf_patch.py 的防护不足**（第 249-254 行）：
   ```python
   # TODO(rhdong): `TrainableWrapper` is not multi-threads safe
   if ("TrainableWrapper" not in node_def.name ...):
       # 仅靠名字匹配来避免 PS 放置，不够可靠
   ```
   这只是一个基于名字的启发式规避，无法从根本上解决问题。

### 3.4 为什么 SGD（无动量）也崩溃？

很多人以为问题出在动量槽位（m, v）的同步上。但实际上：

1. SGD 无动量时，不需要槽位变量
2. 崩溃发生在 `ResourceScatterAdd` 对 **主变量本身** 的更新上
3. 这证明问题的根源是 **TrainableWrapper 作为 ResourceVariable 被 PS strategy 管理时的形状不一致**，与槽位无关

### 3.5 问题本质的一句话总结

> **TFRA 的 TrainableWrapper 将动态大小的哈希表伪装成了静态大小的 ResourceVariable，这个"伪装"在单机执行的串行语义下成立，但在 PS 异步分布式执行的图切分语义下失效——Worker 侧看到的是扩容后的 shape，PS 侧看到的是扩容前的 shape，原生 C++ 算子的边界检查直接判死刑。**

## 四、影响范围

| 场景 | 是否受影响 | 原因 |
|------|----------|------|
| 单机训练 | ❌ 不受影响 | 串行执行，shape 始终一致 |
| MirroredStrategy | ❌ 不受影响 | 变量在每个 worker 上有副本，各自串行 |
| Horovod | ❌ 不受影响 | 各 worker 独立持有完整模型，AllReduce 同步梯度 |
| ParameterServerStrategy | ✅ **必然崩溃** | 变量被放置在 PS 上，图切分导致 shape 不一致 |
| CentralStorageStrategy | ✅ 可能崩溃 | 变量集中存储，类似 PS 的 shape 不一致风险 |

## 五、已有代码中的相关线索

1. **`tf_patch.py:249-254`**: 开发者已经意识到 TrainableWrapper 不是多线程安全的，尝试用名字匹配避免 PS 放置
2. **`dynamic_embedding_optimizer.py:729-735`**: 对 PS + `experimental_aggregate_gradients=False` 显式抛出 `NotImplementedError`
3. **`demo/movielens-1m-keras-ps/`**: 存在一个 PS 模式的 demo，但已知无法正常运行
4. **DeepRec 项目的做法**: 完全自定义了 `EmbeddingVariable` 类型 + 专用优化器内核，不依赖 TF 原生 ResourceScatterAdd

## 六、结论

这不是一个简单的 bug fix，而是 **TFRA 动态 Embedding 架构在 PS 分布式模式下的设计缺陷**。核心问题是：

1. TrainableWrapper 继承自 ResourceVariable，被 TF 的 PS 机制当作普通变量管理
2. 但它的"内容"来自动态哈希表，shape 可以在 step 内变化
3. PS 模式下的图切分和异步执行打破了单机模式下隐含的"shape 在 step 内不变"这一假设
4. TF 原生 C++ 算子（ResourceScatterAdd 等）严格校验 shape 边界，直接崩溃

修复需要在 Python 层面绕过 TF 原生优化器对 DE 变量的直接操作，改用 TFRA 自己的哈希表操作来完成梯度更新。
