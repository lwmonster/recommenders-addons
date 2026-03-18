# TFRA 动态 Embedding 整体设计思路

## 一、设计背景与动机

### 1.1 传统 Embedding 的局限

在传统的推荐系统中，TensorFlow 原生的 `tf.nn.embedding_lookup` 需要一个固定大小的 `tf.Variable` 作为 Embedding 矩阵：

```python
# 传统方式：必须预先定义词表大小
embedding_matrix = tf.Variable(tf.random.normal([vocab_size, dim]))  # vocab_size 必须固定
output = tf.nn.embedding_lookup(embedding_matrix, ids)
```

这带来了几个核心问题：
1. **词表大小必须预先确定**：实际业务中特征 ID 空间可能是数十亿甚至更大，无法预估
2. **哈希冲突**：通常使用取模哈希 `id % vocab_size` 来限制词表，但会导致不同特征共享 Embedding（冲突），影响模型效果
3. **内存浪费**：为避免冲突设置超大词表，但大量坑位可能从未被使用
4. **无法动态增删**：新特征上线或旧特征淘汰，需要重新训练整个 Embedding 矩阵

### 1.2 TFRA 的解决思路

TFRA（TensorFlow Recommenders Addons）引入了 **动态 Embedding 技术（Dynamic Embedding）**，核心思想是：

> **用哈希表（HashTable）替代固定大小的矩阵，实现按需分配、动态扩缩容的 Embedding 存储。**

- 每个特征 ID 是哈希表的 key，对应的 Embedding 向量是 value
- 新 ID 首次出现时自动插入（按 initializer 初始化）
- 不再需要预估词表大小，也没有哈希冲突
- 可以通过 RestrictPolicy 淘汰冷门特征，控制内存

## 二、整体架构

### 2.1 分层架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                        用户 API 层                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │ de.keras.layers. │  │ de.embedding_    │  │ de.get_variable│   │
│  │   Embedding      │  │   lookup()       │  │   ()           │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘   │
│           │                     │                     │            │
├───────────┼─────────────────────┼─────────────────────┼────────────┤
│           │          优化器集成层                       │            │
│           │  ┌──────────────────────────────────────┐ │            │
│           │  │ DynamicEmbeddingOptimizer             │ │            │
│           │  │  - 拦截 apply_gradients               │ │            │
│           │  │  - 管理槽位变量创建                    │ │            │
│           │  │  - 协调前向/反向数据流                 │ │            │
│           │  └──────────────────┬───────────────────┘ │            │
│           │                     │                     │            │
├───────────┼─────────────────────┼─────────────────────┼────────────┤
│           │          变量桥接层                        │            │
│  ┌────────┴─────────┐  ┌───────┴────────┐  ┌────────┴────────┐   │
│  │ ShadowVariable   │  │TrainableWrapper│  │DistributedVar- │   │
│  │ (eager/tf.func)  │  │(graph mode)    │  │iableWrapper    │   │
│  └────────┬─────────┘  └───────┬────────┘  └────────┬────────┘   │
│           │                     │                     │            │
│           │     将哈希表伪装为 TF ResourceVariable      │            │
│           │     使优化器能像操作普通变量一样操作它       │            │
│           │                     │                     │            │
├───────────┼─────────────────────┼─────────────────────┼────────────┤
│           │          核心存储层                        │            │
│  ┌────────┴─────────────────────┴─────────────────────┴────────┐  │
│  │                    de.Variable (动态哈希表)                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│  │  │ Shard 0  │  │ Shard 1  │  │ Shard N  │  ← 按 key 分片  │  │
│  │  │(Device 0)│  │(Device 1)│  │(Device N)│                  │  │
│  │  └──────────┘  └──────────┘  └──────────┘                  │  │
│  │  提供: lookup / upsert / remove / export / size             │  │
│  └─────────────────────────┬───────────────────────────────────┘  │
│                             │                                     │
├─────────────────────────────┼─────────────────────────────────────┤
│                    C++ 后端层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │CuckooHashMap │  │ HKV (Merlin) │  │  Redis Table │            │
│  │ (CPU/GPU)    │  │  (GPU only)  │  │  (Remote)    │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│  通过 REGISTER_OP 注册的自定义算子:                                │
│  *Find, *Insert, *Accum, *Remove, *Size, *Export, *Import        │
└───────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件职责一览

| 组件 | 文件 | 核心职责 |
|------|------|---------|
| `de.Variable` | `dynamic_embedding_variable.py` | 动态哈希表封装，管理多 shard 的 key-value 存储 |
| `TrainableWrapper` | `embedding_weights.py` | 将哈希表查询结果包装为 ResourceVariable，使优化器能处理 |
| `ShadowVariable` | `shadow_embedding_ops.py` | TrainableWrapper 的 eager 模式持久版本，支持 tf.function |
| `DynamicEmbeddingOptimizer` | `dynamic_embedding_optimizer.py` | 猴子补丁优化器，拦截梯度应用流程 |
| `embedding_lookup()` | `dynamic_embedding_variable.py` | 创建 TrainableWrapper + 执行哈希表查询 |
| `create_slots()` | `dynamic_embedding_optimizer.py` | 为优化器创建槽位变量（动量、速度等） |
| `RestrictPolicy` | `restrict_policies.py` | 控制哈希表大小，淘汰冷门特征 |
| `patch_on_tf()` | `tf_patch.py` | 补丁 TF 内部逻辑（设备放置、变量创建等） |
| `KVCreator` | `dynamic_embedding_creator.py` | 工厂类，创建不同后端的哈希表实例 |

## 三、参数存储机制

### 3.1 de.Variable —— 动态哈希表

`de.Variable` 是 TFRA 最核心的数据结构。它不是一个传统的"矩阵"，而是一个 **分布式哈希表**。

```python
import tensorflow_recommenders_addons.dynamic_embedding as de

# 创建一个动态 Embedding 变量
params = de.get_variable(
    name="user_embedding",
    key_dtype=tf.int64,       # key 类型: int64
    value_dtype=tf.float32,   # value 类型: float32
    dim=32,                   # 每个 key 对应 32 维向量
    devices=["/GPU:0"],       # 存放设备
    initializer=tf.keras.initializers.RandomNormal(0.0, 0.1),
    init_size=0,              # 初始容量（0 = 自动）
)
```

**内部结构：**

```
de.Variable
├── key_dtype: int64
├── value_dtype: float32
├── dim: 32
├── shard_num: len(devices)
├── partition_fn: default_partition_fn (key % shard_num)
├── _tables: [                      ← 每个 shard 一个哈希表实例
│     CuckooHashTable(device="/GPU:0"),
│     CuckooHashTable(device="/GPU:1"),
│   ]
├── _trainable_store: {}            ← 缓存该变量的 TrainableWrapper 实例
├── _distribute_trainable_store: {} ← 缓存分布式环境下的 DistributedVariableWrapper
├── initializer: RandomNormal       ← 新 key 的初始化器
├── restrict_policy: None           ← 可选的容量限制策略
└── bp_v2: False                    ← 反向传播模式（覆盖 vs 增量）
```

### 3.2 分片机制（Sharding）

de.Variable 支持将数据分布到多个设备上：

```python
# 创建跨 2 个 GPU 的 Embedding
params = de.get_variable(
    name="item_embedding",
    dim=64,
    devices=["/GPU:0", "/GPU:1"],  # 2 个分片
)
```

分片逻辑：
```python
def default_partition_fn(keys, shard_num):
    """默认分片函数：对 key 取模"""
    return tf.cast(keys % shard_num, dtype=tf.int32)
```

所有操作（lookup/upsert/remove）都会先通过 `partition_fn` 将 keys 路由到对应的 shard，然后在各 shard 上并行执行，最后 stitch（拼接）结果。

```
lookup(keys=[10, 23, 37, 44])
  │
  ├─ partition_fn: 10%2=0, 23%2=1, 37%2=1, 44%2=0
  │
  ├─ Shard 0 (GPU:0): lookup([10, 44]) → [vec_10, vec_44]
  ├─ Shard 1 (GPU:1): lookup([23, 37]) → [vec_23, vec_37]
  │
  └─ stitch: [vec_10, vec_23, vec_37, vec_44]  ← 恢复原始顺序
```

### 3.3 C++ 后端

TFRA 提供三种哈希表后端，通过 `KVCreator` 工厂类选择：

| 后端 | 文件 | 适用场景 | 特点 |
|------|------|---------|------|
| CuckooHashMap | `cuckoo_hashtable_ops.cc` | CPU/GPU 通用 | 默认后端，基于布谷鸟哈希，支持动态扩容 |
| HKV (Merlin) | `hkv_hashtable_ops.cc` | GPU 高性能 | 基于 NVIDIA Merlin HKV，支持 score/eviction |
| Redis | `redis_table_ops.cc` | 远程存储 | 支持 Cluster/Sentinel/Standalone 三种模式 |

```python
# 使用 HKV 后端（GPU 高性能）
params = de.get_variable(
    name="feature_embedding",
    dim=64,
    kv_creator=de.HkvHashTableCreator(config=de.HkvHashTableConfig(
        init_capacity=1024*1024,
        max_capacity=10*1024*1024,
    )),
)
```

每种后端都注册了一组自定义 C++ 算子（以 Cuckoo 为例）：
- `CuckooHashTableOfTensors` — 创建哈希表
- `CuckooHashTableFind` / `FindWithExists` — 查询
- `CuckooHashTableInsert` — 插入
- `CuckooHashTableAccum` — 增量累加（bp_v2 模式用）
- `CuckooHashTableRemove` — 删除
- `CuckooHashTableSize` — 获取当前元素数量
- `CuckooHashTableExport` / `Import` — 导出/导入全量数据
- `CuckooHashTableClear` — 清空
- `CuckooHashTableSaveToFileSystem` / `LoadFromFileSystem` — 文件系统持久化

### 3.4 容量管理（RestrictPolicy）

由于动态哈希表理论上可以无限增长，TFRA 提供了 `RestrictPolicy` 来控制容量：

```python
# 使用 TimestampRestrictPolicy，只保留最近访问的 100 万特征
params = de.get_variable(
    name="feature_embedding",
    dim=64,
    restrict_policy=de.TimestampRestrictPolicy,
)
params.restrict(num_reserved=1_000_000)  # 淘汰超出部分
```

策略类型：
- `TimestampRestrictPolicy`：基于访问时间戳，淘汰最久未访问的特征
- `FrequencyRestrictPolicy`：基于访问频率，淘汰低频特征

RestrictPolicy 会同时管理主表和优化器槽位表，确保淘汰时所有相关数据一致删除。

## 四、前向传播详解

### 4.1 整体流程

```
用户输入: ids = [15, 2, 99, 15, 2]    (batch 中的特征 ID，可能有重复)
                    │
                    ▼
        ┌─────────────────────┐
        │  de.embedding_lookup │  (或 de.keras.layers.Embedding.call)
        │  / shadow_ops.       │
        │  embedding_lookup    │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 1. 创建/获取         │
        │    TrainableWrapper  │  ← 将哈希表伪装为 ResourceVariable
        │    (或 ShadowVariable)│
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 2. prefetch_values() │
        │    = params.lookup   │  ← 从哈希表查询，新 ID 自动初始化并插入
        │      (ids)           │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 3. assign 到 wrapper │
        │    的 resource handle│  ← 将查询结果写入 ResourceVariable 的 handle
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 4. read_variable_op  │
        │    → 返回 Tensor     │  ← TF 自动微分可以追踪这个 Tensor
        └──────────┬──────────┘
                   │
                   ▼
        输出: embeddings = [[vec_15], [vec_2], [vec_99], [vec_15], [vec_2]]
              shape = [5, dim]
```

### 4.2 Graph 模式 vs Eager 模式

TFRA 对两种模式使用不同的桥接对象：

| 模式 | 桥接对象 | 创建位置 | 特点 |
|------|---------|---------|------|
| Graph 模式 | `TrainableWrapper` | `embedding_lookup()` 内部 | 每次 embedding_lookup 创建新实例 |
| Eager/tf.function | `ShadowVariable` | Keras 层 `__init__` 或 `embedding_lookup()` | 持久化实例，复用跨 step |

**Graph 模式的 embedding_lookup（`dynamic_embedding_variable.py:1362`）：**

```python
def embedding_lookup(params, ids, name=None, ...):
    # 1. 计算初始 shape
    if ids.get_shape().is_fully_defined():
        initial_shape = [ids.get_shape().num_elements(), params.dim]
    else:
        initial_shape = (1, params.dim)  # 动态 shape 时用占位
    
    # 2. 创建初始值（全零）
    initial_value = tf.zeros(shape=initial_shape, dtype=params.value_dtype)
    
    # 3. 创建 TrainableWrapper
    wrapper = TrainableWrapper(
        params=params,     # 底层的 de.Variable 哈希表
        ids=ids,           # 本次查询的 key
        max_norm=max_norm,
        initial_value=initial_value,
        dtype=params.value_dtype,
        trainable=params.trainable,
        name=trainable_name,
    )
    
    # 4. wrapper 被当作普通 Tensor 返回
    # TF 自动微分会追踪它，反向传播时产生梯度
    return wrapper
```

**Eager/tf.function 模式的 embedding_lookup（`shadow_embedding_ops.py:239`）：**

```python
def embedding_lookup(shadow: ShadowVariable, ids, ...):
    # 1. 更新 ShadowVariable 的 ids buffer
    shadow._reset_ids(ids)
    
    # 2. 从哈希表查询并写入 shadow 的 handle
    result = shadow.read_value(do_prefetch=True)
    # 内部: params.lookup(ids) → assign_variable_op(handle, values) → read
    
    return result
```

### 4.3 TrainableWrapper 的 prefetch 机制

TrainableWrapper 的核心在于 `prefetch_values()`——它是哈希表与 ResourceVariable 之间的桥梁：

```python
def prefetch_values(self, update=False):
    """从哈希表查询数据，填充到 ResourceVariable 的 handle 中"""
    if self.params.bp_v2:
        # bp_v2 模式：同时返回 exists 标记（用于增量更新）
        r, self.exists = self.params.lookup(self.ids, return_exists=True)
        self.prefetch_values_op = self.transform(r)  # 可选的 max_norm clip
    else:
        self.prefetch_values_op = self.transform(self.params.lookup(self.ids))
    return self.prefetch_values_op
```

在 `_read_variable_op()` 中，先 assign 再 read：
```python
def _read_variable_op(self, do_prefetch=True):
    if self.model_mode == "train":
        if do_prefetch:
            # 先将 prefetch 的值写入 handle
            with ops.control_dependencies([
                gen_resource_variable_ops.assign_variable_op(
                    self._handle, self.prefetch_values())
            ]):
                # 再从 handle 读取（这样 TF 自动微分能追踪到）
                result = gen_resource_variable_ops.read_variable_op(
                    self._handle, self._dtype)
        else:
            result = gen_resource_variable_ops.read_variable_op(
                self._handle, self._dtype)
    else:
        # 推理模式：直接返回查询结果，不经过 handle
        result = self.prefetch_values()
    return self.transform(result)
```

### 4.4 Keras 层的前向传播

`de.keras.layers.Embedding` 在内部使用 ShadowVariable：

```python
class Embedding(Layer):
    def __init__(self, embedding_size, ...):
        # 1. 创建 de.Variable（哈希表）
        self.params = de.get_variable(name + '-parameter', dim=embedding_size, ...)
        
        # 2. 创建 ShadowVariable（可训练的影子变量）
        self.shadow = de.shadow_ops.ShadowVariable(self.params, name=name + '-shadow', ...)
    
    def call(self, ids):
        # 3. 通过 ShadowVariable 做 embedding lookup
        return de.shadow_ops.embedding_lookup_unique(self.shadow, ids, self.embedding_size)
```

`embedding_lookup_unique` 还做了去重优化：

```python
def embedding_lookup_unique(shadow, ids, embedding_size, with_unique=True, name=None):
    ids_flat = tf.reshape(ids, (-1,))
    if with_unique:
        # 去重后查表，减少哈希表访问次数
        unique_ids, idx = tf.unique(ids_flat)
        unique_embeddings = embedding_lookup(shadow, unique_ids)  # 只查不重复的
        embeddings_flat = tf.gather(unique_embeddings, idx)       # 恢复原始顺序
    else:
        embeddings_flat = embedding_lookup(shadow, ids_flat)
    return tf.reshape(embeddings_flat, [shape_of_ids..., embedding_size])
```

## 五、反向传播详解

### 5.1 为什么需要特殊处理？

TF 的原生优化器（Adam、SGD 等）假设它们操作的是 **固定大小的 ResourceVariable**。但 TFRA 的 Embedding 是哈希表，需要特殊的反向传播流程：

1. 优化器更新的目标不是 wrapper handle 本身，而是底层哈希表中的值
2. 优化器的槽位变量（如 Adam 的 m、v）也需要是动态哈希表，而非固定大小矩阵
3. 更新完成后，需要把 wrapper handle 中的新值写回（upsert）到哈希表

### 5.2 DynamicEmbeddingOptimizer 的猴子补丁机制

`DynamicEmbeddingOptimizer` 不是一个新的优化器类，而是一个 **装饰函数**，它通过猴子补丁修改现有优化器的行为：

```python
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer = de.DynamicEmbeddingOptimizer(optimizer)
# 现在 optimizer 仍然是 Adam，但它的 apply_gradients 等方法被替换了
```

被替换的核心方法：
```
DynamicEmbeddingOptimizer(self):
  │
  ├─ 替换 self.apply_gradients → 新的 apply_gradients
  │     （根据优化器类型和分布式策略选择不同的实现）
  │
  ├─ 替换 self.add_slot / add_variable_from_reference → 新的槽位创建逻辑
  │     （为 DE 变量创建哈希表槽位而非固定矩阵槽位）
  │
  ├─ 替换 self._distributed_apply → 新的分布式 apply
  │     （区分 DE 变量和普通变量的处理路径）
  │
  └─ 替换 self._get_or_make_slot → 新的槽位获取逻辑
        （为 DE 变量的槽位创建独立的哈希表）
```

### 5.3 反向传播完整流程

以 Adam 优化器为例，完整的反向传播流程如下：

```
                    梯度计算
                       │
                       ▼
              grad = IndexedSlices(
                indices=[0, 1, 2, ..., 62],  ← 哪些位置有梯度
                values=[[g0], [g1], ...],    ← 梯度值
              )
                       │
                       ▼
        ┌──────────────────────────────┐
        │  optimizer.apply_gradients   │
        │  → DynamicEmbeddingOptimizer │
        │    拦截处理                    │
        └──────────────┬───────────────┘
                       │
          ┌────────────┴────────────┐
          │ 是 DE 变量?              │
          ├─ 否 → 原生路径           │
          └─ 是 → DE 专用路径 ↓      │
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Step 1: 获取槽位变量         │
        │  _slots = [get_slot(var, 'm'),│
        │            get_slot(var, 'v')]│
        │  var._track_optimizer_slots() │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Step 2: 快照当前值           │
        │  v0 = var.read_value()       │  ← 从哈希表查询主变量值
        │  s0 = [slot.read_value()     │  ← 从哈希表查询槽位值
        │        for slot in _slots]   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Step 3: 原生优化器算子更新   │
        │  _resource_apply_sparse(     │  ← TF 原生 C++ 算子
        │    grad.values, var,         │    在 wrapper 的 handle 上执行
        │    grad.indices)             │    更新 m, v, 和参数值
        │  (在 wrapper handle 上操作)   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Step 4: 写回哈希表           │
        │  var.update_op(v0=v0)        │  ← 把更新后的值写回主哈希表
        │  slot.update_op(v0=s0[i])    │  ← 把更新后的槽位值写回槽位哈希表
        └──────────────────────────────┘
```

### 5.4 update_op 的两种模式

`TrainableWrapper.update_op()` 负责将 wrapper handle 中更新后的值写回哈希表：

```python
def update_op(self, v0=None):
    v1 = self.read_value(do_prefetch=False)  # 读取优化器更新后的值（不重新查表）
    
    if self.params.bp_v2:
        # bp_v2=True: 增量模式
        # 计算 delta = v1 - v0，然后：
        #   如果 key 已存在：current_value += delta
        #   如果 key 不存在：current_value = v1
        update_param_op = self.params.accum(self.ids, v0, v1, self.exists)
    else:
        # bp_v2=False: 覆盖模式（默认）
        # 直接用 v1 覆盖哈希表中的值
        update_param_op = self.params.upsert(self.ids, v1)
    
    # 如果配置了 restrict_policy，同时更新访问状态
    if self.params.restrict_policy is not None:
        update_status_op = self.params.restrict_policy.apply_update(self.ids)
        return control_flow_ops.group([update_param_op, update_status_op])
    
    return update_param_op
```

**bp_v2 模式的意义：**
- 在大规模异步分布式训练中，多个 worker 可能同时更新同一个 key
- bp_v2=False（覆盖模式）：后写入的 worker 会覆盖先写入的，造成梯度丢失
- bp_v2=True（增量模式）：每个 worker 只累加自己的 delta，所有梯度都会被保留

### 5.5 槽位变量的创建（create_slots）

当优化器需要为 DE 变量创建槽位时（如 Adam 的 m 和 v），`create_slots` 函数会为每个槽位创建一个独立的 `de.Variable`（哈希表），而不是普通的 `tf.Variable`：

```python
def create_slots(variable, init, slot_name, op_name, bp_v2):
    """
    为 DE 变量创建优化器槽位
    
    例如 Adam 需要 m (一阶动量) 和 v (二阶动量)：
    - 主变量: de.Variable("user_embedding")
    - 槽位 m: de.Variable("user_embedding/Adam/m")  ← 也是哈希表！
    - 槽位 v: de.Variable("user_embedding/Adam/v")  ← 也是哈希表！
    """
    # 1. 创建槽位哈希表（与主表相同配置）
    slot_variable_ = de.Variable(
        name=full_name,                       # e.g., "user_embedding/Adam/m"
        key_dtype=params_var_.key_dtype,       # 与主表相同
        value_dtype=params_var_.value_dtype,   # 与主表相同
        dim=params_var_.dim,                   # 与主表相同
        devices=params_var_.devices,           # 与主表相同
        partitioner=params_var_.partition_fn,  # 与主表相同
        initializer=init,                     # 通常是全零
        kv_creator=params_var_.kv_creator,     # 使用相同后端
        trainable=False,                       # 槽位不需要梯度
    )
    
    # 2. 为槽位创建 TrainableWrapper/ShadowVariable
    #    与主表的 wrapper 共享相同的 ids
    slot_trainable = TrainableWrapper(
        params=slot_variable_,
        ids=variable.ids,   # 关键: 与主表使用相同的 ids！
        ...
    )
    
    return slot_trainable
```

**核心设计点：主表和槽位表的 TrainableWrapper 共享同一个 `ids` 引用。** 这确保了：
- 前向传播中 lookup 了哪些 ID，反向传播中 m 和 v 也只更新这些 ID
- 通过 `_reset_ids()` 级联重置，所有关联的 wrapper 一起切换到新的 ID 集合

```python
def _reset_ids(self, ids):
    """重置 ids 并级联到所有跟踪的槽位"""
    self.ids = ids
    self.prefetch_values(update=True)  # 重新查表
    for s in self._tracked_slots:
        s._reset_ids(ids)  # 级联到槽位 wrapper
```

## 六、Checkpoint 保存与恢复

### 6.1 保存机制

TFRA 的哈希表数据通过自定义 Saveable 对象保存。两种方式：

**方式一：TF Checkpoint（默认）**
```python
# 通过 SaveableObject 机制集成到 tf.train.Checkpoint
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.save("model.ckpt")
# 内部会调用 table.export() 导出所有 key-value 对
```

**方式二：FileSystem 直接保存**
```python
# 直接保存到文件系统（支持 Horovod 多节点）
params.save_to_file_system(
    dirpath="/path/to/save",
    proc_size=hvd.size(),    # 总节点数
    proc_rank=hvd.rank(),    # 当前节点编号
)
```

### 6.2 恢复机制

```python
# 从 Checkpoint 恢复
checkpoint.restore("model.ckpt")
# 内部会调用 table.import() 导入所有 key-value 对

# 从文件系统恢复
params.load_from_file_system(
    dirpath="/path/to/save",
    proc_size=hvd.size(),
    proc_rank=hvd.rank(),
)
```

## 七、分布式训练支持

### 7.1 已支持的分布式模式

| 模式 | 支持状态 | 实现方式 |
|------|---------|---------|
| 单机 | ✅ 完全支持 | 默认模式 |
| Horovod (AllReduce) | ✅ 完全支持 | 各 worker 独立持有完整模型，AllReduce 同步梯度 |
| MirroredStrategy | ✅ 基本支持 | 变量在每个 GPU 上有副本 |
| ParameterServerStrategy | ❌ 不支持 | 动态 shape 与 PS 变量管理冲突（见根因分析文档） |

### 7.2 Horovod 集成

TFRA 提供了 `HvdAllToAllEmbedding` 层，支持 Embedding 的张量并行：

```python
# 各 worker 只持有部分 Embedding 表，通过 AllToAll 通信查询
embedding = de.keras.layers.HvdAllToAllEmbedding(
    embedding_size=64,
    name='distributed_embedding',
)
```

工作流程：
1. 每个 worker 将自己的 IDs 按分片规则路由到对应 worker
2. 各 worker 查询自己负责的 IDs
3. 通过 Horovod AllToAll 将查询结果返回给请求方

### 7.3 TF Distribute Strategy 集成

在 `tf.distribute` 下，TFRA 会为每个 replica 创建独立的 ShadowVariable/TrainableWrapper，并用 `DistributedVariableWrapper` 包装：

```python
# embedding_lookup 中的分布式处理逻辑
if distribute_ctx.has_strategy():
    strategy_devices = strategy.extended.worker_devices
    trainable_impl = []
    for i, device in enumerate(strategy_devices):
        with ops.device(device):
            trainable_impl.append(_create_or_get_trainable(name_replica))
    
    trainable_ = DistributedVariableWrapper(
        strategy, trainable_impl,
        tf.VariableAggregation.NONE, ...)
```

## 八、TF 内部补丁（tf_patch）

TFRA 通过 `patch_on_tf()` 修改了 TF 的几个内部函数：

| 补丁目标 | 原函数 | 修改目的 |
|---------|--------|---------|
| `optimizer._get_processor` | TF1 优化器的变量处理器选择 | 让 TrainableWrapper 走自定义的 `_DenseDynamicEmbeddingTrainableProcessor` |
| `slot_creator._create_slot_var` | TF1 优化器的槽位创建 | 支持 DE 变量的动态 shape 槽位 |
| `device_setter._ReplicaDeviceChooser.device_function` | 设备放置逻辑 | 防止 TrainableWrapper 被放到 PS 上 |
| `VarianceScaling.__call__` | Keras 初始化器 | 支持动态 shape 的初始化（shape 可以是 Tensor） |
| `CheckpointPosition.bind_object` | Checkpoint 绑定 | 跳过 ShadowVariable 的 checkpoint 绑定 |
| `TensorShape.as_list` | Shape 转换 | 处理未知维度的 shape |

## 九、推理模式

TFRA 支持训练/推理模式切换：

```python
de.enable_inference_mode()   # 切换到推理模式
de.enable_train_mode()       # 切换回训练模式
```

在推理模式下：
- `_read_variable_op()` 不再经过 ResourceVariable handle，直接返回 `params.lookup(ids)` 结果
- 减少了 assign + read 的开销
- 不创建 TrainableWrapper，不追踪梯度

## 十、设计总结

### 10.1 核心设计思想

```
┌─────────────────────────────────────────────────────────────┐
│                    TFRA 的设计哲学                            │
│                                                             │
│  1. 存储与计算分离:                                          │
│     哈希表负责存储 → TrainableWrapper 负责桥接 → TF 负责计算   │
│                                                             │
│  2. 按需分配:                                                │
│     特征 ID 首次出现时才分配 Embedding 向量                    │
│     无需预估词表大小，无哈希冲突                               │
│                                                             │
│  3. 伪装透明:                                                │
│     TrainableWrapper 让哈希表看起来像普通 ResourceVariable     │
│     优化器、自动微分、Keras 层 都无需修改                      │
│                                                             │
│  4. 后端可插拔:                                               │
│     CuckooHash / HKV / Redis 三种后端，统一接口               │
│     通过 KVCreator 工厂选择                                  │
│                                                             │
│  5. 与 TF 生态兼容:                                          │
│     通过猴子补丁（tf_patch）最小侵入地集成到 TF 中             │
│     支持 Checkpoint、SavedModel、tf.function 等              │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 数据流端到端

```
训练一个 step 的完整数据流:

输入: batch_ids = [15, 2, 99]
         │
    ═════╤═════ 前向传播 ═════════════════════════════════
         │
    1. de.Variable.lookup(ids)        → 查哈希表 → [vec_15, vec_2, vec_99]
    2. assign → TrainableWrapper.handle   → 写入 ResourceVariable handle
    3. read_variable_op(handle)       → TF 自动微分开始追踪
    4. model forward(embeddings)      → DNN 等后续层计算
    5. loss = loss_fn(output, label)
         │
    ═════╤═════ 反向传播 ═════════════════════════════════
         │
    6. tape.gradient(loss, wrapper)   → 得到 grad (IndexedSlices)
    7. DynamicEmbeddingOptimizer 拦截:
       a. 快照: v0 = wrapper.read_value()
       b. 快照: s0 = [slot.read_value()]  (从槽位哈希表查询)
       c. 原生算子: _resource_apply_sparse(grad, wrapper)  (在 handle 上更新)
       d. 写回主表: wrapper.update_op(v0) → de.Variable.upsert(ids, new_values)
       e. 写回槽位: slot.update_op(s0)   → slot_de_var.upsert(ids, new_slot_values)
         │
    ═════╧═════ step 结束 ════════════════════════════════

哈希表状态变化:
  主表:   {15: vec_15', 2: vec_2', 99: vec_99'}    ← 已更新
  Adam.m: {15: m_15',   2: m_2',   99: m_99'}      ← 已更新
  Adam.v: {15: v_15',   2: v_2',   99: v_99'}      ← 已更新
```

### 10.3 关键源码文件索引

| 文件 | 路径 | 行数 | 核心内容 |
|------|------|------|---------|
| `dynamic_embedding_variable.py` | `python/ops/` | 1551 | de.Variable 类、embedding_lookup 函数 |
| `embedding_weights.py` | `python/ops/` | 540 | TrainableWrapper 类、ModelMode |
| `shadow_embedding_ops.py` | `python/ops/` | 457 | ShadowVariable 类、eager 模式 lookup |
| `dynamic_embedding_optimizer.py` | `python/ops/` | 958 | DynamicEmbeddingOptimizer、create_slots |
| `tf_patch.py` | `python/ops/` | 409 | TF 内部补丁 |
| `distributed_embedding_variable.py` | `python/ops/` | 25 | DistributedVariableWrapper |
| `restrict_policies.py` | `python/ops/` | 362 | 容量限制策略 |
| `dynamic_embedding_creator.py` | `python/ops/` | - | KVCreator 工厂 |
| `keras/layers/embedding.py` | `python/keras/layers/` | 594 | Keras Embedding 层 |
| `cuckoo_hashtable_ops.cc` | `core/ops/` | - | Cuckoo 哈希表 C++ 算子注册 |
| `cuckoo_hashtable_op.cc` | `core/kernels/` | - | Cuckoo 哈希表 C++ 内核实现 |
| `hkv_hashtable_ops.cc` | `core/ops/` | - | HKV 哈希表 C++ 算子注册 |
| `lookup_table_op_cpu.h` | `core/kernels/lookup_impl/` | - | CPU 哈希表底层实现 |
| `lookup_table_op_gpu.h` | `core/kernels/lookup_impl/` | - | GPU 哈希表底层实现 |
