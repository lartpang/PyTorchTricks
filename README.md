# Some Tricks of PyTorch

## changelog

- 2019 年 11 月 29 日: 更新一些模型设计技巧和推理加速的内容, 补充了下 apex 的一个介绍链接, ~~另外删了 tfrecord, pytorch 能用么? 这个我记得是不能, 所以删掉了~~(表示删掉:<)
- 2019 年 11 月 30 日: 补充 MAC 的含义, 补充 ShuffleNetV2 的论文链接
- 2019 年 12 月 02 日: 之前说的 pytorch 不能用 tfrecord, 今天看到<https://www.zhihu.com/question/358632497>下的一个回答, 涨姿势了
- 2019 年 12 月 23 日: 补充几篇关于模型压缩量化的科普性文章
- 2020 年 2 月 7 日: 从文章中摘录了一点注意事项, 补充在了[代码层面](#代码层面)小节
- 2020 年 4 月 30 日:
  - 添加了一个 github 的文档备份
  - 补充了卷积层和 BN 层融合的介绍的链接
  - 另外这里说明下, 对于之前参考的很多朋友的文章和回答, 没有把链接和对应的内容提要关联在一起, 估计会导致一些朋友阅读时相关的内容时的提问, 无法问到原作者, 这里深感抱歉.
  - 调整部分内容, 将内容尽量与参考链接相对应
- 2020 年 5 月 18 日: 补充一些关于 PyTorch 节省显存的技巧. 同时简单调整格式. 另外发现一个之前的错误: `non_blocking=False` 的建议应该是 `non_blocking=True` .
- 2021 年 01 月 06 日：调整下关于读取图片数据的一些介绍.
- 2021 年 01 月 13 日：补充了一条推理加速的策略. 我觉得我应该先更新 github 的文档，知乎答案的更新有点麻烦，也没法比较更改信息，就很费劲。
- 2022 年 6 月 26 日：重新调整了下格式和内容安排，同时补充了更多的参考资料和一些最新发现的有效内容。

## PyTorch 提速

> 原始文档:<https://www.yuque.com/lart/ugkv9f/ugysgn>
>
> 声明: 大部分内容来自知乎和其他博客的分享, 这里只作为一个收集罗列. 欢迎给出更多建议.

知乎回答(欢迎点赞哦):

- [pytorch dataloader 数据加载占用了大部分时间, 各位大佬都是怎么解决的? - 人民艺术家的回答 - 知乎](https://www.zhihu.com/question/307282137/answer/907835663)
- [使用 pytorch 时, 训练集数据太多达到上千万张, Dataloader 加载很慢怎么办? - 人民艺术家的回答 - 知乎](https://www.zhihu.com/question/356829360/answer/907832358)

### 预处理提速

- 尽量减少每次读取数据时的预处理操作, 可以考虑把一些固定的操作, 例如 `resize` , 事先处理好保存下来, 训练的时候直接拿来用。
- 将预处理搬到 GPU 上加速。
  - Linux 可以使用[`NVIDIA/DALI`](https://github.com/NVIDIA/DALI)。
  - 使用基于 Tensor 的图像处理操作。

### IO 提速

- mmcv 对数据的读取提供了比较高效且全面的支持：[OpenMMLab：MMCV 核心组件分析(三): FileClient](https://zhuanlan.zhihu.com/p/339190576)

#### 使用更快的图片处理

- `opencv` 一般要比 `PIL` 要快 。
  - 请注意，`PIL`的惰性加载的策略使得其看上去`open`要比`opencv`的`imread`要快，但是实际上那并没有完全加载数据。可以对`open`返回的对象调用其`load()`方法，从而手动加载数据，这时的速度才是合理的。
- 对于 `jpeg` 读取, 可以尝试 `jpeg4py`。
- 存 `bmp` 图(降低解码时间)。
- 关于不同图像处理库速度的讨论：[Python 的各种 imread 函数在实现方式和读取速度上有何区别？ - 知乎](https://www.zhihu.com/question/48762352)

#### 小图拼起来存放(降低读取次数)

对于大规模的小文件读取, 建议转成单独的文件, [可以选择考虑 `TFRecord（Tensorflow）` , `recordIO（recordIO）` , `hdf5` , `pth` , `n5` , `lmdb` 等](https://github.com/Lyken17/Efficient-PyTorch#data-loader)。

- `TFRecord` ：<https://github.com/vahidk/tfrecord>
- `lmdb` 数据库：
  - <https://github.com/Fangyh09/Image2LMDB>
  - <https://blog.csdn.net/P_LarT/article/details/103208405>
  - <https://github.com/lartpang/PySODToolBox/blob/master/ForBigDataset/ImageFolder2LMDB.py>

#### 预读取数据

预读取下一次迭代需要的数据。使用案例：

- [如何给你 PyTorch 里的 Dataloader 打鸡血 - MKFMIKU 的文章 - 知乎](https://zhuanlan.zhihu.com/p/66145913)
- [给 pytorch 读取数据加速 - 体 hi 的文章 - 知乎](https://zhuanlan.zhihu.com/p/72956595)

#### 借助内存

- 直接载到内存里面。
  - 将图片读取后存到一个固定的容器对象中。
    - YoloV5 中的[`--cache`](https://github.com/ultralytics/yolov5/blob/19f33cbae29ac2127dd877b52e228c178dda6086/utils/dataloaders.py#L521-L534)。
- 把内存映射成磁盘。

#### 借助固态

机械硬盘换成 NVME 固态。参考自[如何给你 PyTorch 里的 Dataloader 打鸡血 - MKFMIKU 的文章 - 知乎](https://zhuanlan.zhihu.com/p/66145913)

### 训练策略

#### 低精度训练

在训练中使用低精度( `FP16` 甚至 `INT8` 、二值网络、三值网络)表示取代原有精度( `FP32` )表示。

可以节约一定的显存并提速, 但是要小心一些不安全的操作如 mean 和 sum。

- 混合精度训练的介绍文章：
  - [由浅入深的混合精度训练教程](https://mp.weixin.qq.com/s/6FCumAWa8fZ1r7xwIRC9ow)
- [`NVIDIA/Apex`](https://github.com/nvidia/apex)提供的混合精度支持。
  - [PyTorch 必备神器 | 唯快不破：基于 Apex 的混合精度加速](https://mp.weixin.qq.com/s/HQnI8rzPvZN6Q_5c8d1nVQ)
  - [Pytorch 安装 APEX 疑难杂症解决方案 - 陈瀚可的文章 - 知乎](https://zhuanlan.zhihu.com/p/80386137)
- [PyTorch1.6 开始提供的`torch.cuda.amp`](https://pytorch.org/docs/stable/notes/amp_examples.html)以支持混合精度。

#### 更大的 batch

更大的 batch 在固定的 epoch 的情况下往往会带来更短的训练时间。但是大的 batch 面临着超参数的设置、显存占用问题等诸多考量，这又是另一个备受关注的领域了。

- 超参数设置
  - Accurate, large minibatch SGD: training imagenet in 1 hour，[论文](https://arxiv.org/abs/1706.02677)
- 优化显存占用
  - [Gradient Accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)
  - [Gradient Checkpointing](https://pytorch.org/docs/1.11/checkpoint.html#torch-utils-checkpoint)
    - Training deep nets with sublinear memory cost，[论文](https://arxiv.org/abs/1604.06174)
  - In-Place Operation
    - In-Place Activated BatchNorm for Memory-Optimized Training of DNNs，[论文](https://arxiv.org/abs/1712.02616)，[代码](https://github.com/mapillary/inplace_abn)

### 代码层面

#### 库设置

- 在训练循环之前设置`torch.backends.cudnn.benchmark = True`可以加速计算。由于计算不同内核大小卷积的 cuDNN 算法的性能不同，自动调优器可以运行一个基准来找到最佳算法。当你的输入大小不经常改变时，建议开启这个设置。如果输入大小经常改变，那么自动调优器就需要太频繁地进行基准测试，这可能会损害性能。它可以将向前和向后传播速度提高 1.27x 到 1.70x。
- 使用页面锁定内存，即在 DataLoader 中设定[`pin_memory=True`](https://pytorch.org/docs/stable/data.html#memory-pinning)。
- 合适的 `num_worker`，细节讨论可见[Pytorch 提速指南 - 云梦的文章 - 知乎](https://zhuanlan.zhihu.com/p/39752167)。
- [optimizer.zero_grad(set_to_none=False](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch-optim-optimizer-zero-grad)这里可以通过设置`set_to_none=True`来降低的内存占用，并且可以适度提高性能。但是这也会改变某些行为，具体可见文档。通过`model.zero_grad()`或`optimizer.zero_grad()`将对所有参数执行`memset`，并通过读写操作更新梯度。但是，将梯度设置为`None`将不会执行`memset`，并且将使用“只写”操作更新梯度。因此，设置梯度为`None`更快。
- 反向传播期间设定使用`eval`模式并使用`torch.no_grad`关闭梯度计算。
- 可以考虑使用[channels_last](https://pytorch.org/docs/stable/tensor_attributes.html#torch-memory-format)的内存格式。
- [用`DistributedDataParallel`代替`DataParallel`](https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel)。对于多 GPU 来说，即使只有单个节点，也总是优先使用 `DistributedDataParallel`而不是 `DataParallel` ，因为 `DistributedDataParallel` 应用于多进程，并为每个 GPU 创建一个进程，从而绕过 Python 全局解释器锁(GIL)并提高速度。

#### 模型

- 不要初始化任何用不到的变量，因为 PyTorch 的初始化和 `forward` 是分开的，他不会因为你不去使用，而不去初始化。
- [`@torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script)，使用 PyTroch JIT 将逐点运算融合到单个 CUDA kernel 上。
- 在使用混合精度的 FP16 时，对于所有不同架构设计，设置尺寸为 8 的倍数。
- BN 之前的卷积层可以去掉 bias。因为在数学上，bias 可以通过 BN 的均值减法来抵消。我们可以节省模型参数、运行时的内存。

#### 数据

- 将 batch size 设置为 8 的倍数，最大化 GPU 内存的使用。
- GPU 上尽可能执行 NumPy 风格的操作。
- 使用`del`释放内存占用。
- 避免不同设备之间不必要的数据传输。
- 创建张量的时候，直接指定设备，而不要创建后再传输到目标设备上。
- 使用[`torch.from_numpy(ndarray)`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html#torch-from-numpy)或者[`torch.as_tensor(data, dtype=None, device=None)`](https://pytorch.org/docs/stable/generated/torch.as_tensor.html#torch-as-tensor)，这可以通过共享内存而避免重新申请空间，具体使用细节和注意事项可参考对应文档。如果源设备和目标设备都是 CPU，`torch.from_numpy`和`torch.as_tensor`不会拷贝数据。如果源数据是 NumPy 数组，使用`torch.from_numpy`更快。如果源数据是一个具有相同数据类型和设备类型的张量，那么`torch.as_tensor`可以避免拷贝数据，这里的数据可以是 Python 的 list， tuple，或者张量。
- 使用非阻塞传输，即设定`non_blocking=True`。这会在可能的情况下尝试异步转换，例如，将页面锁定内存中的 CPU 张量转换为 CUDA 张量。

### 对优化器的优化

- 将模型参数存放到一块连续的内存中，从而减少`optimizer.step()`的时间。
  - [`contiguous_pytorch_params`](https://github.com/PhilJd/contiguous_pytorch_params)
- 使用 APEX 中的[fused building blocks](https://nvidia.github.io/apex/optimizers.html)

### 模型设计

#### CNN

- ShuffleNetV2，[论文](https://arxiv.org/pdf/1807.11164.pdf)。
  - 卷积层输入输出通道一致: 卷积层的输入和输出特征通道数相等时 MAC（内存访问消耗时间, `memory access cost` 缩写为 `MAC` ） 最小, 此时模型速度最快
  - 减少卷积分组: 过多的 group 操作会增大 MAC, 从而使模型速度变慢
  - 减少模型分支: 模型中的分支数量越少, 模型速度越快
  - 减少 `element-wise` 操作: `element-wise` 操作所带来的时间消耗远比在 FLOPs 上的体现的数值要多, 因此要尽可能减少 `element-wise` 操作。 `depthwise convolution` 也具有低 FLOPs 、高 MAC 的特点。

#### Vision Transformer

- TRT-ViT: TensorRT-oriented Vision Transformer，[论文](https://arxiv.org/abs/2205.09579)，[解读](https://www.yuque.com/lart/papers/pghqxg)。
  - stage-level：Transformer block 适合放置到模型的后期，这可以最大化效率和性能的权衡。
  - stage-level：先浅后深的 stage 设计模式可以提升性能。
  - block-level：Transformer 和 BottleNeck 的混合 block 要比单独的 Transformer 更有效。
  - block-level：先全局再局部的 block 设计模式有助于弥补性能问题。

#### 通用思路

- 降低复杂度: 例如模型裁剪和剪枝, 减少模型层数和参数规模
- 改模型结构: 例如模型蒸馏, 通过知识蒸馏方法来获取小模型

### 推理加速

#### 半精度与权重量化

在推理中使用低精度( `FP16` 甚至 `INT8` 、二值网络、三值网络)表示取代原有精度( `FP32` )表示。

- `TensorRT` 是 NVIDIA 提出的神经网络推理(Inference)引擎, 支持训练后 8BIT 量化, 它使用基于交叉熵的模型量化算法, 通过最小化两个分布的差异程度来实现
- Pytorch1.3 开始已经支持量化功能, 基于 QNNPACK 实现, 支持训练后量化, 动态量化和量化感知训练等技术
- 另外 `Distiller` 是 Intel 基于 Pytorch 开源的模型优化工具, 自然也支持 Pytorch 中的量化技术
- 微软的 `NNI` 集成了多种量化感知的训练算法, 并支持 `PyTorch/TensorFlow/MXNet/Caffe2` 等多个开源框架

更多细节可参考[有三 AI:【杂谈】当前模型量化有哪些可用的开源工具?](https://mp.weixin.qq.com/s/3uUwf9vQmQ4jkGjLxzb9aQ)。

#### 操作融合

- [模型推理加速技巧：融合 BN 和 Conv 层 - 小小将的文章 - 知乎](https://zhuanlan.zhihu.com/p/110552861)
- [网络 inference 阶段 conv 层和 BN 层的融合 - autocyz 的文章 - 知乎](https://zhuanlan.zhihu.com/p/48005099)
- [PyTorch 本身提供了类似的功能](https://pytorch.org/docs/1.3.0/quantization.html#torch.quantization.fuse_modules)

#### 重参数化（Re-Parameterization）

- [RepVGG](httsp://arxiv.org/abs/2101.03697)
  - [RepVGG|让你的 ConVNet 一卷到底，plain 网络首次超过 80%top1 精度](https://mp.weixin.qq.com/s/M4Kspm6hO3W8fXT_JqoEhA)

### 时间分析

- Python 自带了几个性能分析的模块 `profile` , `cProfile` 和 `hotshot` , 使用方法基本都差不多, 无非模块是纯 Python 还是用 C 写的。
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html?highlight=profile#module-torch.profiler) 是一种工具，可在训练和推理过程中收集性能指标。Profiler 的上下文管理器 API 可用于更好地了解哪种模型算子成本最高，检查其输入形状和堆栈记录，研究设备内核活动并可视化执行记录。

### 项目推荐

- [基于 Pytorch 实现模型压缩](https://github.com/666DZY666/model-compression):
  - 量化:8/4/2 bits(dorefa)、三值/二值(twn/bnn/xnor-net)。
  - 剪枝: 正常、规整、针对分组卷积结构的通道剪枝。
  - 分组卷积结构。
  - 针对特征二值量化的 BN 融合。

### 扩展阅读

- [pytorch dataloader 数据加载占用了大部分时间, 各位大佬都是怎么解决的? - 知乎](https://www.zhihu.com/question/307282137)
- [使用 pytorch 时, 训练集数据太多达到上千万张, Dataloader 加载很慢怎么办? - 知乎](https://www.zhihu.com/question/356829360)
- [PyTorch 有哪些坑/bug? - 知乎](https://www.zhihu.com/question/67209417)
- [Optimizing PyTorch training code](https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/)
- [26 秒单 GPU 训练 CIFAR10, Jeff Dean 也点赞的深度学习优化技巧 - 机器之心的文章 - 知乎](https://zhuanlan.zhihu.com/p/79020733)
- [线上模型加入几个新特征训练后上线, tensorflow serving 预测时间为什么比原来慢 20 多倍? - TzeSing 的回答 - 知乎](https://www.zhihu.com/question/354086469/answer/894235805)
- [深度学习模型压缩](https://www.yuque.com/lart/gw5mta)
- [今天, 你的模型加速了吗? 这里有 5 个方法供你参考(附代码解析)](https://mp.weixin.qq.com/s/_ATSwwVqigvqmDB0Y9lOAQ)
- [pytorch 常见的坑汇总 - 郁振波的文章 - 知乎](https://zhuanlan.zhihu.com/p/77952356)
- [Pytorch 提速指南 - 云梦的文章 - 知乎](https://zhuanlan.zhihu.com/p/39752167)
- [优化 PyTorch 的速度和内存效率（2022）](https://mp.weixin.qq.com/s/ShgNdizIPzeXOREoz8rgJA)

## PyTorch 节省显存

> 原始文档:<https://www.yuque.com/lart/ugkv9f/nvffyf>
>
> 整理自: Pytorch 有什么节省内存(显存)的小技巧? - 知乎 <https://www.zhihu.com/question/274635237>

### 使用 In-Place 操作

- 对于默认支持 `inplace` 的操作尽量启用。比如 `relu` 可以使用 `inplace=True` 。
- 可以将 `batchnorm` 和一些特定的激活函数打包成 [`inplace_abn`](https://github.com/mapillary/inplace_abn)。

### 损失函数

每次循环结束时删除 loss, 可以节约很少显存, 但聊胜于无。可见[Tensor to Variable and memory freeing best practices](https://discuss.pytorch.org/t/tensor-to-variable-and-memory-freeing-best-practices/6000/2)

### 混合精度

可以节约一定的显存并提速, 但是要小心一些不安全的操作如 mean 和 sum。

- 混合精度训练的介绍文章：
  - [由浅入深的混合精度训练教程](https://mp.weixin.qq.com/s/6FCumAWa8fZ1r7xwIRC9ow)
- [`NVIDIA/Apex`](https://github.com/nvidia/apex)提供的混合精度支持。
  - [PyTorch 必备神器 | 唯快不破：基于 Apex 的混合精度加速](https://mp.weixin.qq.com/s/HQnI8rzPvZN6Q_5c8d1nVQ)
  - [Pytorch 安装 APEX 疑难杂症解决方案 - 陈瀚可的文章 - 知乎](https://zhuanlan.zhihu.com/p/80386137)
- [PyTorch1.6 开始提供的`torch.cuda.amp`](https://pytorch.org/docs/stable/notes/amp_examples.html)以支持混合精度。

### 管理不需要反向传播的操作

- 对于不需要反向传播的前向阶段，如验证和推理期间，使用 `torch.no_grad` 来包裹代码。
  - 注意 `model.eval()` 不等于 `torch.no_grad()` , 请看如下讨论: ['model.eval()' vs 'with torch.no_grad()'](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- 将不需要计算梯度的变量的 `requires_grad`设为 `False`, 让变量不参与梯度的后向传播，以减少不必要的梯度的显存占用。
- 移除不需要计算的梯度路径：
  - [Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models](https://arxiv.org/abs/2203.16755)，解读可见：
    - <https://www.yuque.com/lart/papers/xu5t00>
    - <https://blog.csdn.net/P_LarT/article/details/124978961>

### 显存清理

- `torch.cuda.empty_cache()` 这是 `del` 的进阶版, 使用 `nvidia-smi` 会发现显存有明显的变化. 但是训练时最大的显存占用似乎没变. 大家可以试试: [How can we release GPU memory cache?](https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530)
- 可以使用 `del` 删除不必要的中间变量, 或者使用 `replacing variables` 的形式来减少占用.

### 梯度累加（Gradient Accumulation）

把一个 `batchsize=64` 分为两个 32 的 batch，两次 forward 以后，backward 一次。但会影响 `batchnorm` 等和 `batchsize` 相关的层。

在[PyTorch 的文档](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)中提到了梯度累加与混合精度并用的例子。

使用梯度累加技术可以对分布式训练加速，这可以参考：[[原创][深度][PyTorch] DDP 系列第三篇：实战与技巧 - 996 黄金一代的文章 - 知乎](https://zhuanlan.zhihu.com/p/250471767)

### 梯度检查点（Gradient Checkpointing）

PyTorch 中提供了[`torch.utils.checkpoint`](https://pytorch.org/docs/1.11/checkpoint.html#torch-utils-checkpoint)。这是通过在反向传播期间，在每个检查点位置重新执行一次前向传播来实现的。

论文[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)基于梯度检查点技术，将显存从 O(N) 降到了 O(sqrt(N))。对于越深的模型, 这个方法省的显存就越多, 且速度不会明显变慢。

- [PyTorch 之 Checkpoint 机制解析](https://www.yuque.com/lart/ugkv9f/azvnyg)
- [torch.utils.checkpoint 简介 和 简易使用](https://blog.csdn.net/one_six_mix/article/details/93937091)
- [Sublinear Memory Cost 的一份 PyTorch 实现](https://github.com/Lyken17/pytorch-memonger)，参考自：[Pytorch 有什么节省内存(显存)的小技巧? - Lyken 的回答 - 知乎](https://www.zhihu.com/question/274635237/answer/755102181)

### 相关工具

- These codes can help you to detect your GPU memory during training with Pytorch. [https://github.com/Oldpan/Pytorch-Memory-Utils](https://github.com/Oldpan/Pytorch-Memory-Utils)
- Just less than nvidia-smi? [https://github.com/wookayin/gpustat](https://github.com/wookayin/gpustat)

### 参考资料

- [Pytorch 有什么节省内存(显存)的小技巧? - 郑哲东的回答 - 知乎](https://www.zhihu.com/question/274635237/answer/573633662)
- [浅谈深度学习: 如何计算模型以及中间变量的显存占用大小](https://oldpan.me/archives/how-to-calculate-gpu-memory)
- [如何在 Pytorch 中精细化利用显存](https://oldpan.me/archives/how-to-use-memory-pytorch)
- [Pytorch 有什么节省显存的小技巧? - 陈瀚可的回答 - 知乎](https://www.zhihu.com/question/274635237/answer/756144739)
- [PyTorch 显存机制分析 - Connolly 的文章 - 知乎](https://zhuanlan.zhihu.com/p/424512257)

## 其他技巧

### 重现

可关注文档中[相关章节](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility)。

#### 强制确定性操作

[避免使用非确定性算法](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms)。

PyTorch 中，[`torch.use_deterministic_algorithms()`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms)可以强制使用确定性算法而不是非确定性算法，并且如果已知操作是非确定性的（并且没有确定性的替代方案），则会抛出错误。

#### 设置随机数种子

```python
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
```

参考自<https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%E9%9A%8F%E6%9C%BA%E7%A7%8D%E5%AD%90/>

#### PyTorch 1.9 版本前 DataLoader 中的隐藏 BUG

具体细节可见[可能 95%的人还在犯的 PyTorch 错误 - serendipity 的文章 - 知乎](https://zhuanlan.zhihu.com/p/523239005)

解决方法可参考[文档](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)：

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

DataLoader(..., worker_init_fn=seed_worker)
```
