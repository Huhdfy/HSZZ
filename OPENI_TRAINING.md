# 启智训练服务：CPU 训练与 TensorBoard

本项目从模拟器在线采样，不依赖外部训练数据；`openi_train.py` 是训练服务的启动文件，使用实例 CPU，不调用昇腾 NPU。训练代码和数值设计保持在 `src/clasher_new/train.py` 中。

## 首次准备

1. 将此 GitHub 仓库导入或镜像为启智项目，确认项目代码分支包含 `openi_train.py`、`src/clasher_new/*.py` 和同目录 JSON 数据文件。
2. 选择已安装 **PyTorch** 的 Linux 镜像。还需安装 `requirements-train-cpu.txt` 中的依赖；建议先在调试任务中运行 `python -m pip install -r requirements-train-cpu.txt`，测试通过后保存镜像，再给训练任务选择该镜像。不要选择仅有 MindSpore、没有 PyTorch 的镜像。
3. 创建训练任务：项目选导入的项目，代码分支选已上传的分支，启动文件选仓库根目录的 `openi_train.py`，模型首次训练留空。表单若强制要求数据集，可关联一个项目内的小型占位数据集，训练代码不会读取它。
4. 勾选“查看 TensorBoard 可视化结果”。当前代码默认优先把模型、checkpoint、事件文件写到容器内 `/model/clash-royale/`；若该目录不存在则尝试 `/result/clash-royale/`。平台版本可能不同，务必以创建任务页面的“查看样例”给出的**持久化输出目录**为准。若两者都不存在，入口会立即报错而不会把产物误写入临时目录。

日志位于 `<输出目录>/tensorboard/`，模型位于 `<输出目录>/latest.zip`，周期 checkpoint 位于 `<输出目录>/checkpoints/`。启动日志会打印实际输出目录和 TensorBoard 目录。任务开始后到任务详情页查看运行日志，等待首次 PPO 更新完成，再打开平台的 TensorBoard 可视化入口；首次更新前可能尚无曲线。若页面没有曲线，先核对日志打印的目录中是否有 `events.out.tfevents.*` 文件，以及平台 TensorBoard 是否扫描该目录。

如平台将“运行参数”附加到启动文件命令行，可填 `--output-dir /平台确认的输出目录/本项目 --timesteps 8192` 做一次短测；确认无误后再提交长任务。平台若不支持命令行参数，可以在镜像或任务环境中设置 `TRAIN_OUTPUT_DIR`、`TRAIN_TOTAL_TIMESTEPS`，或按任务样例修改入口。默认训练目标仍是 500 万步，截图中的 NPU 实例即使只用 CPU 也会按实例规格扣积分，不要直接用长任务试错。

重新创建的训练任务未必保留上一个任务的文件；要续训，请先把先前任务的 `latest.zip` 放到新任务的 `<输出目录>/latest.zip`。当前没有自动从“模型”输入项复制 checkpoint 的逻辑。
