# Progress
## fixed bugs


1. nn.LSTM返回的是一个tuple类型：(Tensor, Tuple(Tensor, Tensor)), 这个在`main_with_runtime.py`中不被支持（Stage的返回只能是单个或多个（tuple形式）的Tensor）
2. 空的Stage中没有module，也就没有参数，创建Adam的时候会报错：参数为空。
3. PipeDream支持部分stages做replica，并且通过`get_messaging_index()`来做轮询确定从哪个节点接受/发送数据。在rank0 --> rank1/rank2; rank0 --> rank3; rank1/rank2 --> rank3 这种比较交错的切分方式下，`get_messaging_index()`并不能返回一个正确的值，准确的说是rank3接受来自rank0的数据，生成的index应该一直是0，但是程序会返回1。