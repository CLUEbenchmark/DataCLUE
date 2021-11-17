# DataCLUE Toolkit

为了方便各个算法之间的整合，这里提供了一套统一的输入输出接口。并且提供了一些辅助函数，帮助大家更方便地使用DataCLUE。

```python
from dckit import read_datasets, random_split_data
from dckit.evaluate import evaluate

data = read_datasets()
# TODO 对数据进行处理
data = example_transform(data)

random_split_data(data)
f1 = evaluate()
```
