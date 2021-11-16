# DataCLUE Toolkit

为了方便各个算法之间的整合，这里提供了一套统一的输入输出接口

```python
from dckit.utils import read_datasets, random_split_data
from dckit.eval import evaluate

data = read_datasets()
# TODO 对数据进行处理，处理完的文件存储在对应文件夹下


random_split_data(data)
f1 = evaluate()
```