# DataCLUE Toolkit

[安装](#安装) | [使用](#使用) | [示例](#示例) | [贡献](#贡献) | [**References**](#references)

为了方便各个算法之间的整合，这里提供了一套统一的输入输出接口。
并且提供了一些辅助函数，帮助大家更方便地使用DataCLUE。
(我们鼓励大家用dckit进行开发，以更好的实现不同算法的共享。但是你也完全可以自己实现相应功能完成DataCLUE的任务)。

# Updates
[Nov 16, 2021] First version of dckit is released.

# 安装
在DataCLUE目录下

`pip install -e .`

# 使用
```python
from dckit import read_datasets, random_split_data, evaluate

data = read_datasets(dataset='CIC')  # 读取数据
# TODO 对数据进行处理，这里example_transform 是你需要实现的变换
data = example_transform(data)

random_split_data(data, test_size=2000, seed=0)  # 随机切分数据到训练、测试集
f1 = evaluate()  # 运行模型并返回相应的结果
```

# 示例
我们在中`baseline`实现了几个策略都用到了dckit，比如你可以看`baseline/single/data_aug`或其它相应baseline代码中的实现


# 贡献
- 如果你觉得dckit缺少一些通用的基本功能，你可以提一个issue。
- 如果你已经实现了dckit的扩展功能，欢迎开启一个PR。

# References
```bib
@article{xu2021dataclue,
      title={DataCLUE: A Benchmark Suite for Data-centric NLP}, 
      author={Liang Xu and Jiacheng Liu and Xiang Pan and Xiaojing Lu and Xiaofeng Hou},
      year={2021},
      eprint={2111.08647},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
