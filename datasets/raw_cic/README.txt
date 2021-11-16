DataCLUE: 以数据为中心的AI测评
项目地址：https://github.com/CLUEbenchmark/DataCLUE

意图识别任务，类别数：118
train.json：1万，包含部分有问题的数据
dev.json：2000条，包含部分有问题的数据
test_public.json：2000条，仅用于学术和实验，和作为训练完后的效果评估。不能用于训练；高质量数据（标注准确率95%或以上）

train.json/dev.json: 含有噪声的数据，都含有一定比例有标注错误的标签；


1) 请做实验，报告你的方法下改进后的数据集(train.json/dev.json)，在test_public.json上做最终测试(2个数值，f1_macro & f1_micro)；或
2) 你也可以提交到CLUE平台：www.CLUEbenchmarks.com， 或发送邮件包含训练集和验证集的压缩包到邮箱。联系邮箱：CLUEbenchmark@163.com

www.CLUEbenchmarks.com
