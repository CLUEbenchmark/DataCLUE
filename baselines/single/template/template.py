from dckit import read_datasets
from dckit.evaluate import evaluate


def template(data):
    """
        输入读取的字典，输出还是这个字典，但是修改其内容，如果修改了标签请注意同时修改label_des和label字段
    """
    # TODO add your code here
    return data


def main():
    data = read_datasets()
    template(data)
    f1 = evaluate()
    print('Macro-F1=', f1)
    return f1


if __name__ == '__main__':
    main()
