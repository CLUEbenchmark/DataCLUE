# 数据增强

对输入数据进行增强

# 参数说明

这里只用了增强次数作为参数

# 参数选择实验

|增强次数 | 0  | 1  | 3  |  5 |10|
|---|---|---|---|---|---|
| Marco-F1| .7278 | .7388 | .7462 | .7363 | .6694 |


# 可能问题
这里的数据增强部分用了[synonyms](https://github.com/chatopera/Synonyms)，其中资源下载可能存在问题。如果存在问题请按照如下设置：
```bash
export SYNONYMS_WORD2VEC_BIN_URL_ZH_CN=https://gitee.com/chatopera/cskefu/attach_files/610602/download/words.vector.gz
pip install -U synonyms
python -c "import synonyms" # download word vectors file
```

