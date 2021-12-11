from concurrent.futures import ProcessPoolExecutor, as_completed
from textda.data_expansion import data_expansion


def expand(sentence, label, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.2, num_aug=9):
    res = data_expansion(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs,
                         p_rd=p_rd, num_aug=num_aug)
    return res, [label] * len(res)


def parallel_expansion(sentences, labels, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.2, num_aug=9, workers=16):
    '''
    if you set alpha_ri and alpha_rs is 0 that means use linear classifier for it, and insensitive to word location
    :param sentences: input sentence list
    :param labels: input label list
    :param alpha_sr: Replace synonym control param. bigger means more words are Replace
    :param alpha_ri: Random insert. bigger means more words are Insert
    :param alpha_rs: Random swap. bigger means more words are swap
    :param p_rd: Random delete. bigger means more words are deleted
    :param num_aug: How many times do you repeat each method
    :param workers: Number of process
    :return:
    '''
    assert len(sentences) == len(labels)
    res_sentences = []
    res_labels = []
    with ProcessPoolExecutor(max_workers=workers) as t:
        obj_list = []
        for idx, sentence in enumerate(sentences):
            obj = t.submit(expand, sentence, labels[idx], alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs,
                           p_rd=p_rd, num_aug=num_aug)
            obj_list.append(obj)

        for future in as_completed(obj_list):
            res = future.result()
            res_sentences.extend(res[0])
            res_labels.extend(res[1])
    return res_sentences, res_labels
