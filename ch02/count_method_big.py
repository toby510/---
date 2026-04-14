# coding: utf-8
"""
代码功能：
PPMI->SVD降维->计算相似度
"""
import sys

sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
# 矩阵W是 10000 × 10000 的（PTB 数据集词汇量约 10000），所以需要降维
W = ppmi(C, verbose=True)
print('calculating SVD ...')
try:
    # truncated SVD (fast!)：截断的SVD,会快一些，n_components=wordvec_size表示只算前 100 维（快、省内存、NLP 必备）
    from sklearn.utils.extmath import randomized_svd

    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (slow)：维度全部算出，比较慢
    U, S, V = np.linalg.svd(W)
# 取所有行，前wordvec_size列的数据
word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
