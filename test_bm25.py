from BM25Vectorizer import Bm25Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    # format_weibo(word=False)
    # format_xiaohuangji_corpus(word=True)
    bm_vec = Bm25Vectorizer()
    tf_vec = TfidfVectorizer()
    # 1. 原始数据
    data = [
        'hello world',
        'oh hello there',
        'Play it',
        'Play it again Sam,24343,123',
    ]

    # 2. 原始数据向量化
    bm_vec.fit(data)
    tf_vec.fit(data)
    features_vec_bm = bm_vec.transform(data)
    features_vec_tf = tf_vec.transform(data)
    print("Bm25 result:",features_vec_bm.toarray())
    print("*"*100)
    print("Tfidf result:",features_vec_tf.toarray())