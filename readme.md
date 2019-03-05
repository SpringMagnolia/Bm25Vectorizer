# BM25的Vectorizer
基于sciket-learn的TfidfVectorizer，继承修改得到的Bm25Vectorizer
使用方法和sciket-learn的方法完全相同

```python
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
    
```
输出如下：
```
Bm25 result: [[0.         0.         0.         0.47878333 0.         0.
  0.         0.         0.         0.8779331 ]
 [0.         0.         0.         0.35073401 0.         0.66218791
  0.         0.         0.66218791 0.        ]
 [0.         0.         0.         0.         0.70710678 0.
  0.70710678 0.         0.         0.        ]
 [0.47038081 0.47038081 0.47038081 0.         0.23975776 0.
  0.23975776 0.47038081 0.         0.        ]]
*****************************************************
Tfidf result: [[0.         0.         0.         0.6191303  0.         0.
  0.         0.         0.         0.78528828]
 [0.         0.         0.         0.48693426 0.         0.61761437
  0.         0.         0.61761437 0.        ]
 [0.         0.         0.         0.         0.70710678 0.
  0.70710678 0.         0.         0.        ]
 [0.43671931 0.43671931 0.43671931 0.         0.34431452 0.
  0.34431452 0.43671931 0.         0.        ]]

Process finished with exit code 0

```