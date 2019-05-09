"""
实现BM25的vectorizer
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse as sp
from sklearn.feature_extraction.text import _document_frequency
from sklearn.preprocessing import normalize

class Bm25Vectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"\w+",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(Bm25Vectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = Bm25Transformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self._tfidf.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TfidfVectorizer
        """
        self._check_params()
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(Bm25Vectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)


class Bm25Transformer(BaseEstimator, TransformerMixin):

    def __init__(self,k=1.2,b=0.75, norm="l2", use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.k = k
        self.b = b
        ######### tfidf的代码###########
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        X = X.toarray()
        self.avdl = X.sum()/X.shape[0] #句子的平均长度
        # print("原来的fit的数据：\n",X)

        #计算每个词语的tf的值
        self.tf = X.sum(0)/X.sum()  #[M] #M表示总词语的数量
        self.tf = self.tf.reshape([1,self.tf.shape[0]]) #[1,M]
        # print("tf\n",self.tf)
        ######       原来tfidf的代码  ######

        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X).astype(dtype)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(n_samples / df) + 1
            self._idf_diag = sp.diags(idf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        ########### 计算中间项  ###############
        cur_tf = np.multiply(self.tf, X.toarray()) #[N,M] #N表示数据的条数，M表示总词语的数量
        norm_lenght = 1 - self.b + self.b*(X.toarray().sum(-1)/self.avdl) #[N] #N表示数据的条数
        norm_lenght = norm_lenght.reshape([norm_lenght.shape[0],1]) #[N,1]
        X = (self.k+1)*cur_tf /(cur_tf +self.k*norm_lenght)
        ############# 结算结束  ################

        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work

            X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(value, diags=0, m=n_features,
                                    n=n_features, format='csr')


if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer
    Bm25 = Bm25Vectorizer(norm="l2")
    tfidf = TfidfVectorizer(norm="l2")
    Bm25.fit(["hello world hello","hello 2","1 w 2 ","da sad 3"])
    ret = Bm25.transform(["hello ,hello,world","hello","1 w 2"])
    print("ret:\n",ret.toarray())
    print("*"*100)

    tfidf.fit(["hello world hello", "hello 2", "1 w 2 ", "da sad 3"])
    ret = tfidf.transform(["hello ,hello,world","hello,world","hello","1 w 2"])
    print(ret.toarray())
