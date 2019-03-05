
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,_document_frequency
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
import numpy as np
import scipy.sparse as sp

class Bm25Transformer(BaseEstimator, TransformerMixin):

    def __init__(self,k=1.2,b=0.75, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.k = k
        self.b = b
        ##################以下是TFIDFtransform代码##########################
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
        # self.avdl = np.mean(np.count_nonzero(X.toarray(), axis=-1))

        self.avdl = np.mean(np.count_nonzero(X.toarray(), axis=-1))
        ##################以下是TFIDFtransform代码##########################
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')

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
        # TODO 对x乘上中间项
        # X [N,features]
        X = X.toarray()
        cur_d = np.count_nonzero(X, axis=-1)  # [N]
        cur_d = np.reshape(cur_d, (cur_d.shape[0], 1))  # [N,1]

        normalize_line_length = 1 - self.b + self.b * (cur_d / self.avdl)
        mid_part = (self.k + 1) * X / (X + self.k * normalize_line_length)
        # element -wize product
        X = np.multiply(X, mid_part)

        ##################以下是TfidfTransform代码##########################

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

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
        ##################以下是TFIDFtransform代码##########################
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))




class Bm25Vectorizer(TfidfVectorizer):
    def __init__(self,k=1.2,b=0.75, norm="l2", use_idf=True, smooth_idf=True,sublinear_tf=False,*args,**kwargs):
        super(Bm25Vectorizer,self).__init__(*args,**kwargs)
        self._tfidf = Bm25Transformer(k=k,b=b,norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    @property
    def k(self):
        return self._tfidf.k

    @k.setter
    def k(self, value):
        self._tfidf.k = value

    @property
    def b(self):
        return self._tfidf.b

    @b.setter
    def b(self, value):
        self._tfidf.b = value

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.
        """
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.
        """
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.
        """
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(Bm25Vectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)