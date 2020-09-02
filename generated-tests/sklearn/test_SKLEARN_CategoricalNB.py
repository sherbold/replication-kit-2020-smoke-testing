import unittest
import xmlrunner
import pandas as pd
import numpy as np
import threading
import functools
import inspect
import math
import traceback
import warnings

from parameterized import parameterized
from scipy.io.arff import loadarff
from scipy.stats import chisquare, ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB


class TestTimeoutException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# thanks to https://gist.github.com/vadimg/2902788
def timeout(duration, default=None):
    def decorator(func):
        class InterruptableThread(threading.Thread):
            def __init__(self, args, kwargs):
                threading.Thread.__init__(self)
                self.args = args
                self.kwargs = kwargs
                self.result = default
                self.daemon = True
                self.exception = None

            def run(self):
                try:
                    self.result = func(*self.args, **self.kwargs)
                except Exception as e:
                    self.exception = e

        @functools.wraps(func)
        def wrap(*args, **kwargs):
            it = InterruptableThread(args, kwargs)
            it.start()
            it.join(duration)
            if it.is_alive():
                raise TestTimeoutException('timeout after %i seconds for test %s' % (duration, func))
            if it.exception:
                raise it.exception
            return it.result
        return wrap
    return decorator

class test_SKLEARN_CategoricalNB(unittest.TestCase):

    params = [("{'fit_prior':True,'alpha':0.0,}", {'fit_prior':True,'alpha':0.0,}),
              ("{'fit_prior':True,'alpha':0.5,}", {'fit_prior':True,'alpha':0.5,}),
              ("{'fit_prior':True,'alpha':1.0,}", {'fit_prior':True,'alpha':1.0,}),
              ("{'fit_prior':True,'alpha':1.5,}", {'fit_prior':True,'alpha':1.5,}),
              ("{'fit_prior':False,'alpha':1.0,}", {'fit_prior':False,'alpha':1.0,}),
             ]

    def assert_morphtest(self, evaluation_type, testcase_name, iteration, deviations_class, deviations_score, pval_chisquare, pval_kstest):
        if evaluation_type=='score_exact':
            self.assertEqual(deviations_score, 0)
        elif evaluation_type=='class_exact':
            self.assertEqual(deviations_class, 0)
        elif evaluation_type=='score_stat':
            self.assertTrue(pval_kstest>0.05)
        elif evaluation_type=='class_stat':
            self.assertTrue(pval_chisquare>0.05)
        else:
            raise ValueError('invalid evaluation_type: %s (allowed: score_exact, class_exact, score_stat, class_stat' % evaluation_type)

    @parameterized.expand(params)
    @timeout(21600)
    def test_Categorical(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Categorical_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Categorical_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = CategoricalNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_RandomCategorial(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/RandomCategorial_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/RandomCategorial_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = CategoricalNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_DisjointCategorical(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/DisjointCategorical_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/DisjointCategorical_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = CategoricalNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_StarvedMany(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/StarvedMany_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/StarvedMany_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = CategoricalNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_StarvedBinary(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/StarvedBinary_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/StarvedBinary_%i_test.arff' % iter)
            lb_make = LabelEncoder()
            data_df = pd.DataFrame(data)
            data_df["classAtt"] = lb_make.fit_transform(data_df["classAtt"])
            data_df = pd.get_dummies(data_df)
            
            testdata_df = pd.DataFrame(data)
            testdata_df["classAtt"] = lb_make.fit_transform(testdata_df["classAtt"])
            testdata_df = pd.get_dummies(testdata_df, sparse=True)
            
            classIndex = -1
            for i, s in enumerate(data_df.columns):
                if 'classAtt' in s:
                    classIndex = i
            
            classifier = CategoricalNB(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            



if __name__ == '__main__':
    unittest.main()
#    with open('results.xml', 'wb') as output:
#        unittest.main(
#            testRunner=xmlrunner.XMLTestRunner(output=output),
#            failfast=False, buffer=False, catchbreak=False)