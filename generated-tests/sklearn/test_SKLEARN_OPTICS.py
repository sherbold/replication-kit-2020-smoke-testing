import unittest
import pandas as pd
import numpy as np
import threading
import functools
import inspect
import math
import warnings
import traceback

from parameterized import parameterized
from scipy.io.arff import loadarff
from scipy.stats import ttest_1samp, ks_2samp
from sklearn.cluster import OPTICS


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

class test_SKLEARN_OPTICS(unittest.TestCase):

    params = [("{'p':2,'xi':0.05,'min_samples':2,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':2,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':8,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':8,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':0.1,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':0.1,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':10.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':10.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'cityblock','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'cityblock','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'cosine','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'cosine','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'euclidean','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'euclidean','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'l1','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'l1','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'l2','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'l2','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'manhattan','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'manhattan','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':1,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':1,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':3,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':3,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':4,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':4,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'dbscan','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'dbscan','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':0.1,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':0.1,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':1.0,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':1.0,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':10.0,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':10.0,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.01,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.01,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.1,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.1,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':False,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':False,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':2,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':2,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':10,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':10,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'ball_tree',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'ball_tree',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'kd_tree',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'kd_tree',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'brute',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'brute',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':10,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':10,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':90,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':90,'predecessor_correction':True,'metric':'minkowski','n_jobs':None,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':-1,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':-1,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
              ("{'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':2,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}", {'p':2,'xi':0.05,'min_samples':5,'leaf_size':50,'predecessor_correction':True,'metric':'minkowski','n_jobs':2,'max_eps':1.0,'eps':None,'min_cluster_size':None,'cluster_method':'xi','algorithm':'auto',}),
             ]

    def assert_morphtest(self, evaluation_type, testcase_name, iteration, deviations_clust, pval_ttest, deviations_pvals, no_exception, exception_type, exception_message, exception_stacktrace):
        if no_exception:
            if evaluation_type=='clust_exact':
                self.assertEqual(deviations_clust, 0)
            elif evaluation_type=='clust_stat':
                self.assertTrue(pval_ttest > 0.05)
            elif evaluation_type=='score_stat':
                self.assertEqual(deviations_pvals, 0)
            else:
                raise ValueError('invalid evaluation_type: %s (allowed: clust_exact, clust_stat, score_stat)' % evaluation_type)
        else:
            raise RuntimeError('%s encountered: %s %s' % exception_type, exception_message, exception_stacktrace)

    def flip_same_clusters(self, morph_clusters, expected_clusters):
        flipped_clusters = {}
        for morph_cluster in morph_clusters:
            flipped = False
            for exp_cluster in expected_clusters:
                if morph_clusters[morph_cluster] == expected_clusters[exp_cluster]:
                    flipped_clusters[exp_cluster] = expected_clusters[exp_cluster]
                    flipped = True
                    break
            if not flipped:
                flipped_clusters[morph_cluster] = morph_clusters[morph_cluster]
        return flipped_clusters

    def create_cluster_map(self, data):
        cluster_map = {}
        for i, c in enumerate(data):
            if c not in cluster_map:
                cluster_map[c] = [i]
            else:
                cluster_map[c].append(i)
        return cluster_map

    def create_scores_map(self, cluster_map, scores):
        scores_map = {}
        for c in cluster_map:
            for i in cluster_map[c]:
                if c not in scores_map:
                    scores_map[c] = [scores[i]]
                else:
                    scores_map[c].append(scores[i])
        return scores_map

    @parameterized.expand(params)
    @timeout(21600)
    def test_Uniform(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Uniform_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_MinFloat(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MinFloat_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_VerySmall(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/VerySmall_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_MinDouble(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MinDouble_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_MaxFloat(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MaxFloat_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_VeryLarge(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/VeryLarge_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_MaxDouble(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MaxDouble_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_Split(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Split_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_LeftSkew(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/LeftSkew_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_RightSkew(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/RightSkew_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_OneClass(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/OneClass_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_Bias(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Bias_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_Outlier(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Outlier_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_Zeroes(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Zeroes_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_RandomNumeric(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/RandomNumeric_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            
    @parameterized.expand(params)
    @timeout(21600)
    def test_DisjointNumeric(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/DisjointNumeric_%i_training.arff' % iter)
            data_df = pd.DataFrame(data)
            data_df = pd.get_dummies(data_df)
            
            clusterer = OPTICS(**kwargs)
            np.random.seed(42)
            clusterer.fit_predict(data_df.values)
            


if __name__ == '__main__':
    unittest.main()
#    with open('results.xml', 'wb') as output:
#        unittest.main(
#            testRunner=xmlrunner.XMLTestRunner(output=output),
#            failfast=False, buffer=False, catchbreak=False)