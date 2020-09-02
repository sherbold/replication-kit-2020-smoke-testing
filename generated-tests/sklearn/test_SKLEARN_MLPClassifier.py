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
from sklearn.neural_network import MLPClassifier


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

class test_SKLEARN_MLPClassifier(unittest.TestCase):

    params = [("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,50,10),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,50,10),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(1, 1),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(1, 1),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(10, 20),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(10, 20),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'identity','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'identity','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'logistic','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'logistic','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'tanh','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'tanh','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'lbfgs',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'lbfgs',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'sgd',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'sgd',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.00001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.00001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':1,'beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':1,'beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':10,'beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':10,'beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':19,'beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':19,'beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'invscaling','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'invscaling','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'adaptive','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'adaptive','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.01,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.01,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.0001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.0001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.1,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.1,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.5,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.5,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.9,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.9,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':False,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':False,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.1,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.1,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.5,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.5,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':True,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':True,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.5,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.5,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.9,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.9,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.0,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.0,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.45,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.45,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.90,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.90,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.0,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.0,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.45,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.45,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.90,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.90,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':1,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':1,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':9,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':9,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':15000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':10000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':10000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
              ("{'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':20000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}", {'beta_1':0.9,'batch_size':'auto','beta_2':0.9,'n_iter_no_change':5,'early_stopping':False,'hidden_layer_sizes':(100,),'momentum':0.9,'tol':0.0001,'epsilon':0.00000001,'max_fun':20000,'alpha':0.0001,'power_t':0.4,'validation_fraction':0.1,'activation':'relu','learning_rate_init':0.001,'shuffle':True,'learning_rate':'constant','solver':'adam',}),
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
    def test_Uniform(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Uniform_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Uniform_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MinFloat(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MinFloat_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/MinFloat_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_VerySmall(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/VerySmall_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/VerySmall_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MinDouble(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MinDouble_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/MinDouble_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MaxFloat(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MaxFloat_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/MaxFloat_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_VeryLarge(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/VeryLarge_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/VeryLarge_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_MaxDouble(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/MaxDouble_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/MaxDouble_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Split(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Split_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Split_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_LeftSkew(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/LeftSkew_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/LeftSkew_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_RightSkew(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/RightSkew_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/RightSkew_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_OneClass(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/OneClass_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/OneClass_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Bias(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Bias_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Bias_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Outlier(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Outlier_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Outlier_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_Zeroes(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/Zeroes_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/Zeroes_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_RandomNumeric(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/RandomNumeric_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/RandomNumeric_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            

    @parameterized.expand(params)
    @timeout(21600)
    def test_DisjointNumeric(self, name, kwargs):
        for iter in range(1,1+1):
            data, meta = loadarff('smokedata/DisjointNumeric_%i_training.arff' % iter)
            testdata, testmeta = loadarff('smokedata/DisjointNumeric_%i_test.arff' % iter)
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
            
            classifier = MLPClassifier(**kwargs)
            np.random.seed(42)
            classifier.fit(np.delete(data_df.values, classIndex, axis=1),data_df.values[:,classIndex])
            classifier.predict(np.delete(testdata_df.values, classIndex, axis=1))
            



if __name__ == '__main__':
    unittest.main()
#    with open('results.xml', 'wb') as output:
#        unittest.main(
#            testRunner=xmlrunner.XMLTestRunner(output=output),
#            failfast=False, buffer=False, catchbreak=False)