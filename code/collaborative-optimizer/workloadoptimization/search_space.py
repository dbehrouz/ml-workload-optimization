from openml import datasets, tasks, runs, flows, setups, config, evaluations
import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope, as_apply

class SearchSpaceDesigner:
    searchSpace = {}
    constantParams = {}
  
    def __init__(self, experimentObjects):
        possibleSearchSpace = {}
        paramRange = {}
        for e in experimentObjects:
            if possibleSearchSpace.has_key(e.flow):
                searchSpace = possibleSearchSpace[e.flow]
                possibleSearchSpace[e.flow] = self.updateSearchSpace(searchSpace, e)
            else:
                searchSpace = {}
                possibleSearchSpace[e.flow] = self.updateSearchSpace(searchSpace, e)
        for pipeline in possibleSearchSpace.keys():
            dynamicParams = {}
            # For now we are not interested in constant params, so we ignore them
            constantParams = {}
            for k in possibleSearchSpace[pipeline].keys():
                try:
                    size = len(set(possibleSearchSpace[pipeline][k]))
                    if size == 1:
                        constantParams[k] = possibleSearchSpace[pipeline][k][0]
                    else:
                        # is it safe to assume parameters are unique
                        dynamicParams[k] = list(set(possibleSearchSpace[pipeline][k]))
                except: 
                    constantParams[k] = possibleSearchSpace[pipeline][k][0] 

            paramRange[pipeline] = dynamicParams
            self.constantParams[pipeline] = constantParams
            
            # Manual fixes
            if pipeline == 8568:
                self.constantParams[8568]['randomforestclassifier__max_depth'] = None
        
        recommender = SpaceRecommender()
        self.searchSpace = recommender.recommendSpace(paramRange)
    
    def updateSearchSpace(self,searchSpace, experimentObject):
        for k,v in experimentObject.extractParams().iteritems():
            if searchSpace.has_key(k):
                searchSpace[k].append(v)
            else:
                searchSpace[k] = [v]
        return searchSpace
    
    def getConstantParams(self):
        return self.constantParams
    
    def getSearchSpace(self):
        return self.searchSpace
    
                
class SpaceRecommender:
    
    # Eventually this will change into our novel contribution
    # Where we automatically extract the perfect search space
    def recommendSpace(self, paramRange):
        paramSpace = {}
        for k,v in paramRange.iteritems():
            if k == 8568:
                space = self.space8568(paramRange[8568])
                paramSpace[8568] = space
            if k == 8353:
                space = self.space8353(paramRange[8353])
                paramSpace[8353] = space
            if k == 8315:
                space = self.space8315(paramRange[8315])
                paramSpace[8315] = space
            if k == 7707:
                space = self.space7707(paramRange[7707])
                paramSpace[7707] = space
        return paramSpace
      
    ## HARDCODED SPACE
    def space8568(self,points):
        space = {}
        space['conditionalimputer__strategy'] = hp.choice('conditionalimputer__strategy', [u'mean', u'median', u'most_frequent'])
        array = np.array(points['randomforestclassifier__max_features'])
        array = np.delete(array, np.where(array == 'auto'), axis=0)
        array = array.astype(np.float)
        space['randomforestclassifier__max_features'] = hp.uniform('randomforestclassifier__max_features', min(array), max(array))
        space['randomforestclassifier__min_samples_leaf'] = hp.choice('randomforestclassifier__min_samples_leaf',range(1,21))
        space['onehotencoder__sparse'] = hp.choice('onehotencoder__sparse',[True, False])
        space['randomforestclassifier__min_impurity_split'] = hp.choice('randomforestclassifier__min_impurity_split',[None, 1e-07])
        space['randomforestclassifier__min_samples_split'] = hp.choice('randomforestclassifier__min_samples_split', range(2,21))
        #array = np.array(points['randomforestclassifier__max_depth'])
        #array = np.delete(array, np.where(array == None), axis=0)
        #array = array.astype(np.float)
        #space['randomforestclassifier__max_depth'] = hp.uniform('randomforestclassifier__max_depth', min(array), max(array))
        space['randomforestclassifier__criterion'] = hp.choice('randomforestclassifier__criterion',[u'gini', u'entropy'])
        space['randomforestclassifier__bootstrap'] = hp.choice('randomforestclassifier__bootstrap',[True, False])
        return space 
    def space8353(self,points):
        space = {}
        space['clf__gamma'] = hp.lognormal('clf__gamma', 0.04, 1.0)
        space['clf__tol'] = hp.lognormal('clf__tol',-7, 1)
        space['clf__shrinking'] = hp.choice('clf__shrinking',[True, False])
        space['clf__C'] = hp.lognormal('clf__C',7, 1.1)
        space['clf__coef0'] = hp.uniform('clf__coef0', -1, 1)
        return space

    def space8315(self,points):
        space = {}
        space['clf__bootstrap'] = hp.choice('clf__bootstrap',[True, False])
        space['clf__criterion'] = hp.choice('clf__criterion',[u'gini', u'entropy'])
        space['clf__max_features'] = hp.uniform('clf__max_features',0 , 1)
        space['clf__min_samples_split'] = hp.choice('clf__min_samples_split', range(2,21))
        space['clf__min_samples_leaf'] = hp.choice('clf__min_samples_leaf', range(1,21))
        return space
    
    def space7707(self,points):
        space = {}
        # between 1.0509652110524482e-05 0.09706102908291375
        space['classifier__tol'] = hp.lognormal('classifier__tol',-7, 1)
        # between 3.122280314190532e-05 7.998532268538166
        space['classifier__gamma'] = hp.lognormal('classifier__gamma',0.0001, 1.3)
        # One of True or False
        space['classifier__C'] = hp.lognormal('classifier__C',2.5, 3)
        # choice
        space['imputation__strategy'] = hp.choice('imputation__strategy', [u'mean', u'median', u'most_frequent'])
        # choice
        space['classifier__degree'] = hp.choice('classifier__degree', [1, 2, 3, 4, 5])
        # Between -0.9942534412466477 0.9975887639931769
        space['classifier__coef0'] = hp.uniform('classifier__coef0', -1, 1)
        # True or False
        space['classifier__shrinking'] = hp.choice('classifier__shrinking',[True, False])
        return space
    
        