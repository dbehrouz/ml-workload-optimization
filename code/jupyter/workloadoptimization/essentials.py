import networkx as nx
import numpy as np


class Component:
    def __init__(self, name, fullName, parameters):
        self.name = name
        self.params = parameters
        self.fullName = fullName

    def equals(self, other):
        if self.name != other.name:
            return False
        if self.fullName != other.fullName:
            return False
        if len(self.params) != len(other.params):
            return False
        for key in self.params.keys():
            if self.params[key] != other.params[key]:
                return False
        return True

    def toString(self):
        return '%s , %s' % (self.name, self.params)

    estimatedRunTime = 0
    name = ""
    fullName = ""
    params = dict()


class ExperimentObject:
    def __init__(self, run, flow, task, quality):
        self.run = run
        self.flow = flow
        self.task = task
        self.quality = quality
        self.components = []

    def toString(self):
        return '{}, {}, {}, {}, {}'.format(self.run, self.flow, self.task, self.quality,
                                           [comp.name for comp in self.components])

    def extractParams(self):
        params = {}
        for c in self.components:
            preKey = c.name
            # print c.params
            for k, v in c.params.iteritems():
                params['{}__{}'.format(preKey, k)] = v
        return params

    def equals(self, other):
        if self.flow != other.flow:
            return False
        if self.task != other.task:
            return False
        if len(self.components) != len(other.components):
            return False
        for i in range(len(self.components)):
            if not self.components[i].equals(other.components[i]):
                return False
        return True

    def defaultRunTime(self):
        return self.estimatedRunTime

    def stepSize(self):
        return len(self.components)

    def asTrialDoc(self, space, index):
        from hyperopt.pyll.base import Apply
        PARAMS = self.extractParams()
        misc = {}
        idxs = {}
        vals = {}
        for k in space.keys():
            idxs[k] = [index]
            v = space[k]
            # for switch hyperparameter types, their index should be used instead of their actual value
            if v.name == 'switch':
                # TODO: some pipeliens are missing some of the parameters or the values dont match, this
                # should be investiaged further
                # default incase the variable is missing from some experiment_notebooks
                vals[k] = [0]
                for i in range(len(v.pos_args)):
                    if type(v.pos_args[i]) is not Apply:
                        if v.pos_args[i].obj == PARAMS[k]:
                            vals[k] = [i - 1]
            else:
                if isinstance(PARAMS[k], float):
                    vals[k] = [PARAMS[k]]
                else:
                    raise Exception(
                        'This should be solved: {} should be float for run {} since the hyperopt space requires a float'.format(
                            PARAMS[k], self.run))

        misc['cmd'] = ('domain_attachment', 'FMinIter_Domain')
        misc['tid'] = index
        misc['idxs'] = idxs
        misc['vals'] = vals
        misc['workdir'] = None
        return {'tid': index, 'spec': None, 'result': {'loss': 1 - self.quality, 'status': 'ok'}, 'misc': misc}

    estimatedRunTime = 0
    run = 0
    flow = 0
    task = 0
    # TODO: later on change to an actual evaluation object 
    quality = 0
    components = []


class ExperimentGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    # construct the directed graph (for now we only support acyclic graphs)
    graph = None

    # Update the graph by add the experiment object
    def insertExperimentObject(self, experimentObject):
        import uuid
        if not self.graph.has_node(experimentObject.task):
            self.graph.add_node(experimentObject.task, label=experimentObject.task)
        # setting the starting point to root
        v = experimentObject.task
        i = 0
        for c in experimentObject.components:
            i = i + 1
            exists = False
            for e in self.graph.out_edges(v):
                if c.equals(self.graph[e[0]][e[1]]['transformation']):
                    self.graph[e[0]][e[1]]['weight'] += 1
                    self.graph[e[0]][e[1]]['time'] += c.estimatedRunTime
                    v = e[1]
                    exists = True
                    break
            if not exists:
                dataset_id = str(uuid.uuid4())
                self.graph.add_node(dataset_id, label='')
                isModelNode = True if i == len(experimentObject.components) else False
                self.graph.add_edge(v, dataset_id, name=c.name, weight=1, time=c.estimatedRunTime, isModel=isModelNode,
                                    transformation=c)
                v = dataset_id
            prev = c

    # For an experiment object, find the common prefix size and component names
    def findPrefix(self, experimentObject):
        if not self.graph.has_node(experimentObject.task):
            return '', 0, 0
        v = experimentObject.task
        savedTime = 0
        prefix = []
        for c in experimentObject.components:
            for e in self.graph.out_edges(v):
                if c.equals(self.graph[e[0]][e[1]]['transformation']):
                    prefix.append(c.name)
                    savedTime = savedTime + c.estimatedRunTime
                    v = e[1]
                    break
        return ' ==> '.join(prefix), len(prefix), savedTime

    def optimizeAndRun(self, experimentObject):
        prefix, size, savedTime = self.findPrefix(experimentObject)
        stepsWithOptimization = experimentObject.stepSize() - size
        # print experimentObject.defaultRunTime(), totalTime
        timeWithOptimization = experimentObject.defaultRunTime() - savedTime
        self.insertExperimentObject(experimentObject)

        return prefix, stepsWithOptimization, timeWithOptimization

    def prettyLabel(self, edgeLabel):
        if (edgeLabel['weight'] <= 1):
            return ''
        else:
            return '{} , ({})'.format(edgeLabel['name'], edgeLabel['weight'])


class ExperimentParser:
    # Internal methods for parsing
    def parseValue(self, value):
        import ast
        try:
            if (value == 'null'):
                return None
            if (value == 'true'):
                return True
            if (value == 'false'):
                return False
            actual = ast.literal_eval(value)
            if type(actual) is list:
                return sorted(actual)
            if type(actual) is str:
                return actual.replace('"', '')

            return actual
        except:
            return value

    # Internal methods
    def getFullyQualifiedName(self, o):
        return o.__module__ + "." + o.__class__.__name__

    def fromSKLearnPipeline(self, runId, flowId, taskId, quality, setup, pipeline):
        assert flowId == setup.flow_id
        experimentObject = ExperimentObject(runId, flowId, taskId, quality)
        for componentKey, componentValue in pipeline.steps:
            prefix = componentKey
            fullName = self.getFullyQualifiedName(componentValue)
            componentParams = dict()
            for paramKey, paramValue in setup.parameters.items():
                if paramValue.full_name.startswith(fullName):
                    # Openml saves the type informatino in a weird way so we have to write a special piece of code
                    if (paramValue.parameter_name == 'dtype'):
                        componentParams[str(paramValue.parameter_name)] = np.float64
                        # typeValue = self.parseValue(paramValue.value)['value']
                        # if (typeValue == 'np.float64'):
                        #    componentParams[str(paramValue.parameter_name)] = np.float64
                        # else:
                        #    componentParams[str(paramValue.parameter_name)] = typeValue
                    elif (paramValue.parameter_name == 'random_state'):
                        componentParams[str(paramValue.parameter_name)] = 14766
                    else:
                        componentParams[str(paramValue.parameter_name)] = self.parseValue(paramValue.value)
            comp = Component(prefix, fullName, componentParams)
            experimentObject.components.append(comp)
        return experimentObject

    def fromOpenMLFlow(self, runId, flowId, taskId, quality, setup, pipeline):
        assert flowId == setup.flow_id
        experimentObject = ExperimentObject(runId, flowId, taskId, quality)
        for componentKey, componentValue in pipeline.components.items():
            prefix = componentKey
            fullName = componentValue.class_name
            componentParams = dict()
            for paramKey, paramValue in setup.parameters.items():
                if paramValue.full_name.startswith(fullName):
                    # Openml saves the type informatino in a weird way so we have to write a special piece of code
                    if (paramValue.parameter_name == 'dtype'):
                        componentParams[str(paramValue.parameter_name)] = np.float64
                        # typeValue = self.parseValue(paramValue.value)['value']
                        # if (typeValue == 'np.float64'):
                        #    componentParams[str(paramValue.parameter_name)] = np.float64
                        # else:
                        #    componentParams[str(paramValue.parameter_name)] = typeValue
                    elif (paramValue.parameter_name == 'random_state'):
                        componentParams[str(paramValue.parameter_name)] = 14766
                    else:
                        componentParams[str(paramValue.parameter_name)] = self.parseValue(paramValue.value)
            comp = Component(prefix, fullName, componentParams)
            experimentObject.components.append(comp)
        return experimentObject

    def extractSKLearnPipelines(self, experiments, pipelines):
        experimentObjects = []
        for index, row in experiments.iterrows():
            runId, flowId, taskId, accuracy, setup = row.run_id, row.flow_id, row.task_id, row.accuracy, row.setup
            pipeline = pipelines[flowId]
            experimentObjects.append(self.fromSKLearnPipeline(runId, flowId, taskId, accuracy, setup, pipeline))
        return experimentObjects

    def extractOpenMLFlows(self, experiments, pipelines):
        experimentObjects = []
        for index, row in experiments.iterrows():
            runId, flowId, taskId, accuracy, setup = row.run_id, row.flow_id, row.task_id, row.accuracy, row.setup
            pipeline = pipelines[flowId]
            experimentObjects.append(self.fromOpenMLFlow(runId, flowId, taskId, accuracy, setup, pipeline))
        return experimentObjects
