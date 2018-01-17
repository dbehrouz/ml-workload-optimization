# Goal
Optimizing Machine learning workloads by removing redundancies, caching, and several other optimization techniques.

# Problem
- (From ModelDB paper: The current style of model building is ad-hoc and there is no practical solution for a data scientist to manage model that are built over time.

# Solution
- Device an automatic materialization strategies of datasets after various stages of machine learning pipeline. Two metrics that can guide the materialization strategy are the frequency of the execution of a certain transformation on a dataset, and the final quality of the model created from the specific pipeline.