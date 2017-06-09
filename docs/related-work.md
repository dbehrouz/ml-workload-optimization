# Related Work
In this document, we list the related work and a short summary of the interesting and important contributions of each work.

## SQLShare [1]
SQLShare is a collaborative database, allowing users to perform SQL like queries on a database.
Users work are shared and visible to others.
The requirement of such system are as follows:
- Weakly structured data: datasets exhibit inconsistent types, missing and incorrect values, inconsistencies across column and table names, horizontal and vertical decomposition of logical datasets into sets of files
- Derived datasets: most of the datasets go through the same set of preprocessing steps. As a result, a considerable amount of time are spent on recomputations.
- Collaborative sharing: public datasets, scripts and results
- Complex manipulation: scripts are often complex, therefore full support for SQL is required
- Diverse users: users have different backgrounds, therefore the system should be easy to use for people without much prior experience in setting up database systems.
- Low data lifetime: contrary to traditional database systems, here the data is transient and short term. It is stored, analyzed and put aside. Therefore, it is difficult to amortize the cost of schema design and data loading


The main metrics they extracted from their query logs are:
- Query length
- Query runtime
- Number and type of physical and logical operators
- Number and types of expression operators
- Table and columns referenced
- Operator costs






## References
[1] Jain, Shrainik, et al. "Sqlshare: Results from a multi-year sql-as-a-service experiment." Proceedings of the 2016 International Conference on Management of Data. ACM, 2016.s
