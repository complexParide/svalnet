# The SValNet module

This Python module statistically validates networks in bipartite complex systems, implementing the algorithm described in the paper:

*Tumminello, M., Miccichè, S., Lillo, F., Piilo, J., & Mantegna, R. N. (2011). Statistically Validated Networks in Bipartite Complex Systems. In E. Ben-Jacob (Ed.), PLoS ONE (Vol. 6, Issue 3, p. e17994). Public Library of Science (PLoS). https://doi.org/10.1371/journal.pone.0017994*

This unsupervised method statistically validates each link of a projected network against a null hypothesis that takes into account system heterogeneity. It can detect network structures that are very informative about the organization and specialization of the investigated systems and identify those relationships between elements of the projected network that cannot be explained simply by system heterogeneity. The method can be used to detect **over-expressions** (nodes of class A have more common neighbors of class B than the null-hypothesis expects) or **under-expression** (nodes of class A have less common neighbors of class B than the null-hypothesis expects).


This module has been primarily used to infer comorbidity networks from a bipartite network of patients and disease nodes. Unfortunately, such medical data cannot be released publically. We thus demonstrate how to use the module with a bipartite network of Marvel comics and Marvel characters to infer a validated network of superheroes. The data has been preprocessed from *http://bioinfo.uib.es/~joemiro/marvel.html* (*arXiv:cond-mat/0202174*)
