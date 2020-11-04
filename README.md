# competition3395

Les codes présentés dans ce répertoire furent utilisés lors de la compétition kaggle : https://www.kaggle.com/c/ift3395-6390-arxiv

Le code bayesV3.py est une implémentation de l'algorithme de bayes naif multinomial. Cette algorithme permit d'obtenir un score de classification de 0.79777 sur Kaggle
Pour lancer cce code, simplement l'exécuter. Il créera alors un ficchier solution.csv contenant l'ensemble de test classifiés.

Le code tensorflow_code.py contient une implémentation d'un pseudo réseau de neurone avec Tensorflow. En fait, ce code représente un classifieur linéaire simple permettant d'obtenir un score de 0.78600


Le code tensorflow_codeV2.py contient une impplémentation d'un vrai réseau de neurone. Toutefois, celui-ci utilise les résultats préalablement classés par l'algorithme de bayes naif, ce qui ne lui permet pas d'obtenir un bien meilleur score sur kaggle : 0.80222 

Edit : Le résultat de 80% fut obtenu par chance après l'écriture du rapport. Tous les autres dépôts sur kaggle effectué avec cet algorithme ne permirent pas de dépasser 80%, tel qu'indiqué dans le rapport.
