from adjustText import adjust_text
import matplotlib.pyplot as plt
from pyrdf2vec.graphs import KG
import pandas as pd
import rdflib
import numpy as np
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from sklearn.manifold import TSNE


# ------------------------------------------------------------------------------
# Read a CSV file containing the entities we want to classify.
data = pd.read_csv(
    "/home/nasredine/dev/research/rdf2vec/entities.tsv", sep="\t")
entities = [entity for entity in data["location"]]
print(entities)


# Define our knowledge graph (here: DBPedia SPARQL endpoint).
knowledge_graph = KG(
    "https://dbpedia.org/sparql",
    skip_predicates={"www.w3.org/1999/02/22-rdf-syntax-ns#type"},
    literals=[
        [
            "http://dbpedia.org/ontology/wikiPageWikiLink",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
        ],
        ["http://dbpedia.org/ontology/humanDevelopmentIndex"],
    ],
)
# Create our transformer, setting the embedding & walking strategy.
transformer = RDF2VecTransformer(
    Word2Vec(epochs=10),
    walkers=[RandomWalker(10, 20, with_reverse=False, n_jobs=2)],
    # verbose=1
)
# Get our embeddings.
embeddings, literals = transformer.fit_transform(knowledge_graph, entities)
print(embeddings)

walk_tsne = TSNE(random_state=42)
X_tsne = walk_tsne.fit_transform(embeddings)

plt.figure(figsize=(15, 15))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

texts = []
for x, y, lab in zip(X_tsne[:, 0], X_tsne[:, 1], entities):
    lab = lab.split('/')[-1]
    text = plt.text(x, y, lab)
    texts.append(text)

adjust_text(texts, lim=5, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()
