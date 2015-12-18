'''
Code adapted from https://github.com/soorajmr/moviemeta/blob/master/notebooks/doc2vec.ipynb

Doc2Vec on plot descriptions from Wikipedia.

The pre-processed wikipedia corpus from the above project is copied to the 'data' directory.
This if fed into doc2vec model.

'''

import pandas as pd
import numpy as np
import os
from gensim import models
import multiprocessing
from gensim import models
import pickle
import re

'''
Tagged docs for Doc2Vec
'''		
def create_tagged_doc_list(plots):
    tagged_docs = []
    for i, plot in enumerate(plots):
        plot_words = []
        for tagged_sent in plot:
            plot_words += tagged_sent.words

        tag = plot[0].tags
        tagged_docs.append(models.doc2vec.TaggedDocument(words = plot_words, tags = tag))

    return tagged_docs

wiki_meta_df = pd.read_csv('data/wikipedia/wiki_meta_df.csv')
with open('data/wikipedia/wiki_plots_d2v.pickle', 'rb') as f:
    wiki_plots = np.load(f)

plots_tagged_with_titles = create_tagged_doc_list(wiki_plots) 

# Filter out those with very small plot summaries
plots_tagged_with_titles = [p for p in plots_tagged_with_titles if len(p.words) > 100]

cores = multiprocessing.cpu_count()
d2v_wiki = models.Doc2Vec(dm=1, size=100, window=10, negative=5, min_count=10, workers=cores)
d2v_wiki.build_vocab(plots_tagged_with_titles)
d2v_wiki.train(plots_tagged_with_titles)

# Convert titles to the same format as plot tags
wiki_meta_df.title = wiki_meta_df.apply(lambda row: "%s (%d)" % (row.title, row.year), axis=1)
wiki_titles = list(set([s.tags[0] for s in plots_tagged_with_titles]))
wiki_vectors = [d2v_wiki.docvecs[t] for t in wiki_titles]
#wiki_titles = [unicode(s, 'utf-8') for s in wiki_titles]
#wiki_titles = [t.encode('utf-8') for t in wiki_titles]
vecdf = pd.DataFrame(dict(title=wiki_titles, vector=wiki_vectors))
print("Generated content vectors for %d movies" %len(vecdf))
vecdf.to_csv("data/output/content_vectors.csv")
print(vecdf[:10])
