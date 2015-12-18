# Match movie titles in Movielens and Wikipedia data
# Code adapted from https://github.com/soorajmr/moviemeta/blob/master/notebooks/link_movies.ipynb
#

import re
import pandas as pd
from fuzzywuzzy import process

def extract_year(title):
    match = re.search(r'.*\(([0-9]{4})\)', title)
    if(match is None):
        year = 0
    else:
        year = match.group(1)
    return int(year)

movies = pd.read_csv("data/ml-20m/movies.csv")
movies['year'] = movies.title.apply(extract_year)
ml_titlemap = dict(zip(movies.movieId, movies.title))
ml_yearmap = dict(zip(movies.movieId, movies.year))
#movies.year.plot(kind='hist', bins=range(1900, 2020, 5)).set_title("Movies per year")

pref = pd.read_csv("data/output/pref_vectors.csv")
content = pd.read_csv("data/output/content_vectors.csv")
content['year'] = content.title.apply(lambda x: x.decode('utf-8')).apply(extract_year)
pref['year'] = pref.apply(lambda x: ml_yearmap[x.movieId], axis=1)
pref['title'] = pref.apply(lambda x: ml_titlemap[x.movieId].decode('utf-8'), axis=1)

def fuzzy_find_matching_index(string, string_list):
    fuzzy_match, similarity_pct = process.extractOne(string, string_list)

    # Boost the similarity of long strings
    if(similarity_pct >= 90):
        similarity_pct += len(string) / 4
        
    if(similarity_pct >= 95): # Take only the ones with high similarity
        #print(string, "\t\t", fuzzy_match, "\t\t", similarity_pct)
        return string_list.index(fuzzy_match)
    else:
        return None


def create_title_map(wiki, mv):
    wiki.title = wiki.title.apply(lambda t: t.decode('utf-8'))
    titlemap = mv.title.apply(lambda s: fuzzy_find_matching_index(s, list(wiki.title)))
    titlemap = titlemap.dropna()
    titlemap = titlemap.apply(lambda i: wiki.index[i])
    return titlemap.astype(int).to_dict()    

mv2wiki = {}
for y in range(2000, 2015):
    mv2wiki.update(create_title_map(content[content.year == y], pref[pref.year == y]))
    print("processed year %d, total %d" % (y, len(mv2wiki)))


import pickle

with open('data/output/mv2wiki.pickle', 'wb') as f:
    pickle.dump(mv2wiki, f)


