import pandas as pd
from fuzzywuzzy import fuzz
import nltk

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

df = pd.read_csv('dataset/world-universities.csv')

def clean_query(query):
    query = query.lower()
    query = nltk.word_tokenize(query) 
    stop_words = set(stopwords.words('english')) 
    query = [w for w in query if not w in stop_words] 
    tagged_query = nltk.pos_tag(query)
    tagged_query = [item[0] for item in tagged_query if item[1] not in ['VBD', 'POS','CD']]


    final_cleaned_sentence = ' '.join(tagged_query)
    return final_cleaned_sentence

def return_best_match(df, query):
    query = clean_query(query)
    
    def fuzzy_matcher(university_name):
        return fuzz.token_set_ratio(university_name.lower(), query)
    
    df['percent_match'] = df['Name'].apply(fuzzy_matcher)
    return df.sort_values(by='percent_match', ascending=False).head(10)
    