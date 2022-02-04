# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 00:33:22 2022

@author: user
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
# Load the model

def course_recommend(course_titles, top_n):

    idx = indices[course_titles]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #print(len(sim_scores))
    sim_scores = sim_scores[1:40]
    course_indices = [i[0] for i in sim_scores]
    df_recommendation = course_title.iloc[course_indices].head(top_n)
    return df_recommendation.tolist()

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    print(data)
    course_title = data['course_title']    
    result_pred = course_recommend(course_title, 3)
    # Make prediction using model loaded from disk as per the data.
    
    response = jsonify( rec_1 = result_pred[0], rec_2 = result_pred[1], rec_3 = result_pred[2] )
    print(response)
    return response
if __name__ == '__main__':
    
    df = pd.read_csv('demo_dataset.csv')
    from sklearn.feature_extraction.text import TfidfVectorizer

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    matrix = tf.fit_transform(df['course_tags'])

    from sklearn.metrics.pairwise import linear_kernel
    cosine_similarities = linear_kernel(matrix, matrix)

    course_title = df['course_title']
    indices = pd.Series(df.index, index=df['course_title'])
    app.run(port=5000, debug=True)