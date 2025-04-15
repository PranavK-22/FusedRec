from flask import Flask, jsonify, request
from recommender import HybridRecommender  # Assuming this contains your recommendation logic
import torch
import numpy as np

app = Flask(__name__)

# Initialize the recommender system
recommender = HybridRecommender()

# Route for movie recommendation
@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the movie title from the query parameters
    movie_title = request.args.get('movie', type=str)

    if not movie_title:
        return jsonify({"error": "No movie title provided!"}), 400

    try:
        # Get recommendations from the HybridRecommender system
        recommendations = recommender.recommend(movie_title, top_k=5)

        # Format the response
        recommended_movies = [
            {
                "title": movie['title'],
                "genres": movie['genres'],
                "score": movie['score']
            }
            for movie in recommendations
        ]

        return jsonify({
            "movie_title": movie_title,
            "recommendations": recommended_movies
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)