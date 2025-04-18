from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import HybridRecommender

app = Flask(__name__)
CORS(app)  # Enables CORS for all domains on all routes

# Initialize the recommender
recommender = HybridRecommender()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get('title', '')

    if not title:
        return jsonify({'error': 'No title provided'}), 400

    recommendations = recommender.recommend(title)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True, port=5001)