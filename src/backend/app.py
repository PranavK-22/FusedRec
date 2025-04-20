from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import (
    hybrid_recommender,
    genre_based_recommender,
    year_based_recommender,
    genome_tag_based_recommender
)

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_title = data.get('title', '').strip()
    model_type = data.get('model', '').strip()

    if not movie_title or not model_type:
        return jsonify({'error': 'Missing title or model type'}), 400

    movie_title = movie_title.lower()

    try:
        if model_type == 'hybrid':
            recommendations = hybrid_recommender(movie_title)
            return jsonify({'recommendations': recommendations})

        elif model_type == 'content_genre':
            recommendations = genre_based_recommender(movie_title)
            return jsonify({'recommendations': recommendations})

        elif model_type == 'year_based':
            recommendations = year_based_recommender(movie_title)
            return jsonify({'recommendations': recommendations})

        elif model_type == 'genome_tag_based':
            result = genome_tag_based_recommender(movie_title)
            return jsonify({
                'recommendations': result['recommendations'],
                'top_tags': result['top_tags']
            })

        else:
            return jsonify({'error': 'Invalid model type selected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)