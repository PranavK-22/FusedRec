from recommender import HybridRecommender

# Initialize the recommender
recommender = HybridRecommender()

# Ask user for input
movie_input = input("Enter a movie title (e.g., The Matrix (1999)): ")

# Get recommendations
results = recommender.recommend(movie_input, top_k=5)

# Display results
if results:
    print("\nRecommended Movies:")
    for title in results:
        print("-", title)
else:
    print("\nNo recommendations found. Please check the movie title or try a different one.")