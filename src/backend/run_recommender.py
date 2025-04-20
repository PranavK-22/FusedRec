from recommender import hybrid_recommender

# Ask user for input
movie_input = input("Enter a movie title (e.g., The Matrix (1999)): ").strip()

# Get recommendations
results = hybrid_recommender(movie_input, top_n=5)

# Display results
if results:
    print("\nRecommended Movies:")
    for title in results:
        print("-", title)
else:
    print("\nNo recommendations found. Please check the movie title or try a different one.")