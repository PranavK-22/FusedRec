import React, { useState, useEffect } from 'react';
import './App.css';
import TMDB_API_KEY from './api_key';

const placeholderTitles = [
  'Inception', 'The Matrix', 'Interstellar', 'Titanic', 'Gladiator', 'The Godfather'
];

function App() {
  const [movieTitle, setMovieTitle] = useState('');
  const [modelType, setModelType] = useState('hybrid');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [dynamicPlaceholder, setDynamicPlaceholder] = useState(placeholderTitles[0]);
  const [movieInfo, setMovieInfo] = useState(null);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      setDynamicPlaceholder(placeholderTitles[index]);
      index = (index + 1) % placeholderTitles.length;
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchMovieDetails = async (title) => {
    // Remove year in parentheses from the movie title (e.g., "Inception (2010)" -> "Inception")
    const movieTitleWithoutYear = title.replace(/\s*\(\d{4}\)/, '');

    const googleLink = `https://www.google.com/search?q=${encodeURIComponent(movieTitleWithoutYear)} movie`;

    try {
      const response = await fetch(
        `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_API_KEY}&query=${encodeURIComponent(movieTitleWithoutYear)}`
      );
      const data = await response.json();
      const result = data.results?.[0];

      if (result && (result.poster_path || result.overview)) {
        return {
          title: result.title,
          overview: result.overview || 'No description available.',
          poster: result.poster_path ? `https://image.tmdb.org/t/p/w500${result.poster_path}` : null,
          googleLink,
        };
      }
    } catch (err) {
      console.error("TMDB fetch error:", err);
    }

    return {
      title: movieTitleWithoutYear,
      overview: 'No description available.',
      poster: null,
      googleLink,
    };
  };

  const handleRecommend = async () => {
    if (!movieTitle) return;
    setError('');
    setRecommendations([]);
    setLoading(true);
    setMovieInfo(null);

    try {
      const response = await fetch('http://localhost:5001/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: movieTitle, model: modelType }),
      });

      const data = await response.json();

      if (response.ok) {
        const movieDetails = await fetchMovieDetails(movieTitle);
        setMovieInfo(movieDetails);

        const movieDetailsList = await Promise.all(
          data.recommendations.map(title => fetchMovieDetails(title))
        );

        setRecommendations(movieDetailsList);
      } else {
        setError(data.error || 'Something went wrong!');
      }
    } catch (err) {
      setError('Request failed. Please make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">FusedRec</h1>
      </header>

      <div className="search-section">
        <input
          type="text"
          value={movieTitle}
          onChange={(e) => setMovieTitle(e.target.value)}
          placeholder={`e.g. ${dynamicPlaceholder}`}
          className="search-box"
        />
        <select value={modelType} onChange={(e) => setModelType(e.target.value)} className="dropdown">
          <option value="hybrid">Hybrid (NCF)</option>
          <option value="content_genre">Content-Based (Genre)</option>
          <option value="year_based">Content-Based (Year)</option>
          <option value="genome_tag_based">Content-Based (Genome Tags)</option>
        </select>
        <button className="recommend-btn" onClick={handleRecommend}>
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
        {error && <p className="error">{error}</p>}
      </div>

      {movieInfo && (
        <div className="movie-info">
          <img
            src={movieInfo.poster || '/fallback.jpg'}
            alt={movieInfo.title}
            className="movie-poster"
          />
          <div className="movie-details">
            <h2>{movieInfo.title}</h2>
            <p>{movieInfo.overview || 'Description not available.'}</p>
          </div>
        </div>
      )}

      <div className="recommendation-section">
        {recommendations.map((rec, index) => (
          <div className="movie-card" key={index}>
            <a
              href={rec.googleLink}
              target="_blank"
              rel="noopener noreferrer"
            >
              {rec.poster ? (
                <img
                  src={rec.poster}
                  alt={rec.title}
                  className="movie-card-poster"
                />
              ) : (
                <div className="movie-card-placeholder">No Image</div>
              )}
              <div className="movie-card-info">
                <h3>{rec.title}</h3>
                {rec.overview ? (
                  <p>{rec.overview}</p>
                ) : (
                  <p className="no-description">Description not available</p>
                )}
              </div>
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;