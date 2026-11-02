# Movie Recommendation System

## Overview

A Python-based movie recommendation system that uses natural language processing and machine learning to recommend movies based on a user's description of the type of film they want to watch.

The project explores different recommendation approaches, including semantic similarity, movie reranking and hybrid ranking.

## Features

- Collects movie information from the TMDB API.
- Stores movie data locally in a JSON file to avoid repeated API requests.
- Filters movies by selected genres.
- Supports release-year filtering using specific years and year ranges.
- Accepts natural language descriptions of desired movie preferences.
- Generates movie recommendations using E5 semantic similarity.
- Reranks movies using the BGE Reranker model.
- Combines E5 and BGE rankings using Reciprocal Rank Fusion (RRF).

## Technologies Used

- Python
- NumPy
- Requests
- Sentence Transformers
- TMDB API

## Models and Techniques

- **E5 (intfloat/e5-base-v2):** Generates text embeddings and uses cosine similarity to find movies that match the user's description.
- **BGE Reranker (BAAI/bge-reranker-base):** Scores the relevance of movie descriptions against the user's prompt.
- **Reciprocal Rank Fusion (RRF):** Combines the rankings from E5 and BGE to produce a hybrid recommendation list.

## How It Works

1. **Data Collection:** Retrieves movie information from TMDB, including titles, genres, plot summaries, release dates, cast and production details.
2. **Data Storage:** Saves the collected information in `movies.json` so it can be reused without repeatedly accessing the API.
3. **Data Processing:** Converts movie information into text descriptions for the recommendation models.
4. **Filtering:** Narrows down the movie collection using selected genres and optional release-year conditions.
5. **Semantic Similarity:** Uses E5 embeddings and cosine similarity to compare movie descriptions with the user's prompt.
6. **Reranking:** Uses BGE to score and rank eligible movies according to their relevance to the prompt.
7. **Hybrid Ranking:** Combines both rankings using Reciprocal Rank Fusion to generate a final list of recommendations.

## Installation

Install the required packages:

```bash
pip install sentence-transformers requests scikit-learn numpy
```

## Running the Project

1. Clone the repository.
2. Ensure `movies.json` is in the project directory.
3. Install the required dependencies.
4. Run `movie_algorithm.py`.
5. Select the desired movie genres.
6. Enter your movie description in the `user_prompt` variable.
7. Optionally include a release-year condition in the prompt.

## Limitations

- Runs as a Python script and does not currently have a graphical user interface.
- Recommendations depend on the quality and availability of movie descriptions.
- Model inference requires downloading or accessing the pretrained models.
- The current system displays movie titles and ranking scores rather than full movie recommendations with posters and additional details.

## Data Source

Movie metadata is collected using the [TMDB API](https://developer.themoviedb.org/docs).

This project uses TMDB data but is not endorsed or certified by TMDB.