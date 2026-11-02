from sentence_transformers import SentenceTransformer, util
import requests
from datetime import datetime
import time
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import CrossEncoder



existing_genres = {
0:'Science Fiction', 1:'Action', 2:'Adventure',3:'Fantasy',4: 'Mystery',5: 'Horror', 
6:'Comedy', 7:'Thriller', 8:'Drama', 9:'Family', 10:'Romance',   11: 'Documentary'
,12: 'Animation',13: 'Crime', 14:'History', 15:'Music',16: 'War',17: 'Western', 18:'TV Movie'
}
movie_collection={}
movie_text_descrip= []

if os.path.exists("movies.json"):

  with open("movies.json", "r") as file:
    movie_collection =json.load(file)
else:
  for page in range(1,41):

    url = f"https://api.themoviedb.org/3/discover/movie?page={page}"
    movie_data_response = requests.get( url, headers=headers)
    movie_data = movie_data_response.json()



    for raw_movie in movie_data['results']:
      temp_list_of_actors=[]
      temp_list_of_directors = []
      temp_list_of_producers = []
      temp_list_of_genres=[]
      
      credit_url = f"https://api.themoviedb.org/3/movie/{raw_movie['id']}/credits?"
      credit_response = requests.get(credit_url, headers = headers)
      creditData = credit_response.json()

      details_url = f"https://api.themoviedb.org/3/movie/{raw_movie['id']}"
      details_response = requests.get(details_url, headers=headers)
      detailsData = details_response.json()

      for genre in detailsData['genres']:
        temp_list_of_genres.append(genre['name'])
        
      
      i= 0
      try:
          
          while(i<10):
            temp_list_of_actors.append(creditData['cast'][i]["name"])

            i += 1
    
      except IndexError:
            pass
      for crew_member in creditData['crew']:
                  if "Producer" in crew_member['job'] :
                    temp_list_of_producers.append(crew_member['name'])
    
                  if crew_member['job'] == 'Director' :
                    temp_list_of_directors.append(crew_member['name'])

      movie_collection[raw_movie['id']] = { 
            
            'title' : raw_movie['title'],
            'overview' : raw_movie['overview'],
            'genres' : temp_list_of_genres,
            'spoken_languages' : detailsData['spoken_languages'],
            'release_date' : raw_movie['release_date'],
            'origin_country' : detailsData['origin_country'],
            'production_countries' : detailsData['production_countries'],
            'poster_path':  raw_movie['poster_path'],
            'backdrop_path' : raw_movie['backdrop_path'],
            'original_language' : raw_movie['original_language'],
            'vote_average':raw_movie['vote_average'],
            'runtime': detailsData['runtime'],
            'actors': temp_list_of_actors,
            'producers':  temp_list_of_producers,
            'directors': temp_list_of_directors,
            }
  with open("movies.json", "w") as file:
    json.dump(movie_collection,file)

"""for movie in movie_collection:
    genres = ", ".join(movie_collection[movie]['genres'])
    producers = ", ".join(movie_collection[movie]['producers'])
    directors = ", ".join(movie_collection[movie]['directors'])
    actors = ", ".join(movie_collection[movie]['actors'])
    production_country = "Unknown" if not movie_collection[movie]['production_countries'] else movie_collection[movie]['production_countries'][0]['name']
    movie_time = None if  movie_collection[movie]['runtime']=="None" else int(movie_collection[movie]['runtime'])
    raw_date_str = movie_collection[movie]['release_date']

    if raw_date_str:
        date_obj = datetime.strptime(raw_date_str, "%Y-%m-%d")
        date = date_obj.strftime("%B %d, %Y")
    else:
        date = "Unknown"

    if movie_time is not None and movie_time > 60:
        hours, minutes = divmod(movie_time, 60)
        movie_time = f'{hours} hr {minutes} min'
    elif movie_time is not None:
        movie_time = f'{movie_time} min'
    else:
        movie_time = "Unknown"
        
    text = (
        f"{movie_collection[movie]['title']} "
        f"({date}) is a movie in the genres of {genres}. "
        f"Plot: {movie_collection[movie]['overview']} "
        f"Cast: {actors}. "
        f"Director: {directors}. "
        f"Producers: {producers}. "
        f"Runtime: {movie_time}. "
        f"Production country: {production_country}."
    )
    # text = f"{movie_collection[movie]['title']} is a {genres} movie starring {actors} which follows {movie_collection[movie]['overview']} with a run time of {time}. It's produced by {producers} and directed by {directors}. The movie was released in {date} and made in {production_country}"

   
    movie_text_descrip.append(text)
"""    

for movie in movie_collection:
    genres = ", ".join(movie_collection[movie]["genres"])
    title = movie_collection[movie]["title"]
    overview = movie_collection[movie]["overview"]

    text = (
        f"Movie: {title}. "
        f"Genres: {genres}. "
        f"Story and plot: {overview}"
    )

    movie_text_descrip.append(text)
    
sentence_Model = SentenceTransformer('all-MiniLM-L6-v2')


start = time.time()

sentence_embedding = sentence_Model.encode(
    movie_text_descrip,
    convert_to_tensor=True,
    show_progress_bar=True
).cpu().numpy()

print("Encoding completed:", round(time.time() - start, 2), "seconds")

user_selected_genres = []
filtered_by_genre_movie_collection = {}
def user_genre_selection():
  user_selected_genres.clear()
  #these will be options on the final javascript built page, no need to catch an exception
  ask_user = input("0: Science Fiction, 1: Action, 2: Adventure, 3: Fantasy, 4: Mystery, 5: Horror, 6: Comedy, 7: Thriller, 8: Drama, 9: Family, 10: Romance, 11: Documentary, 12: Animation, 13: Crime, 14: History, 15: Music, 16: War, 17: Western, 18: TV Movie: ").split(",")

  for index_no in ask_user:
    index_no = int(index_no.strip())
    user_selected_genres.append(existing_genres[index_no])
  return user_selected_genres

def filter_movies_by_genre():
  filtered_by_genre_movie_collection.clear()
  for movie_id,movie in movie_collection.items():
     if any(genre in user_selected_genres for genre in movie["genres"]):
      filtered_by_genre_movie_collection[movie_id] = movie
  return filtered_by_genre_movie_collection

user_genre_selection()
filter_movies_by_genre()

user_prompt = "I want an emotional film about family relationships, personal struggles, and second chances that feels moving and meaningful without being completely depressing."
user_prompt_encode = sentence_Model.encode(user_prompt, convert_to_tensor=True).cpu().numpy().reshape(1,-1)


movie_ids = list(movie_collection.keys())

filtered_indices = []

for index, movie_id in enumerate(movie_ids):
    if movie_id in filtered_by_genre_movie_collection:
        filtered_indices.append(index)

filtered_embeddings = sentence_embedding[filtered_indices]

cosine_sim = cosine_similarity(
    filtered_embeddings, user_prompt_encode
)

scores = np.argsort(cosine_sim.flatten())[::-1]

filtered_movie_ids = [
    movie_ids[i] for i in filtered_indices
]


# cosine_sim = cosine_similarity(sentence_embedding,user_prompt_encode)
# movie_ids = list(movie_collection.keys())
# scores = np.argsort(cosine_sim.flatten())[::-1]


words = user_prompt.split()
user_years =[]
year_condition = None

for word in words:
    # Before and after data
    if word.startswith("before:"):
        year_condition = "before"
        user_years.append(int(word.split(":")[1]))

    elif word.startswith("after:"):
        year_condition = "after"
        user_years.append(int(word.split(":")[1]))

    # Specific year
    elif word.startswith("year:"):
        year_condition = "year"
        user_years.append(int(word.split(":")[1]))

    # Between data
    elif word.startswith("between:"):
        year_condition = "between"
        years_between = word.split(":")[1].split("-")

        for year in years_between:
            if year.isdigit() and len(year) == 4:
                user_years.append(int(year))


def filter_movies_by_year(movie_collection, year_condition, user_years):
    filtered_movies = []

    for movie_id, movie_data in movie_collection.items():
        if not movie_data["release_date"]:
            continue

        movie_year = datetime.strptime(
            movie_data["release_date"], "%Y-%m-%d"
        ).year

        if len(user_years) == 1:
            if year_condition == "before" and movie_year < user_years[0]:
                filtered_movies.append(movie_id)

            elif year_condition == "after" and movie_year > user_years[0]:
                filtered_movies.append(movie_id)

            elif year_condition == "year" and movie_year == user_years[0]:
                filtered_movies.append(movie_id)

        elif len(user_years) == 2 and year_condition == "between":
            min_year = min(user_years)
            max_year = max(user_years)

            if min_year <= movie_year <= max_year:
                filtered_movies.append(movie_id)

    return filtered_movies
      

if year_condition is not None:
    filtered_movies = filter_movies_by_year(
        filtered_by_genre_movie_collection, year_condition, user_years
    )
else:
    filtered_movies = None

# Save MiniLM's top 10 before reranking
minilm_top_10 = []

for i in scores:
    movie_id = filtered_movie_ids[i]

    if filtered_movies is not None and movie_id not in filtered_movies:
        continue

    minilm_top_10.append(movie_id)

    if len(minilm_top_10) == 10:
        break
"""count = 0

for i in scores:
    movie_id = filtered_movie_ids[i]

    if filtered_movies is not None and movie_id not in filtered_movies:
        
        continue
    

    print(
        movie_collection[movie_id]["title"],
        movie_collection[movie_id]["release_date"],
        movie_collection[movie_id]["genres"]
    )
  

    count += 1

    if count == 10:
        break0,4,7
  """  
    #reverted fully


cross_enc_model = CrossEncoder("BAAI/bge-reranker-base")
cross_enc_model.activation_fn=None

# Match each movie ID to its description
movie_description_by_id = dict(zip(movie_ids, movie_text_descrip))



# Stage 1: Retrieve the top 50 eligible movies
candidate_movie_ids = []

for i in scores:
    movie_id = filtered_movie_ids[i]

    if filtered_movies is not None and movie_id not in filtered_movies:
        continue

    candidate_movie_ids.append(movie_id)

    if len(candidate_movie_ids) == 50:
        break


# Stage 2: Score each candidate with the cross-encoder
cross_enc_pairs = []

for movie_id in candidate_movie_ids:
    movie_description = movie_description_by_id[movie_id]
    cross_enc_pairs.append((user_prompt, movie_description))

cross_enc_scores = cross_enc_model.predict(cross_enc_pairs)
# print("Minimum score:", cross_enc_scores.min())
# print("Maximum score:", cross_enc_scores.max())


# Print all 50 candidates and their scores
# for movie_id, score in zip(candidate_movie_ids, cross_enc_scores):
#     print(
#         movie_collection[movie_id]["title"],
#         float(score)
#     )

# Rank candidates by cross-encoder score
ranked_indices = np.argsort(cross_enc_scores)[::-1]


# Display the top 10 recommendations
for i in ranked_indices[:10]:
    movie_id = candidate_movie_ids[i]

    # print(
    #     movie_collection[movie_id]["title"],
    #     movie_collection[movie_id]["release_date"],
    #     movie_collection[movie_id]["genres"],
    #     round(float(cross_enc_scores[i]), 3)
    # )

# E5 model
e5_model = SentenceTransformer("intfloat/e5-base-v2")

# Encode movie descriptions
e5_movie_descriptions = [
    "passage: " + description
    for description in movie_text_descrip
]

e5_embedding = e5_model.encode(
    e5_movie_descriptions,
    convert_to_tensor=True,
    show_progress_bar=True
)

# Encode the user's prompt
e5_prompt = e5_model.encode(
    "query: " + user_prompt,
    convert_to_tensor=True
)

# Calculate cosine similarity
e5_cosine_sim = util.cos_sim(
    e5_prompt,
    e5_embedding
).cpu().numpy().flatten()

# Rank the movies
e5_scores = np.argsort(e5_cosine_sim)[::-1]

# Display results
# print("\nMiniLM's original top 10:")

# for movie_id in minilm_top_10:
#     print(movie_collection[movie_id]["title"])
print(f'user prompt: {user_prompt}')

print("\nBGE's reranked top 10:")

for i in ranked_indices[:10]:
    movie_id = candidate_movie_ids[i]
    print(movie_collection[movie_id]["title"])

print("\nE5's top 10:")

e5_count = 0

for i in e5_scores:
    movie_id = movie_ids[i]

    if movie_id not in filtered_by_genre_movie_collection:
        continue

    if filtered_movies is not None and movie_id not in filtered_movies:
        continue

    print(movie_collection[movie_id]["title"])
    e5_count += 1

    if e5_count == 10:
        break
   









       

  


