import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
train = pd.read_csv("train.csv")
train.rename(columns={'track_name': 'name', 'track_genre': 'genres'}, inplace=True)
train.drop_duplicates(subset=['name', 'artists'], keep='first', inplace=True)

# Preprocess data
features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = train[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

genres_encoded = pd.get_dummies(train['genres'], prefix='genre')
X_with_genres = pd.concat([X, genres_encoded], axis=1)
X_scaled_with_genres = scaler.fit_transform(X_with_genres)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
train['cluster'] = kmeans.fit_predict(X_scaled_with_genres)

# Recommendation function
def recommend_from_cluster(song_name, artist_name, data):
    # Find the song with the specific name and artist
    song_row = data[(data['name'] == song_name) & (data['artists'] == artist_name)]
    
    if song_row.empty:
        return pd.DataFrame({"Error": ["Song not found in the dataset."]})
    
    song_cluster = song_row['cluster'].values[0]
    song_genre = song_row['genres'].values[0]
    
    # Get songs from the same cluster
    same_cluster_songs = data[data['cluster'] == song_cluster]
    
    # Filter songs with the same genre
    recommendations = same_cluster_songs[same_cluster_songs['genres'] == song_genre]
    recommendations = recommendations[same_cluster_songs['name'] != song_name].head(5)
    
    return recommendations[['name', 'artists', 'genres']]

# Streamlit interface
st.title("Music Recommender System")
st.write("Get song recommendations based on your input!")

# Inputs
song_name = st.text_input("Enter the song name:")
artist_name = st.text_input("Enter the artist name:")

if st.button("Get Recommendations"):
    if not song_name or not artist_name:
        st.error("Please enter both song name and artist name.")
    else:
        recommendations = recommend_from_cluster(song_name, artist_name, train)
        if "Error" in recommendations.columns:
            st.error(recommendations["Error"].iloc[0])
        else:
            st.write("### Recommendations:")
            st.table(recommendations.reset_index(drop=True))