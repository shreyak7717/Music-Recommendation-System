import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from annoy import AnnoyIndex
from sklearn.metrics import mean_squared_error

# Load and preprocess data for Annoy-based recommendation
@st.cache_data
def load_and_preprocess_data_for_annoy():
    data = pd.read_csv("train.csv")
    data_unique = data.drop_duplicates(subset=['track_name', 'artists'], keep='first').reset_index(drop=True)
    
    genre_encoder = OneHotEncoder()
    genre_encoded = genre_encoder.fit_transform(data_unique[['track_genre']]).toarray()
    genre_encoded_df = pd.DataFrame(genre_encoded, columns=genre_encoder.get_feature_names_out(['track_genre']))
    
    features = ['popularity', 'danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_unique[features])
    
    combined_data = pd.concat([pd.DataFrame(data_scaled, columns=features), genre_encoded_df], axis=1)
    
    return data_unique, combined_data

# Build Annoy index
@st.cache_resource
def build_annoy_index(combined_data):
    f = combined_data.shape[1]
    annoy_index = AnnoyIndex(f, 'angular')
    for i in range(combined_data.shape[0]):
        annoy_index.add_item(i, combined_data.iloc[i])
    annoy_index.build(10)
    return annoy_index

# Load and preprocess data for K-means clustering
def load_and_preprocess_data_for_kmeans():
    train = pd.read_csv("train.csv")
    train.rename(columns={'track_name': 'name', 'track_genre': 'genres'}, inplace=True)
    train.drop_duplicates(subset=['name', 'artists'], keep='first', inplace=True)
    
    features = ['danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    X = train[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    genres_encoded = pd.get_dummies(train['genres'], prefix='genre')
    X_with_genres = pd.concat([X, genres_encoded], axis=1)
    X_scaled_with_genres = scaler.fit_transform(X_with_genres)

    kmeans = KMeans(n_clusters=3, random_state=42)
    train['cluster'] = kmeans.fit_predict(X_scaled_with_genres)
    
    return train

# K-means recommendation function
def recommend_from_cluster(song_name, artist_name, data):
    song_row = data[(data['name'] == song_name) & (data['artists'] == artist_name)]
    
    if song_row.empty:
        return pd.DataFrame({"Error": ["Song not found in the dataset."]})
    
    song_cluster = song_row['cluster'].values[0]
    song_genre = song_row['genres'].values[0]
    
    same_cluster_songs = data[data['cluster'] == song_cluster]
    recommendations = same_cluster_songs[same_cluster_songs['genres'] == song_genre]
    recommendations = recommendations[same_cluster_songs['name'] != song_name].head(5)
    
    return recommendations[['name', 'artists', 'genres']]

# Annoy-based dropdown
def song_get_dropdown(data):
    return sorted(list({f"{track} - {artist}" for track, artist in zip(data['track_name'], data['artists'])}))

def find_song_index(song_option, data):
    parts = song_option.rsplit(" - ", 1)
    if len(parts) != 2:
        return None
    track_name, artist = parts
    matches = data[(data['track_name'] == track_name) & (data['artists'] == artist)]
    return matches.index[0] if not matches.empty else None

def recommend_similar_songs(song_index, annoy_index, data):
    indices = annoy_index.get_nns_by_item(song_index, 4)[1:4]
    return data[['track_name', 'artists']].iloc[indices]

# Combined app
def main():
    st.title('Music Recommender System')
    st.write("Enter a song to get recommendations from both models:")

    # Load data for both models
    kmeans_data = load_and_preprocess_data_for_kmeans()
    annoy_data, combined_data = load_and_preprocess_data_for_annoy()
    annoy_index = build_annoy_index(combined_data)
    
    # User inputs
    song_name = st.text_input("Enter the song name:")
    artist_name = st.text_input("Enter the artist name:")

    if st.button("Get Recommendations"):
        if not song_name or not artist_name:
            st.error("Please enter both song name and artist name.")
        else:
            # K-means recommendations
            kmeans_recommendations = recommend_from_cluster(song_name, artist_name, kmeans_data)

            # Annoy recommendations
            dropdown_song = f"{song_name} - {artist_name}"
            song_index = find_song_index(dropdown_song, annoy_data)
            if song_index is not None:
                annoy_recommendations = recommend_similar_songs(song_index, annoy_index, annoy_data)
            else:
                annoy_recommendations = pd.DataFrame({"Error": ["Song not found in the dataset."]})

            # Display recommendations side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("K-means Recommendations")
                if "Error" in kmeans_recommendations.columns:
                    st.error(kmeans_recommendations["Error"].iloc[0])
                else:
                    st.table(kmeans_recommendations.reset_index(drop=True))

            with col2:
                st.subheader("Annoy Recommendations")
                if "Error" in annoy_recommendations.columns:
                    st.error(annoy_recommendations["Error"].iloc[0])
                else:
                    st.write("Recommended Songs:")
                    for idx, row in annoy_recommendations.iterrows():
                        st.write(f"• {row['track_name']} by {row['artists']}")

if __name__ == '__main__':
    main()
