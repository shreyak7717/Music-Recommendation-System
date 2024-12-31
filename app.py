import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from annoy import AnnoyIndex
from sklearn.metrics import mean_squared_error

@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv("D:/saksham/college stuff/Munjal/Third Semester/classroom/PRJ 1/Saksham_230663_MRS/train.csv")
    data_unique = data.drop_duplicates(subset=['track_name', 'artists'], keep='first').reset_index(drop=True)
    
    genre_encoder = OneHotEncoder()
    genre_encoded = genre_encoder.fit_transform(data_unique[['track_genre']]).toarray()
    genre_encoded_df = pd.DataFrame(genre_encoded, columns=genre_encoder.get_feature_names_out(['track_genre']))
    
    features = ['popularity', 'danceability', 'energy', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_unique[features])
    
    combined_data = pd.concat([pd.DataFrame(data_scaled, columns=features), genre_encoded_df], axis=1)
    
    return data_unique, combined_data

@st.cache_resource
def build_annoy_index(combined_data):
    f = combined_data.shape[1]
    annoy_index = AnnoyIndex(f, 'angular')
    for i in range(combined_data.shape[0]):
        annoy_index.add_item(i, combined_data.iloc[i])
    annoy_index.build(10)
    return annoy_index

def song_get_dropdown(data):
    song_dropdown = list({f"{track} - {artist}" for track, artist in zip(data['track_name'], data['artists'])})
    return sorted(song_dropdown)

def find_song_index(song_option, data):
    # Split from the right to separate artist from song name
    parts = song_option.rsplit(" - ", 1)
    
    if len(parts) != 2:
        return None
    
    track_name, artist = parts
    matches = data[(data['track_name'] == track_name) & (data['artists'] == artist)]
    return matches.index[0] if not matches.empty else None

def recommend_similar_songs(song_index, annoy_index, data, reroll=False):
    if not reroll:
        indices = annoy_index.get_nns_by_item(song_index, 4)[1:4]
    else:
        indices = annoy_index.get_nns_by_item(song_index, 16)[5:16:5]
    
    return data[['track_name', 'artists']].iloc[indices]

def calculate_mse(selected_song_index, recommended_song_indices, combined_data):
    selected_song_vector = combined_data.iloc[selected_song_index].values
    
    mse_values = []
    
    for index in recommended_song_indices:
        recommended_song_vector = combined_data.iloc[index].values
        mse = mean_squared_error(selected_song_vector, recommended_song_vector)
        mse_values.append(mse)
    
    return mse_values

def main():
    st.title('Music Recommender System')
    
    data, combined_data = load_and_preprocess_data()
    annoy_index = build_annoy_index(combined_data)
    
    song_dropdown = song_get_dropdown(data)
    selected_song = st.selectbox(
        'Select a song:', 
        song_dropdown, 
        index=None,
        placeholder="Choose a song..."
    )
    
    if selected_song:
        song_index = find_song_index(selected_song, data)
        if song_index is not None:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Selected Song")
                st.write(selected_song)
            
            with col2:
                st.subheader(f"Recommended Songs")
                recommended_songs = recommend_similar_songs(song_index, annoy_index, data)
                
                recommended_song_indices = [find_song_index(f"{row['track_name']} - {row['artists']}", data) for idx, row in recommended_songs.iterrows()]
                
                for idx, row in recommended_songs.iterrows():
                    st.write(f"• {row['track_name']} by {row['artists']}")
                
                mse_values = calculate_mse(song_index, recommended_song_indices, combined_data)
                
                # Log MSE values to the console
                print(f"\n\nMSE Values for Recommendations for the song {selected_song}:")
                for i, mse in enumerate(mse_values):
                    print(f"Recommended Song {i+1} MSE: {mse:.4f}")
                
                if st.button('Reroll Recommendations'):
                    recommended_songs = recommend_similar_songs(song_index, annoy_index, data, reroll=True)
                    
                    recommended_song_indices = [find_song_index(f"{row['track_name']} - {row['artists']}", data) for idx, row in recommended_songs.iterrows()]
                    
                    for idx, row in recommended_songs.iterrows():
                        st.write(f"• {row['track_name']} by {row['artists']}")
                    
                    mse_values = calculate_mse(song_index, recommended_song_indices, combined_data)
                    
                    # Log rerolled MSE values to the console
                    print(f"\n\nMSE Values for Rerolled Recommendations for {selected_song}:")
                    for i, mse in enumerate(mse_values):
                        print(f"Recommended Song {i+1} MSE: {mse:.4f}")
        else:
            st.error("Song not found. Please check the selection.")

if __name__ == '__main__':
    main()