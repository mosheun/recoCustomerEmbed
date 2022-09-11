import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def getStats(file, users):
    ratings = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values
    merged = pd.merge(ratings, users, how='inner', on='user')
    merged = merged[['user', 'rate']]
    mean_df = merged.groupby("user").describe()
    mean_df = mean_df.reset_index()
    return mean_df

def mergeUsers():
    folder = 'C:\\Users\\moshu\\Documents\\NYU\\360Degree\\Amazon\\'
    file = str(folder) + 'users_art.csv'
    users = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values

    file = str(folder) + 'users_fashion.csv'
    usersF = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values

    merged = pd.merge(users, usersF, how='inner', on='user')

    file = str(folder) + 'users_beauty.csv'
    usersB = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values

    merged = pd.merge(merged, usersB, how='inner', on='user')

    file = str(folder) + '17.7.similarUsers.csv'
    merged.to_csv(file, encoding='utf-8', index=False)

    print("done")
    return



def selectUsers360():
    folder = 'C:\\Users\\moshu\\Documents\\NYU\\360Degree\\Amazon\\'
    file = str(folder) + 'blend_users.csv'
    users = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values
    #similarUsers = ["664170", "794777", "432516", "603969", "280909", "876328", "450094", "209217", "1081465"]

    file = str(folder) + 'amazon_bert_embedding_all.csv'
    ratings = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values
    merged = pd.merge(ratings, users, how='inner', on='user')
    file = str(folder) + 'amazon_bert_embedding_blend.csv'
    merged.to_csv(file, encoding='utf-8', index=False)


    #file = str(folder) + 'amazon_artcrafts_filter.csv'
    #songs = getStats(file, users)
    #file = str(folder) + 'amazon_artcrafts_filter_stats.csv'
    #songs.to_csv(file, encoding='utf-8', index=False)

    #file = str(folder) + 'amazon_beauty_filter.csv'
    #games = getStats(file, users)
    #file = str(folder) + 'amazon_beauty_filter_stats.csv'
    #games.to_csv(file, encoding='utf-8', index=False)

    #file = str(folder) + 'amazon_fashion_filter.csv'
    #movies = getStats(file, users)
    #file = str(folder) + 'amazon_fashion_filter_stats.csv'
    #movies.to_csv(file, encoding='utf-8', index=False)

    #file = str(folder) + 'rates_map_filter_books.csv'
    #books = getStats(file, users)
    #file = str(folder) + 'rates_map_filter_books_stats.csv'
    #books.to_csv(file, encoding='utf-8', index=False)



    return

def getStatistics(folder):
    file = str(folder) + 'blend_users.csv'
    users = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values

    file = str(folder) + 'amazon_artcrafts_filter.csv'
    songs = getStats(file, users)
    file = str(folder) + 'amazon_artcrafts_filter_blend_stats.csv'
    songs.to_csv(file, encoding='utf-8', index=False)

    file = str(folder) + 'amazon_beauty_filter.csv'
    games = getStats(file, users)
    file = str(folder) + 'amazon_beauty_filter_blend_stats.csv'
    games.to_csv(file, encoding='utf-8', index=False)

    file = str(folder) + 'amazon_fashion_filter.csv'
    movies = getStats(file, users)
    file = str(folder) + 'amazon_fashion_filter_blend_stats.csv'
    movies.to_csv(file, encoding='utf-8', index=False)

    return songs, games, movies

    # file = str(folder) + 'rates_map_filter_books.csv'
    # books = getStats(file, users)
    # file = str(folder) + 'rates_map_filter_books_stats.csv'
    # books.to_csv(file, encoding='utf-8', index=False)

def plt360():
    folder = 'C:\\Users\\moshu\\Documents\\NYU\\360Degree\\Amazon\\'
    file = str(folder) + 'embeddings_blend.csv'
    x = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    ax = sns.heatmap(x, cmap=cmap, linewidths=.5, square=False)
    ax.set(ylabel='Users', xlabel='Composite Dimensions')
    plt.show()
    return

def calculateSimilarity():
    folder = 'C:\\Users\\moshu\\Documents\\NYU\\360Degree\\Amazon\\'
    file = str(folder) + 'Embeddings.csv'
    x = pd.read_csv(file, low_memory=False, na_values='?')  # switch '?' with na values
    m1 = euclidean_distances(x, x)
    file = str(folder) + 'Embeddings_sim_euc.csv'
    pd.DataFrame(m1).to_csv(file, encoding='utf-8', index=False)


if __name__ == "__main__":
    #selectUsers360()
    #plt360()
    #getStatistics()
    calculateSimilarity()
    #mergeUsers()