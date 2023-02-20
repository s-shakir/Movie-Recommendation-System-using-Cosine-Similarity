import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import pandas as pd
import random


def load_data():
  rating_df = pd.read_csv("./ml-25m/ratings.csv")
  movies_df = pd.read_csv("./ml-25m/movies.csv")
  Tags = pd.read_csv("./ml-25m/tags.csv")

  Mean = rating_df.groupby(by="userId",as_index=False)['rating'].mean()
  rate_avg = pd.merge(rating_df,Mean,on='userId')
  rate_avg['norm_rate']=rate_avg['rating_x']-rate_avg['rating_y']

  #rate_avg = rate_avg.head(2000)

  rate_avg.to_csv(r'/content/gdrive/My Drive/Rating_avg.csv')

  return rating_df, movies_df, Tags, Mean, rate_avg

def create_pivot():
  chunker = pd.read_csv('Rating_avg.csv', chunksize=3)
  piv=pd.DataFrame()
  for piece in chunker:
    piv=piv.add(piece.pivot('userId', 'movieId', 'norm_rate'), fill_value=0)

  return piv

def create_check(rate_avg):
  chunker = pd.read_csv('Rating_avg.csv', chunksize=3)
  piv=pd.DataFrame()
  for piece in chunker:
    piv=piv.add(piece.pivot('userId', 'movieId', 'rating_x'), fill_value=0)
  return piv

def movie_user_pivot(pivot):
  # Replacing NaN by Movie Average
  movie_pivot = pivot.fillna(pivot.mean(axis=0))

  # Replacing NaN by user Average
  user_pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)

  return movie_pivot,user_pivot


def user_c_s(user_pivot):
  # user similarity on replacing NAN by user avg
  cs = cosine_similarity(user_pivot)
  np.fill_diagonal(cs, 0 )
  similar_user = pd.DataFrame(cs,index=user_pivot.index)
  similar_user.columns=user_pivot.index

  return similar_user

def mov_c_s(movie_pivot, user_pivot):
  # user similarity on replacing NAN by item(movie) avg
  cs2 = cosine_similarity(movie_pivot)
  np.fill_diagonal(cs2, 0)
  similar_movie = pd.DataFrame(cs2,index=movie_pivot.index)
  similar_movie.columns=user_pivot.index
  #similar_movie.head()

  return similar_movie

def nearest_neighbour(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index,
    index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)

    return df

def usm(u1, u2, rate_avg, movies_df):
    mov_like = rate_avg[rate_avg.userId == u1].merge( rate_avg[rate_avg.userId == u2], on = "movieId", how = "inner" )
    result = mov_like.merge(movies_df, on = 'movieId' )
    return result


def u_rated(u, pivot, top_10_mov, mov_u, movie_pivot, Mean, similar_movie, movies_df):
    seen = pivot.columns[pivot[pivot.index==u].notna().any()].tolist()
    a = top_10_mov[top_10_mov.index==u].values
    b = a.squeeze().tolist()
    d = mov_u[mov_u.index.isin(b)]
    l = ','.join(d.values)
    seen_movie = l.split(',')
    m_unseen = list(set(seen_movie)-set(list(map(str, seen))))
    m_unseen = list(map(int, m_unseen))
    score = []
    for item in m_unseen:
        c = movie_pivot.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = Mean.loc[Mean['userId'] == u,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = similar_movie.loc[u,index]
        final = pd.concat([f, corr], axis=1)
        final.columns = ['adg_score','correlation']
        final['score']=final.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        n = final['score'].sum()
        d = final['correlation'].sum()
        fs = avg_user + (n/d)
        score.append(fs)
    data = pd.DataFrame({'movieId':m_unseen,'score':score})
    t5recom = data.sort_values(by='score',ascending=False).head(5)
    mname = t5recom.merge(movies_df, how='inner', on='movieId')
    mname = mname.title.values.tolist()
    return mname


def main():

  rating_df, movies_df, Tags, Mean, rate_avg = load_data()
  pivot = create_pivot()
  check = create_check(rate_avg)
  movie_pivot,user_pivot = movie_user_pivot(pivot)
  similar_user = user_c_s(user_pivot)
  similar_movie = mov_c_s(movie_pivot, user_pivot)
  ran_val = np.random.randint(5,30,size=len(rate_avg))
  top_10_users = nearest_neighbour(similar_user,10)
  top_10_mov = nearest_neighbour(similar_movie,10)
  get_df = usm(ran_val, ran_val, rate_avg, movies_df)
  get2_df = get_df.loc[ : , ['rating_x_x','rating_x_y','title']]
  rate_avg = rate_avg.astype({"movieId": str})
  mov_u = rate_avg.groupby(by = 'userId')['movieId'].apply(lambda x:','.join(x))
  user = int(input("Enter the user id to whom you want to recommend : "))
  predicted_movies = u_rated(user, pivot, top_10_mov, mov_u, movie_pivot, Mean, similar_movie, movies_df)
  print(" ")
  print("The Recommendations for User Id : ", user)
  print("   ")
  for i in predicted_movies:
    print(i)




if __name__ == '__main__':
  main()
