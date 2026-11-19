from recsys_lab.data import load_ratings
from sklearn.model_selection import train_test_split
from recsys_lab.config import TEST_SIZE,RANDOM_STATE,VALID_SIZE

ratings = load_ratings()

def splitting(ratings):
    """Splitting our data into train , valid, test"""
    outer_train_df,test_df = train_test_split(ratings,test_size=TEST_SIZE,random_state=RANDOM_STATE)
    fitting_df,validation_df = train_test_split(outer_train_df,test_size = VALID_SIZE ,random_state=RANDOM_STATE)
    return outer_train_df,test_df,fitting_df,validation_df

def filter(outer_train_df,test_df,fitting_df,validation_df):
    known_users_fit = set(fitting_df['user_id'].unique())
    known_movies_fit = set(fitting_df['movie_id'].unique())

    validation_filtered_df = validation_df[
                validation_df['user_id'].isin(known_users_fit)&
                validation_df['movie_id'].isin(known_movies_fit)
    ].copy()


    known_users_outer = set(outer_train_df['user_id'].unique())
    known_movies_outer = set(outer_train_df['movie_id'].unique())

    test_filtered_df = test_df[
                test_df['user_id'].isin(known_users_outer)&
                test_df['movie_id'].isin(known_movies_outer)
    ].copy()

    return test_filtered_df,validation_filtered_df

def matrix(fitting_df):
    fitting_matrix=fitting_df.pivot(index='user_id',columns='movie_id',value='rating')
    return fitting_matrix