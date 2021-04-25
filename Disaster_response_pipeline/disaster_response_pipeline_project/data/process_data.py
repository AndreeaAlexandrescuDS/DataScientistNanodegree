### Scope : data cleaning pipeline that:
# -    Loads the messages and categories datasets
# -    Merges the two datasets
# -    Cleans the data
# -    Stores it in a SQLite database


# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
      Function that reads two datasets from specified filepath,
    merges them and returns only the merged result.
      Arguments:
      --messages_filepath: path to the messages set (ex. 'disaster_messages.csv')
      --categories_filepath: path to the categories set(ex. 'disaster_categories.csv') 
    """       
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how="inner", on ="id")
    return df

def clean_data(df):
    """
      Function that processes the dataset returned from the load_data() function,
    in order to get it ready for the next modelling part: make 36 separate targets 
    from a single concateneted list of targets.
      Arguments: 
      -- df: dataframe to be processed
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', n=-1, expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
      Save data into a SQLite database
      Arguments:
      --df: data to be saved
      --database_filename: name of the SQL data base(ex. 'DBAndreeaA.db')
    """
    # create SQLite engine
    engine = create_engine('sqlite:///' + database_filename)
    # save df to the SQLite engine
    df.to_sql('input_disaster', engine, index=False)
    print('input_disaster table saved to ' + 'sqlite:///' + database_filename)


def main():
    """
      Main function that takes 4 arguments and calls the functions defined above,
    with the provided arguments.
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()