import pandas as pd
import numpy as np
import os 

# Data visualization
import matplotlib.pyplot as plt

def pre_process_df(df):
  """
  Remove rows without ratings. 
  Remove feature columns that have nulls after agregating information per artist.
  Reset user ids to [0-n_unique_users].
  """
  df = df.loc[:, ~df.columns.str.contains('Unnamed')]

  # Prepare dataset with non-null users, artists and ratings entries.
  not_null_df = df[df['LIKE_ARTIST'].notnull()]
  print("Dataset Without Null Ratings Size: {:,}\n".format(len(not_null_df)))

  # Reset users ids to 0-n_users
  idx_label = np.sort(np.array(not_null_df.User.unique()))
  #  Use np.argwhere to covert value to its sorted index.
  not_null_df.loc[:,'User'] = not_null_df.apply(lambda x: np.argwhere(idx_label == x.User)[0][0], 
                          axis=1)

  # Drop features columns that have nulls after aggregating per artist.
  drop = ['HEARD_OF','OWN_ARTIST_MUSIC']
  artiststing = df.groupby('Artist').mean()

  for column in artiststing.columns:
    if column != 'Artist' and (artiststing[column].isnull().sum() > 0) :
      drop.append(column)

  df = df.drop(columns=drop)

  return df

def plot_histogram(distribution,bins_num, xlabel, ylabel, title):
  print('{} avg. value: {:.2f}'.format(xlabel, np.mean(distribution)))
  plt.hist(distribution, bins=bins_num)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()

def plot_xygraph(yvals, title, xlabel, ylabel):
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.plot(yvals)
  plt.show()

def plot_training(loss, accuracy, num_epochs):
  plt.title("Training Evaluaiton per Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Metrics")

  plt.gca().set_prop_cycle(color=['blue', 'green'])
  # Set x axis
  plt.xticks(range(num_epochs), range(1,num_epochs+1))
  plt.plot(loss)
  plt.plot(accuracy)

  plt.legend(['val_loss', 'val_accuracy'], loc='lower left')
  plt.show()

def plot_results(content, content_embed, collaborative, user, top):
  plt.title("Top 5 recommendation for user {}".format(user))
  plt.xlabel("Rank")
  plt.ylabel("model")

  plt.gca().set_prop_cycle(color=['blue', 'green','orange'])
  # Set x axis
  plt.xticks(range(top), range(1,top+1))
  plt.plot(content)
  plt.plot(content_embed)
  plt.plot(collaborative)

  plt.legend(['features content', 'embedding content', 'collaborative'], loc='upper left')
  plt.show()
