import pandas as pd

def rankings(dataframe, classifiers):

  #ordenando dataframe por ranking
  dataframe = dataframe.sort_values(by='rank_test_score')
  dataframe.reset_index(drop=True, inplace=True)

  #colocando as pontuações por ordem dos classificadores que recebemos
  scores = list()
  for classifier in classifiers:
    query = dataframe[dataframe.param_classifier.eq(classifier)].iloc[0,:]
    res = query[['mean_train_score', 'mean_test_score']].values
    scores.append(res)
  
  ranking = pd.DataFrame(columns=['model', 'mean_train_score', 'mean_test_score'])
  for i in range(len(classifiers)):
    ranking.loc[i,:] = [classifiers[i], scores[i][0], scores[i][1]]
  
  mean_train = ranking.sort_values(by='mean_train_score', ascending=False)
  mean_test = ranking.sort_values(by='mean_test_score', ascending=False)
  
  return mean_train, mean_test
