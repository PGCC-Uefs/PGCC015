# ==================================================================
# Universidade Estadual de Feira de Santana
# Mestrado em Ciência da Computação
# Disciplina: PGCC015 - Inteligência Computacional
# Professor: Matheus Giovanni Pires
# Aluno: Noberto Pires Maciel
# EPC04 - 23/10/2020
# ==================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
 
def closest_node(data, t, map, m_rows, m_cols, raio):
  result = (0,0)
  for i in range(m_rows):
    for j in range(m_cols):
      ed = euc_dist(map[i][j], data[t])
      if ed < raio:
        raio = ed
        result = (i, j)
  return result
 
def euc_dist(v1, v2):
  return np.linalg.norm(v1 - v2)
 
def most_common(lst, n):
  if len(lst) == 0: return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)
 
def main(Rows,Cols,LearnMax,StepsMax,normalize, k_means, raio):
  RangeMax = Rows + Cols
  np.random.seed(1)
  Dim = 4
 
  # 1. carregar dados
  data_file = "https://raw.githubusercontent.com/PGCC-Uefs/PGCC015/master/EPC04/datasets/iris.txt"
  data_x = np.loadtxt(data_file, delimiter=",", usecols=range(0,4), dtype=np.float64)
  data_y = np.loadtxt(data_file, delimiter=",", usecols=[4], dtype=np.int)

  data_file_test = "https://raw.githubusercontent.com/PGCC-Uefs/PGCC015/master/EPC04/datasets/iris_tst2.txt"
  data_x_test = np.loadtxt(data_file_test, delimiter=",", usecols=range(0,4), dtype=np.float64)
  scalertest = MinMaxScaler().fit(data_x_test);
  data_x_test = scalertest.transform(data_x_test);

  # 1.1. normalização
  if normalize:
    scaler = MinMaxScaler().fit(data_x);
    data_x = scaler.transform(data_x);
 
  # 2. construção do SOM
  print("Mapa Auto-organizável %s x %s"%(Rows,Cols)+" Matriz U:")
  map = np.random.random_sample(size=(Rows,Cols,Dim))
  for s in range(StepsMax):
    #if s % (StepsMax/5) == 0: print("step: ", str(s))
    restante = 1.0 - ((s * 1.0) / StepsMax)
    curr_range = (int)(restante * RangeMax)
    curr_rate = restante * LearnMax
 
    t = np.random.randint(len(data_x))
    (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols, raio)
    for i in range(Rows):
      for j in range(Cols):
        if euc_dist(bmu_row, bmu_col) < curr_range:
          map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])
  
  # 3. construção da Matriz U
  u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
  for i in range(Rows):
    for j in range(Cols):
      v = map[i][j]  # a vector 
      sum_dists = 0.0; ct = 0     
      if i-1 >= 0:    # above
        sum_dists += euc_dist(v, map[i-1][j]); ct += 1
      if i+1 <= Rows-1:   # below
        sum_dists += euc_dist(v, map[i+1][j]); ct += 1
      if j-1 >= 0:   # left
        sum_dists += euc_dist(v, map[i][j-1]); ct += 1
      if j+1 <= Cols-1:   # right
        sum_dists += euc_dist(v, map[i][j+1]); ct += 1
      
      u_matrix[i][j] = sum_dists / ct

  # configura o tamanho do gráfico
  plt.rcParams['figure.figsize'] = (5,5)

  # exibir Matriz U
  print(u_matrix)
  plt.imshow(u_matrix, cmap='gray', interpolation='gaussian')  # black = close = clusters
  plt.title('Matriz U %s x %s'%(Rows,Cols))
  plt.xlabel('vetor X')
  plt.ylabel('vetor Y')
  plt.show()
 
  # associa as classes aos node-maps
  mapping = np.empty(shape=(Rows,Cols), dtype=object)
  for i in range(Rows):
    for j in range(Cols):
      mapping[i][j] = []
 
  for t in range(len(data_x)):
    (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols, raio)
    mapping[m_row][m_col].append(data_y[t])
  

  classes = np.zeros(shape=(Rows,Cols), dtype=np.int)  
  for i in range(Rows):
    for j in range(Cols):
      classes[i][j] = most_common(mapping[i][j], 3)

  # roda o algoritmo k-means
  if k_means == True:
    kmeans = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300)
    kmeans.fit(classes)
    kmeans.cluster_centers_
    kmeans.predict(classes)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'red',label = 'Centroids', alpha=0.5,)
    plt.scatter(classes[:,0], classes[:,1], s = 40, c = kmeans.labels_)
    plt.scatter(data_x_test[:,0], data_x_test[:,1], s = 20, c ='b')
    plt.scatter(data_x_test[:,2], data_x_test[:,3], s = 20, c ='g')
    plt.title('Iris Clusters and Centroids')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.show()

    plt.imshow (classes, cmap = plt.cm.get_cmap ('terrain_r', 4))
    plt.colorbar ()
    plt.show()
# ==================================================================
if __name__=="__main__":
  #main(Rows,Cols,Dim,LearnMax,StepsMax,normalize,k_means,raio)
  main(4,4,0.001,10000,False,True,1)
  main(6,6,0.001,10000,False,False,1)
  main(8,8,0.001,10000,False,False,1)
