# !/usr/bin/python
# coding:utf-8

from sklearn.cluster import KMeans
import numpy as np
import random
import matplotlib.pyplot as plt

# 图的类定义
class Graph:
    def __init__(self, maps, edgenum=0):
        self.map = maps  # 图的矩阵结构
        self.nodenum = len(maps)
        self.edgenum = edgenum
        self.W = np.zeros((len(maps), len(maps)))#maps
        self.path = np.zeros((len(maps), len(maps)))
        self.W_Bellman_Ford = np.zeros((len(maps), len(maps)))
        self.path_Bellman_Ford = np.zeros((len(maps), len(maps)))
        self.edge = []
        e = []
        for i in range(self.nodenum):
            for j in range(self.nodenum):
                if(self.map[i][j] != 0):
                    e.append(i)
                    e.append(j)
                    e.append(self.map[i][j])
                    e1 = e[:] # 之前没有这一步，list赋值为引用赋值
                    self.edge.append(e1)
                e[:] = []
        #for m in self.edge:
        #    print "11111"
        #    print m
        #    print "22222"

    def isOutRange(self, x):
        try:
            if x >= self.nodenum or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    # 得到图的顶点数目
    def GetNodenum(self):
        self.nodenum = len(self.map)
        return self.nodenum

    # 得到图的边的数目
    def GetEdgenum(self):
        self.edgenum = 0
        for i in range(self.nodenum):
            for j in range(self.nodenum):
                if self.map[i][j] != 0:
                    self.edgenum = self.edgenum + 1
        return self.edgenum

    # 添加顶点
    def InsertNode(self):
        for i in range(self.nodenum):
            self.map[i].append(0)
            self.W[i].append(0)
            self.path.append(0)
        self.nodenum = self.nodenum + 1
        ls = [0] * self.nodenum
        self.map.append(ls)
        self.W.append(ls)
        self.path.append(ls)

    # 假删除，只是归零而已
    def DeleteNode(self, x):
        for i in range(self.nodenum):
            if self.map[i][x] != 0:
                self.map[i][x] = 0
                self.edgenum = self.edgenum - 1
            if self.map[x][i] != 0:
                self.map[x][i] = 0
                self.edgenum = self.edgenum - 1

    # 添加边
    def AddEdge(self, x, y):
        if self.map[x][y] != 1:
            self.map[x][y] = 1
            self.edgenum = self.edgenum + 1

    # 移除边
    def RemoveEdge(self, x, y):
        if self.map[x][y] != 0:
            self.map[x][y] = 0
            self.edgenum = self.edgenum - 1

    # 图的广度优先遍历
    def BreadthFirstSearch(self):
        def BFS(self, i):
            print(i)
            visited[i] = 1
            for k in range(self.nodenum):
                if self.map[i][k] == 1 and visited[k] == 0:
                    BFS(self, k)
        visited = [0] * self.nodenum
        for i in range(self.nodenum):
            if visited[i] is 0:
                BFS(self, i)

    # 图的深度优先遍历
    def DepthFirstSearch(self):
        def DFS(self, i, queue):
            queue.append(i)
            print(i)
            visited[i] = 1
            if len(queue) != 0:
                w = queue.pop()
                for k in range(self.nodenum):
                    if self.map[w][k] is 1 and visited[k] is 0:
                        DFS(self, k, queue)
        visited = [0] * self.nodenum
        queue = [] # 模拟栈的操作
        for i in range(self.nodenum):
            if visited[i] is 0:
                DFS(self, i, queue)

    # 图的相似矩阵 KNN, 未采用
    def K_Nearest_Neighbor_Similarity_Matrix_Computr(self, k):
        remain = [0] * k
        for row in range(self.nodenum):
            for i in range(k):
                remain[i] = row + i +1
            min_index = remain[0]
            for j in range(1, k):
                if(self.map[row][remain[j]] <= self.map[row][remain[0]]):
                    min_index = remain[j]
            col = row + k + 1
            while col%self.nodenum != row:
                if(self.map[row][col] > self.map[row][min_index]):
                    self.map[row][min_index] = 0
                    min_index = col

                    for j in range(k):
                        if(self.map[row][remain[j]] <= self.map[row][min_index]):
                            min_index = remain[j]
                else:
                    self.map[row][col] = 0

        for row in range(self.nodenum):
            for col in range(self.nodenum):
                if(self.map[row][col] != self.map[col][row]):
                    if(self.map[row][col] != 0):
                        self.map[col][row] = self.map[row][col]
                    if(self.map[col][row] != 0):
                        self.map[row][col] = self.map[col][row]
        pass

    # W为图的相似矩阵，此处为(1/最短路径),即路径越短，相似度越高;最短路径的求法为Floyd算法，可以包含负边;path为路径保存。
    def Similarity_matrix(self):
        for row in range(self.nodenum):
            for col in range(self.nodenum):
                self.W[row][col] = self.map[row][col]
                if(self.map[row][col] != 0 and row != col):
                    self.path[row][col] = row
                else:
                    self.path[row][col] = -1 # -1表示i j不通

        for k in range(self.nodenum):
            for i in range(self.nodenum):
                for j in range(self.nodenum):
                    if(self.W[i][k] != 0 and self.W[k][j] != 0):
                        if((self.W[i][k] + self.W[k][j] < self.W[i][j]) or (self.W[i][j] == 0 and i != j)): # 0表示不通或者无穷大
                            self.W[i][j] = self.W[i][k] + self.W[k][j]
                            self.path[i][j] = self.path[k][j]

    # 最短路径算法，Bellman_Ford算法
    # Bellman-Ford算法的结果是一个bool值，表明图中是否存在着从源点s可达的负权回路。
    # 若不存在这样的回路，算法将给出从源点s到图G任意顶点v的最短路径dist[v]；
    # 若存在这样的回路，说明该问题无解，即存在一个从源点s到某一个点的最短路径趋向于负无穷(无限循环可得)
    def Bellman_Ford(self, original):
        for i in range(self.nodenum):
            if i == original:
                self.W_Bellman_Ford[original][i] = 0
            else:
                self.W_Bellman_Ford[original][i] = 1000000 # max
            if self.map[original][i] != 0:
                self.W_Bellman_Ford[original][i] = self.map[original][i]
                self.path_Bellman_Ford[original][i] = original
            else:
                self.path_Bellman_Ford[original][i] = -1 # -1表示不通

        # 如果在某一遍的迭代中，并没有进行松弛操作，说明该遍迭代所有边都没有松弛，可以证明， 至此以后，所有的边都不需要再松弛，因此可以提前结束迭代过程。
        for j in range(self.nodenum):
            flag_end = False
            for e in range(len(self.edge)):
                if self.W_Bellman_Ford[original][(self.edge[e])[1]] > self.W_Bellman_Ford[original][(self.edge[e])[0]] + (self.edge[e])[2]:
                    self.W_Bellman_Ford[original][(self.edge[e])[1]] = self.W_Bellman_Ford[original][(self.edge[e])[0]] + (self.edge[e])[2]
                    self.path_Bellman_Ford[original][(self.edge[e])[1]] = (self.edge[e])[0] # 记录前驱顶点
                    flag_end = True
            if(flag_end == False):
                break

        flag = True # 判断是否含有负权回路,True表示不存在负权值回路
        # 检验负权回路：判断边集E中的每一条边的两个端点是否收敛。如果存在未收敛的顶点，则算法返回false，表明问题无解。
        for e1 in range(len(self.edge)):
            if self.W_Bellman_Ford[original][(self.edge[e1])[1]] > self.W_Bellman_Ford[original][(self.edge[e1])[0]] + (self.edge[e1])[2]:
                flag = False
                break
        return flag

# 欧式距离
def distance(p1,p2):
    return  np.linalg.norm(p1-p2)

# 利用KNN获得相似度矩阵
def getWbyKNN(dis_matrix, k):
    print type(dis_matrix)
    W = np.zeros((len(dis_matrix), len(dis_matrix)))
    for idx,each in enumerate(dis_matrix):
        index_array  = np.argsort(each)
        for i in index_array[1:k+1]:
            W[idx][i] = dis_matrix[idx][i]  # 距离最短的是自己
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2  # 转置相加除以2是为了让矩阵对称
    return W

# 获得度矩阵
def getD(W):
    points_num = len(W)
    D = np.diag(np.zeros(points_num))
    for i in range(points_num):
        D[i][i] = sum(W[i])
    return D

 # 从拉普拉斯矩阵获得特征矩阵
def getEigVec(L,cluster_num):
    eigval,eigvec = np.linalg.eig(L)
    dim = len(eigval)
    dictEigval = dict(zip(eigval,range(0,dim)))
    kEig = np.sort(eigval)[0:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix],eigvec[:,ix]

# 获得中心位置
def getCenters(data,C):
    centers = []
    for i in range(max(C)+1):
        points_list = np.where(C==i)[0].tolist()
        centers.append(np.average(data[points_list],axis=0))
    return centers

def randRGB():
    return (random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0)

def plot(matrix,C,centers,k):
    colors = []
    for i in range(k):
        colors.append(randRGB())
    for idx,value in enumerate(C):
        plt.plot(matrix[idx][0],matrix[idx][1],'o',color=colors[int(C[idx])])
    for i in range(len(centers)):
        plt.plot(centers[i][0],centers[i][1],'rx')
    plt.show()

# 路径乘积/路径个数
def printPath(path, W, fro, to):
    i = 0
    distance = 1
    while(path[fro][to] != fro):
        i = i + 1
        distance = distance * W[path[fro][to]][to]
        to = path[fro][to]
    i = i + 1
    distance = distance * W[fro][to]
    return distance/i

def Similarity_matrix(W):
    M = np.zeros((len(W), len(W)))
    for i in range(len(W)):
        for j in range(len(W)):
            if W[i][j] != 0:
                M[i][j] = 1 / W[i][j]
    return M

# 聚类个数评价函数,模块度函数
# len(k)为聚类个数,W为权重矩阵 k = [[],[]]
# Q_pk表示网络中社团内部边的概率减去网络中同一社团结构下结点间任意连接边的概率的期望值。
# 数值越大，表示图的聚类效果越好。在实际网络中的取值范围常常为0.3~0.7。
def Q(k, W):
    Q_pk = 0
    A_V_V = 0
    A_Vc_Vc = 0
    A_Vc_V = 0
    for i in range(len(W)):
        for j in range(len(W)):
            if W[i][j] != 0:
                A_V_V = W[i][j] + A_V_V
    for i in range(len(k)):
        # k[i]
        for j1 in k[i]:
            for j2 in k[i]:
                if W[j1][j2] != 0:
                    A_Vc_Vc = W[j1][j2] + A_Vc_Vc
            for j3 in range(len(W)):
                if W[j1][j3] != 0:
                    A_Vc_V = W[j1][j3] + A_Vc_V
                if W[j3][j1] != 0:
                    A_Vc_V = W[j3][j1] + A_Vc_V
        A_Vc_V = A_Vc_V - A_Vc_Vc
        Q_pk = Q_pk + ( A_Vc_Vc / A_V_V - ( A_Vc_V / A_V_V ) ^ 2 )
        A_Vc_Vc = 0
        A_Vc_V = 0
    return Q_pk

# 谱聚类算法
def Spectral_clustering():

    maps = [
        [0, 3, 5, 8, 0],
        [3, 0, 6, 4, 11],
        [5, 6, 0, 2, 0],
        [8, 4, 2, 0, 10],
        [0, 11, 0, 10, 0]
    ]
    G = Graph(maps)
    print G.map
    G.Similarity_matrix()
    W1 = G.W
    print W1
    path = G.path
    print path

    M = Similarity_matrix(W1)

    # 用KNN得到W矩阵，用W得到D矩阵
    W = getWbyKNN(M, 2)             # k=2
    D = getD(W)
    print W
    print D

    # 得到拉普拉斯矩阵
    L = D - W
    cluster_num = 3                 # cluster = 3

    # 取出前cluster_num小的特征值对应的特征向量
    eigval, eigvec = getEigVec(L, cluster_num)

    # 获得特征矩阵之后，我们使用kmeans方法来对特征矩阵进行一个聚类，每个特征向量是特征矩阵的列，而每行当成一个聚类样本。这样一聚类就是最终的成果了。
    # 直接使用sklearn中的KMeans函数来调用：
    clf = KMeans(n_clusters=cluster_num)
    s = clf.fit(eigvec)
    C = s.labels_
    print C
    # centers = getCenters(data,C)
    # plot(data,s.labels_,centers,cluster_num)

# 测试图类的函数
def DoTest():
    maps = [
        [0, 3, 5, 8, 0],
        [3, 0, 8, 4, 11],
        [5, 6, 0, 2, 0],
        [8, 4, 2, 0, 10],
        [0, 11, 0, 10 , 0]
        ]
    G = Graph(maps)
    #G.InsertNode()
    #G.AddEdge(1, 4)
    #print("广度优先遍历")
    #G.BreadthFirstSearch()
    #print("深度优先遍历")
    #G.DepthFirstSearch()
    print G.map
    G.Similarity_matrix()
    W = G.W
    path = G.path
    print path
    print W

    for i in range(G.nodenum):
        G.Bellman_Ford(i)
    W1 = G.W_Bellman_Ford
    path1 = G.path_Bellman_Ford
    print W1
    print path1

    i = printPath(path, W, 0, 3) #另一种方法计算从0到3的相似度
    print i
    M = Similarity_matrix(W)
    print M

if __name__ == '__main__':
    DoTest()
    #Spectral_clustering()

