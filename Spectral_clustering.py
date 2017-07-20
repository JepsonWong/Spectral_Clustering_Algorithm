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
                if(self.map[row][col] == 0):
                    self.path[row][col] = 0
                else:
                    self.path[row][col] = row

        for k in range(self.nodenum):
            for i in range(self.nodenum):
                for j in range(self.nodenum):
                    if(self.W[i][k] != 0 and self.W[k][j] != 0):
                        if((self.W[i][k] + self.W[k][j] < self.W[i][j]) or (self.W[i][j] == 0 and i!=j)):
                            self.W[i][j] = self.W[i][k] + self.W[k][j]
                            self.path[i][k] = i
                            self.path[i][j] = self.path[k][j]

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

# 谱聚类算法
def Spectral_clustering():

    maps = [
        [0, 3, 5, 8, 0],
        [3, 0, 8, 4, 11],
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
    i = printPath(path, W, 0, 3) #另一种方法计算从0到3的相似度
    print i
    M = Similarity_matrix(W)
    print M

if __name__ == '__main__':
    #DoTest()
    Spectral_clustering()

