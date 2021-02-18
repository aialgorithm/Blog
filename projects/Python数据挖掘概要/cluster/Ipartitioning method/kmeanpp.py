#kmeans++聚类算法

import random
import numpy as np
from math import sqrt


class KmeanppCluster():
    def __init__(self, k, filename):
        self.k = k
        self.data_dict = {}
        self.data = []
        self.center_dict = {}
        self.new_center = [[0.0, 0.0] for k in range(self.k)]
        # load and standlize
        self.loaddata(filename)
        # random center
        self.center = self.init_center(k)
        #数据格式初始化
        for i in range(k):
            self.center_dict.setdefault(i, {})
            self.center_dict[i].setdefault("center", self.center[i])
            self.center_dict[i].setdefault("items_dist", {})
            self.center_dict[i].setdefault("classify", [])

    def init_center(self, k):
        #kmean初始化随机k个中心点
        #random.seed(1)
        #center = [[self.data[i][r] for i in range(1, len((self.data)))]  
                  #for r in random.sample(range(len(self.data)), k)]
            
        # Kmean ++ 基于距离概率选择k个中心点
        # 1.随机选择一个点
        center = []
        center.append(random.choice(range(len(self.data[0]))))
        # 2.根据距离的概率选择其他中心点
        for i in range(self.k - 1):
            weights = [self.distance_closest(self.data[0][x], center) 
                     for x in range(len(self.data[0])) if x not in center]
            dp = [x for x in range(len(self.data[0])) if x not in center]
            total = sum(weights)
            #基于距离设定权重
            weights = [weight/total for weight in weights]
            num = random.random()
            x = -1
            i = 0
            while i < num :
                x += 1
                i += weights[x]
            center.append(dp[x])
        center = [self.data_dict[self.data[0][center[k]]] for k in range(len(center))]
        print("center",center)
        #center = [[-2.344262295081967, -1.2367724867724865], [-0.2163934426229508, -0.20370370370370366], [1.2172131147540983, 0.6587301587301586]]
        return center
    
    #计算个点与中心的最小距离
    def distance_closest(self, x, center):
        min = 99999
        for centroid in center:
            distance = 0
            cent = self.data_dict[self.data[0][centroid]]
            for i in range(len(cent)):
                distance += abs(self.data_dict[x][i] - cent[i])
            if distance < min:
                min = distance
        return min
    
    def get_median(self, alist):
        """get median of list"""
        tmp = list(alist)
        tmp.sort()
        alen = len(tmp)
        if (alen % 2) == 1:
            return tmp[alen // 2]
        else:
            return (tmp[alen // 2] + tmp[(alen // 2) - 1]) / 2 
    
    #标准化
    def standlize(self, column):
        median = self.get_median(column)
        asd = sum([abs(x - median) for x in column]) / len(column)
        result = [(x - median) / asd for x in column]
        return result
    
    #加载数据并标准化
    def loaddata(self, filename):
        lista = []
        with open(filename , "r") as fileobject:
            lines = fileobject.readlines()
        header = lines[0].split(",")
        self.data = [[] for i in range(len(header))]
        for line in lines[1:]:
            line = line.split(",")
            for i in range(len(header)):
                if i == 0:
                    self.data[i].append(line[i])
                else:
                    self.data[i].append(float(line[i]))                    
        for col in range(1, len(header)):
            self.data[col] = self.standlize(self.data[col])
        #data_dict  data对应Key    
        for i in range(0, len(self.data[0])):
            for col in range(1, len(self.data)):
                self.data_dict.setdefault(self.data[0][i], [])
                self.data_dict[self.data[0][i]].append(self.data[col][i])

    def kcluster(self):
        #设定最大迭代次数times
        times = 1
        for i in range(times):
            class_dict = self.count_distance()
            self.locate_center(class_dict)
            #print(self.data_dict)
            print("----------------迭代%d次----------------"%i)
            print(self.new_center)
            print(self.center)
            print(self.center_dict)
            if sorted(self.center) == sorted(self.new_center):
                break
            else:
                self.center = self.new_center
            if i < times-1 :    
                for j in range(self.k):
                    self.center_dict[j]["center"] = self.center[j]
                    self.center_dict[j]["items_dist"] = {}
                    self.center_dict[j]["classify"] = []            
        
    #self.center_dict{k:{{center:[]},{distance：{item：0.0}，{classify:[]}}}}
    #距离并分类
    def count_distance(self):
        #计算距离，比较的出最小值并分类
        min_list = [99999]
        class_dict = {}
        for i in range(len(self.data[0])):
            class_dict[self.data[0][i]] = 0
            min_list.append(99999)
        #计算距离
        for k in self.center_dict:
            #遍历row,manhadon计算距离
            for i in range(len(self.data[0])):
                #遍历column
                for col in range(1, len(self.data)):
                    self.center_dict[k]["items_dist"].setdefault(self.data[0][i], 0.0)
                    self.center_dict[k]["items_dist"][self.data[0][i]] += abs(self.data[col][i] - self.center_dict[k]["center"][col-1])
        # 分类 {item:clss}
            for i in range(len(self.data[0])):
                if self.center_dict[k]["items_dist"][self.data[0][i]] < min_list[i]:
                    min_list[i] = self.center_dict[k]["items_dist"][self.data[0][i]]
                    class_dict[self.data[0][i]] = k
        return class_dict
     
    # 计算新的中心点
    def locate_center(self, class_dict):
        # class_dict {'Boston Terrier': 0, 'Brittany Spaniel': 1, 
        #加入分类的列表
        for item_name, k in class_dict.items():
            self.center_dict[k]["classify"].append(item_name)
        #print(class_dict)
        #print(self.center_dict)
        self.new_center = [[0.0 for j in range(1, len(self.data))] for i in range(self.k)]
        for k in self.center_dict:
            for i in range(len(self.data)-1):
                for cls_item in self.center_dict[k]["classify"]:
                    self.new_center[k][i]  += self.data_dict[cls_item][i]
                self.new_center[k][i] /= len(self.center_dict[k]["classify"])
                
                 
a = KmeanppCluster(3, "C:/doc/ch8/dogs.csv.txt")
a.kcluster()        
