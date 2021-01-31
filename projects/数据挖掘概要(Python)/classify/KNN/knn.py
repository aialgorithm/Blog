import numpy as np
import copy

class Clsssify():
    def __init__(self):
        # 加载训练数据、测试数据
        self.train_data = self.loaddb("C:\\doc\\ch4\\irisTrainingSet.data.txt") 
        self.test_data = self.loaddb("C:\\doc\\ch4\\irisTestSet.data.txt")
        
    def loaddb(self, filename):
        train_data = []
        with open(filename, "r") as fileobject:
            for line in fileobject.readlines():
                line = line.strip("\n").split("\t")
                for i in range(4):
                    line[i] = float(line[i])
                train_data.append(line)
        #train_data.pop(0)
        return train_data
    
    #中位数
    def median(self, alist):
        if len(alist) % 2 == 1:
            return alist[int(len(alist) / 2)]
        else:
            return (alist[int(len(alist) / 2)] + alist[int(len(alist) / 2) - 1]) / 2.0
        
    #绝对偏差
    def asd(self, alist, median):
        return np.sum(abs(np.array(alist) - median))/len(alist)
     
    #z-factor修正标准化（准确率93.3%）
    def vector_standard(self, data):
        median_list = []
        asd_list = []
        st_list = copy.deepcopy(data)
        #print(self.train_data)
        for i in range(4):
            col_list = [alist[i] for alist in data]
            alist = sorted(col_list)
            median_list.append(self.median(alist)) 
            asd_list.append(self.asd(alist, median_list[i]))
            j = 0
            for line in st_list:
                st_list[j][i] = ((line[i] - median_list[i]) / asd_list[i])
                j += 1
            #st_list.append((np.array(col_list) - median_list[i]) / asd_list[i])
        return st_list
    
    #max-min标准化，准确率83%
    def vector_s(self, data):
        max_list = []
        min_list = []
        st_list = copy.deepcopy(data)
        for i in range(4):
            col_list = [alist[i] for alist in data]
            alist = sorted(col_list)
            min_list.append(alist[0])
            max_list.append(alist[-1])
            j = 0
            for line in st_list:
                st_list[j][i] = ((line[i] - min_list[i]) / (max_list[i] - min_list[i]))
                j += 1
        return st_list
            
    
    def manhanton(self,v1,v2):
        r = np.sum(abs(v2 - v1))
        return r
    
    #manhton距离计算特征值距离
    def recommentor_manh(self):
        st_train = self.vector_standard(self.train_data)
        st_test = self.vector_standard(self.test_data)
        right = 0.0
        test_list = []
        manh_list = []
        for test in st_test:
            v1 = np.array(test[0:4])
            for train in st_train:
                v2 = np.array(train[0:4])
                r = self.manhanton(v1, v2)
                manh_list.append((r, train[4]))
            rel_class = min(manh_list)[1]
            manh_list = []
            #test_list.append((test,rel_class))
            if rel_class == test[4]:
                right += 1
        acurrency = (right / len(st_test)) * 100
        return acurrency
    
    # KNN算法
    def knn(self, oj_list):
        weight_dict = {"Iris-setosa":0.0, "Iris-versicolor":0.0, "Iris-virginica":0.0}
        for atuple in oj_list:
            weight_dict[atuple[1]] += (1.0 / atuple[0])
        rel_class = [(key, value) for key, value in weight_dict.items()]
        #print(sorted(rel_class, key=lambda x:x[1], reverse=True))
        rel_class = sorted(rel_class, key=lambda x:x[1], reverse=True)[0][0]
        return rel_class
            
    def oj(self, v1, v2):
        return np.sqrt(sum(map(lambda v1, v2: pow((v1 - v2), 2), v1, v2)))      
    
    #欧几里得距离计算特征值距离
    def recommentor_oj(self, st_train, st_test):
        #knn系数
        k = 3
        #st_train = self.vector_standard(self.train_data)
        #st_test = self.vector_standard(self.test_data)
        #st_train = self.train_data
        #st_test = self.test_data
        right = 0
        oj_list = []
        for test in st_test:
            v1 = test[0:4]
            for train in st_train:
                v2 = train[0:4]
                r = self.oj(v1, v2)
                oj_list.append((r, train[4]))
            #print (min(oj_list),test[4])
            oj_list = sorted(oj_list, key=lambda x:x[0])[0:k]
            rel_class = self.knn(oj_list)
            #print("test",test[4])
            if rel_class == test[4]:
                right += 1
            oj_list =[]
            #每个test的数据清空
        acurrency = (right / len(st_test)) * 100
        return acurrency
    
    #留一法测试：准确93.3%
    def leaveone_test(self):
        st_data = self.vector_standard(self.train_data)
        output_list = []
        for st_test in st_data:
            st_train = copy.deepcopy(st_data)
            st_train.remove(st_test)
            output_list.append(self.recommentor_oj(st_train, [st_test]))
        print(output_list.count(100)/len(output_list), output_list)
            
              
                
c = Clsssify()
c.leaveone_test() 
    
