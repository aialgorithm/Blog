import os,math
categories = os.listdir('C:\\doc\\ch7\\20news-bydate\\20news-bydate-train\\alt.atheism')
categories = [filename for filename in categories
                           if os.path.isdir('C:\\doc\\ch7\\20news-bydate\\20news-bydate-train\\' + filename)]



#非结构化文本分类
class Bayes_unstructure():
    def __init__(self):
        self.vocabulary = {}
        self.total = {}
        self.prob = {}
        self.load_data_clean()
         
    def load_data_clean(self):
        
        #加载停词表 stoplist174:83.49%,nostoplist:82.87%, stoplist: 83.36%
        #停词表，去除这些词的理由是：
        #1.	能够减少需要处理的数据量； 2.这些词的存在会对分类效果产生负面影响。
        # H.P.	Luhn在他的论文中说“这些组成语法结构的单词是没有意义的，反而会产生很多 噪音”。
        with open('C:\\doc\\ch7\\20news-bydate\\stopwords174.txt', "r") as file_object:
            stopwords = file_object.read().split("\n")
        #加载训练数据
        train_dir = 'C:\\doc\\ch7\\20news-bydate\\20news-bydate-train\\'
        categories = os.listdir(train_dir)
        for category in categories:
            files = os.listdir(train_dir + category)
            contend = ""
            self.total[category] = 0 
            for file in files:
                with open(train_dir + category + "\\" + file, "r", encoding='iso8859-1') as file_object:
                    # encoding utf-8 该gbk unused
                    contend += file_object.read()
            contend = contend.split()
            for lista in contend:
                #分词并去除大小写以及无效字符
                lista = lista.strip('\'".,?:/|/\-<>_()=+^*%~#!').lower()
                # 计算停词表外的单词的频数
                if lista != "" and lista not in stopwords:
                    if category not in self.prob:
                        self.prob[category] = {}
                    if lista not in self.prob[category]:
                        self.prob[category][lista] = 0
                    if lista not in self.vocabulary:
                        self.vocabulary[lista] = 0
                    self.prob[category][lista] += 1
                    self.total[category] += 1
                    self.vocabulary[lista] += 1
        #删除总频数较低的词汇 删除频数为1：83.39，3:83.5%，10：83.5%，50:83.56
        #误删除分类内频数低于3的的词汇，导致准确率底下%2，优化后82.51460435475305%
        to_del = []
        for word,value in self.vocabulary.items():
            if value <= 10:
                to_del.append(word)
                #total 减去无效次数后提高至 83.5%
                for category in categories:
                    if word in self.prob[category]:
                        self.total[category] -= self.prob[category][word]
        for word in to_del:
            del self.vocabulary[word]
         
    def train_data(self):
        #训练组的条件概率
        for word in self.vocabulary:
            for category,value in self.prob.items():
                if word not in self.prob[category]:
                    #count初始值由1优化0 准确率提高1.39%，0优化为-0.999准确率:-0.4%,-0.9:+0.2%
                    count = 0
                else :
                    count = self.prob[category][word]
                #优化的条件概率 83.5%
                self.prob[category][word] = (count + 1) / (self.total[category] + len(self.vocabulary)) 
                
    def test_data(self):
        #测试数据
        total = 0.0
        right = 0.0
        test_dir = 'C:\\doc\\ch7\\20news-bydate\\20news-bydate-test\\'
        categories = os.listdir(test_dir)
        for category in categories:
            files = os.listdir(test_dir + category)
            
            for file in files:
                predict = {}
                with open(test_dir + category + "\\" + file, "r", encoding='iso8859-1') as file_object:
                    # encoding utf-8 ，gbk no useful
                    contend = file_object.read().split()
                for pre_category in categories:
                    predict[pre_category] = 0
                    #遍历文章的所有有效词汇
                    for lista in contend:
                        lista = lista.strip('\'".,?:/|/\-<>_()=+^*%~#!').lower()
                        #使用对数形式运算极小数值
                        if lista in self.prob[pre_category]:
                            predict[pre_category] += math.log(self.prob[pre_category][lista])
                #预测分类
                pre_category = max(predict.items(), key=lambda x:x[1])[0]
                total += 1
                if pre_category == category:
                    right += 1
        accurency = right / total
        print(accurency)                


    
b = Bayes_unstructure()
b.train_data()
b.test_data()
