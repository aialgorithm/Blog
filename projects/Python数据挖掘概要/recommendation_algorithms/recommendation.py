import codecs
from math import sqrt


class Recommendor():
    
    def __init__(self):
        self.frequencies = {}
        self.deviation = {}
        self.data = {}        
        self.movie_id = {}
        self.convert_movieid()
        self.load_data()
        #print(self.data["4"])

    # 加载电影数据对应标签    
    def convert_movieid(self):
        with codecs.open("C:\\doc\\BX-Dump\\ml-latest-small\\ml-latest-small\\movies.csv","r", "utf8") as file_object:
            for line in file_object:
                field = line.split(",")
                self.movie_id[field[0]] = field[1]
                
    # 加载电影数据，处理并保存为字典       
    def load_data(self):
        movie_dict = {}
        with codecs.open("C:\\doc\\BX-Dump\\ml-latest-small\\ml-latest-small\\ratings.csv","r", "utf8") as file_object:
            for line in file_object:
                field = line.split(",")
                user = field[0]
                movie = self.movie_id[field[1]]
                #print(movie)
                rating = float(field[2])
                movie_dict[movie] = rating
                self.data.setdefault(user,{})
                self.data[user].update(movie_dict)
                movie_dict = {}
                  
    

    #user_base协同推荐算法
    def user_base_recommendation(self, username): 
        remd_list = []
        #k位相似用户
        knn = 3
        data = self.data
        #pearson相关系数消除分数膨胀；数据离散，可以使用余弦相关；数据密集，可以使用曼哈顿距离；
        for user in data:
            if user != username:
                r = self.pearson(data[username], data[user])
                remd_list.append((r, user,data[user]))
        remd_list = sorted(remd_list, key=lambda x:x[0], reverse=True)[0:knn]
        #k最近邻算法
        recommend_movie = {}
        remd_sum = 0.0
        for remd in remd_list:
            remd_sum += remd[0]
        if remd_sum == 0:
            return 0
        for remd in remd_list:
            weight = remd[0] / remd_sum
            for k,v in remd[2].items():
                if k not in data[username]:
                    if k not in recommend_movie:
                        recommend_movie[k] = float(v) * weight
                    else :
                        recommend_movie[k] += float(v) * weight
        recommend_movie = sorted(recommend_movie.items(), key=lambda x:x[1], reverse=True)
        #返回综合评价大于3分的电影
        print({key:value for key,value in recommend_movie if value > 3})
        return {key:value for key,value in recommend_movie if value > 3}
    
    #皮尔逊相关系数
    def pearson(self, user_x, user_y):
        x_sum = 0
        y_sum = 0
        xy_sum = 0
        x2_sum = 0
        y2_sum = 0
        n = 0
        r = 0
        for fav_key in user_x:
            if fav_key in user_y:
                if user_x[fav_key] != "" and user_y[fav_key] != "":
                    x_sum += int(user_x[fav_key])
                    y_sum += int(user_y[fav_key])
                    xy_sum += int(user_x[fav_key]) * int(user_y[fav_key])
                    x2_sum += pow(int(user_x[fav_key]), 2)
                    y2_sum += pow(int(user_y[fav_key]), 2)
                    n += 1
        if n == 0:
            return 0
        else:
            fenmu = (sqrt(x2_sum - pow(x_sum, 2) / n) * sqrt(y2_sum - pow(y_sum, 2) / n))
            if fenmu == 0:
                return 0
            else:
                r = (xy_sum - x_sum * y_sum / n) / fenmu
                return r
    
    #余弦相关系数      
    def cosxy(self, user_x, user_y):
        fenzi = 0
        fenmu = 0
        x2_sum = 0
        y2_sum = 0
        for fav_key in user_x:
            if fav_key in user_y:
                fenzi += user_x[fav_key] * user_y[fav_key]
                x2_sum += pow(user_x[fav_key], 2)
                y2_sum += pow(user_y[fav_key], 2)
        fenmu = sqrt(x2_sum) * sqrt(y2_sum)
        if fenmu != 0:
            return (fenzi / fenmu)
        else :
            return 0
        
    #曼哈顿距离   
    def manhaton(self, user1, user2):
        distance = 0
        for key in user1:
            if key in user2:
                distance += abs(user1[key] - user2[key])
        return distance
            
    def compute_deviation(self):
        for ratings in self.data.values():
            for item,rating in ratings.items():
                self.frequencies.setdefault(item, {})
                self.deviation.setdefault(item, {})
                for item2,rating2 in ratings.items():
                    if item != item2:
                        self.frequencies[item].setdefault(item2, 0)
                        self.deviation[item].setdefault(item2, 0)
                        self.frequencies[item][item2] += 1
                        self.deviation[item][item2] += rating - rating2            
        for item,ratings in self.deviation.items():
            for item2 in ratings:
                ratings[item2] /= self.frequencies[item][item2] 

    #items base协同推荐算法 slope one ：为user_a预测item_j的评分：
    def slope_one_recommendation(self, user_a):
        self.compute_deviation()
        recommendations = {}
        fenmu = {}
        for useritem,userrating in self.data[user_a].items():
            for diffitem,diffrating in self.deviation.items():
                if diffitem not in self.data[user_a] and useritem in self.deviation[diffitem]:
                    freq = self.frequencies[diffitem][useritem]
                    recommendations.setdefault(diffitem, 0)
                    fenmu.setdefault(diffitem, 0)
                    recommendations[diffitem] += (diffrating[useritem] + userrating) * freq
                    fenmu[diffitem] += freq
        recommendations = [(k, v/fenmu[k]) for (k,v) in recommendations.items()]
        recommendations.sort(key=lambda x:x[1],reverse=True)
        print(recommendations[0:3])
        return recommendations[0:3]
    
    
    def cos_aver(self, item_a, item_b):
        users3 = self.data
        fenzi = 0
        powsum_a = 0
        powsum_b = 0
        aver_dict = {}
        for user in users3:
            score_aver = sum(users3[user].values()) / len(users3[user].values())
            #print("score_aver",score_aver)
            aver_dict[user] = score_aver
            if item_a in users3[user] and item_b in users3[user]:
                #print("ok",users3[user][item_a],users3[user][item_b])
                fenzi += (users3[user][item_a] - aver_dict[user]) * (users3[user][item_b] - aver_dict[user])
                powsum_a += pow((users3[user][item_a] - aver_dict[user]), 2)
                powsum_b += pow((users3[user][item_b] - aver_dict[user]), 2) 
        fenmu = sqrt(powsum_a) * sqrt(powsum_b)        
        if fenmu != 0:
            return fenzi / fenmu

    #items base协同推荐算法  修正余弦相似度 
    def cos_recommendation(self, user_a):
    #aver_dict = {}
        users3 = self.data
        s_list = []
        for item_a in users3[user_a] :
            for item_b in self.movie_id.values():
                if item_b != item_a and item_b not in users3[user_a]:
                # print((item_a, item_b))
                    s = self.cos_aver(item_a, item_b)
                    if s :
                        s_list.append((item_a, item_b, s))
        s_list.sort(key=lambda x:x[2],reverse=True)
        print(s_list)
        return s_list
    
#users2 测试集
users2 = {"Amy": {"Taylor Swift": 4, "PSY": 3, "Whitney Houston": 4},
          "Ben": {"Taylor Swift": 5, "PSY": 2},
          "Clara": {"PSY": 3.5, "Whitney Houston": 4},
          "Daisy": {"Taylor Swift": 5, "Whitney Houston": 3}}

  

r = Recommendor()

print("items base协同推荐 slope one")
#items base协同推荐算法 Slope one
r.slope_one_recommendation('lyy')

print("items base协同推荐 cos")
#items base协同推荐算法  修正余弦相似度 
r.cos_recommendation('lyy')

print("users base协同推荐")
#userbase协同推荐算法 
r.user_base_recommendation("lyy")
