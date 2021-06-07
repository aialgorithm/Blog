#!/usr/bin/env python
# coding: utf-8


#  初始化SparkSession
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Python Spark RF example").config("spark.some.config.option", "some-value").getOrCreate()

# 加载数据
df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load("./data.csv",header=True)

from pyspark.sql.functions import *
# 数据基本信息分析

df.dtypes # Return df column names and data types
df.show()  #Display the content of df
df.head()  #Return first n rows
df.first()  #Return first row 
df.take(2)  #Return the first n rows
df.schema   # Return the schema of df
df.columns # Return the columns of df
df.count()  #Count the number of rows in df
df.distinct().count()  #Count the number of distinct rows in df
df.printSchema()  #Print the schema of df
df.explain()  #Print the (logical and physical)  plans
df.describe().show()  #Compute summary statistics 

df.groupBy('Survived').agg(avg("Age"),avg("Fare")).show()  # 聚合分析
df.select(df.Sex, df.Survived==1).show()  # 带条件查询 
df.sort("Age", ascending=False).collect() # 排序

df = df.dropDuplicates()   # 删除重复值

df = df.na.fill(value=0)  # 缺失填充值
df = df.na.drop()        # 或者删除缺失值


df = df.withColumn('isMale', when(df['Sex']=='male',1).otherwise(0)) # 新增列：性别0 1
df = df.drop('_c0','Name','Sex') # 删除姓名、性别、索引列

# 设定特征/标签列
from pyspark.ml.feature import VectorAssembler
ignore=['Survived']
vectorAssembler = VectorAssembler(inputCols=[x for x in df.columns  
                  if x not in ignore], outputCol = 'features')
new_df = vectorAssembler.transform(df)
new_df = new_df.select(['features', 'Survived'])

# 划分测试集训练集
train, test = new_df.randomSplit([0.75, 0.25], seed = 12345)

# 模型训练
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol = 'features', 
                         labelCol='Survived')
lr_model = lr.fit(test)

# 模型评估
from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = lr_model.transform(test)
auc = BinaryClassificationEvaluator().setLabelCol('Survived')
print('AUC of the model:' + str(auc.evaluate(predictions)))
print('features weights', lr_model.coefficientMatrix)
