import numpy as np
from sklearn.cluster import KMeans


def loadData(filePath):
    fr = open('E:/C语言入门教程/Python/python_machine/city.txt', 'r+')   # r+:读写
    lines = fr.readlines()      # .readlines()一次读取整个文件（类似于.read()）
    retData = []        # 城市的各项消费
    retCityName = []    # 城市名称
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName

if __name__=='__main__':
    data, cityName = loadData('city.txt')
    km = KMeans(n_clusters = 3)     # 聚类数
    label = km.fit_predict(data)    # 计算簇中心以及为簇分配序号,label为每行数据对应分配到的序列
    expenses = np.sum(km.cluster_centers_, axis = 1)    # 按行求和

    # print(expenses)
    CityCluster = [[], [], []]
    # 将在同一个簇的城市保存在同一个list中
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
        #输出各个簇的平均消费数和对应的城市名称
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])

'''
Expenses:5113.54
['天津', '江苏', '浙江', '福建', '湖南', '广西', '海南', '重庆', '四川', '云南', '西藏']
Expenses:3827.87
['河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江', '安徽', '江西', '山东', '河南', '湖北', '贵州', '陕西', '甘肃', '青海', '宁夏', '新疆']
Expenses:7754.66
['北京', '上海', '广东']

***Repl Closed***
'''