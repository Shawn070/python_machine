import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
    f = open(filePath, 'rb')    # 以二进制形式读取文件
    data = []
    img = image.open(f)         # 以列表形式返回图片像素值
    m, n = img.size             # 获得图片的大小
    for i in range(m):          # 将每个像素点RGB颜色处理到0-1
        for j in range(n):      # 范围内并存放进data
            x, y, z = img.getpixel((i, j))
            data.append([x/256.0, y/256.0, z/256.0])
    f.close()
    return np.mat(data), m, n   # 以矩阵形式返回data， 以及图片大小

imgData, row, col = loadData('bull.jpg') # 加载数据
# 聚类获取每个像素点的类别
label = KMeans(n_clusters=6).fit_predict(imgData)
label = label.reshape(row, col)
# 创建一张新的灰度图保存聚类后的结果
pic_new = image.new('L', (row, col))
# 根据所属类别向图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
# 以JPEG形式保存图像
pic_new.save('bull-result-6.jpg', "JPEG")