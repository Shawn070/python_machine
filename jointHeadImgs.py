import os
from math import sqrt
from PIL import Image

path = 'E:/HeadImages/'
pathList = []
for item in os.listdir(path):
    imgPath = os.path.join(path, item)
    pathList.append(imgPath)
total = len(pathList)   # 头像个数
line = int(sqrt(total)) # line拼接图片行数
NewImage = Image.new('RGB', (128*line, 128*line))
x = y = 0
for item in pathList:
    try:
        img = Image.open(item)
        img = img.resize((128, 128), Image.ANTIALIAS)
        NewImage.paste(img, (x*128, y*128))
        x += 1
    except IOError:
        print("第%d行，%d列文件读取失败！IOError:%s" % (y, x, item))
        x -= 1
    if x == line:
        x = 0
        y += 1
    if (x+line*y) == line*line:
        break
NewImage.save(path+"final.jpg")