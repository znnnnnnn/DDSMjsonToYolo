import os
import random
import shutil
import numpy as np
import json
import PIL.Image
import PIL.ImageDraw
from shapely.geometry import Polygon


def getbbox(points, height, width):
    polygons = points
    mask = polygons_to_mask([height, width], polygons)
    return mask2box(mask,height,width)       #x_center,y_center,w,h归一化后的结果

def mask2box(mask,image_height,image_width):     #取出x_center,y_center,w,h的同时归一化
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]

    left_top_r = np.min(rows)  # y1
    left_top_c = np.min(clos)  # x1

    right_bottom_r = np.max(rows)   #y2
    right_bottom_c = np.max(clos)   #x2

    bboxw = right_bottom_c - left_top_c
    bboxh = right_bottom_r - left_top_r

    x_center = (left_top_c + right_bottom_c)/(2.0 * image_width)
    y_center = (left_top_r + right_bottom_r)/(2.0 * image_height)
    w = bboxw/image_width
    h = bboxh/image_height

    return [x_center, y_center, w , h]  # [x,y,w,h] for yolo box format


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def get_yolo(labelme_json_path, save_yolo_path):
    # 获取每个json的信息
    for labelme_json in os.listdir(labelme_json_path):
        annotations = []    #每个文件的标注信息用一个列表存
        if labelme_json[-3:] == 'png' or labelme_json[-3:] == 'jpg':
            continue
        with open(labelme_json_path+labelme_json, 'r') as fp:
            # load json
            json_data = json.load(fp)
        image_h = json_data["imageHeight"]  #归一化用
        image_w = json_data["imageWidth"]   #归一化用
        image_name = str(json_data["imagePath"][2:]).rstrip(".jpg") #写文件的文件名用

        # 获取json中的标注信息
        for shape in json_data['shapes']:           #一张图片中的所有标注
            with open(save_yolo_path + image_name + '.txt', 'a', encoding='utf-8') as fp:
                if shape['label']=='mass':
                    fp.write('0 ')  # 先写类别：为肿块
                elif shape['label']=='calcification':
                    fp.write('1 ')  # 先写类别：为钙化
                else:
                    continue        #other类别的不考虑，跳过
            for i in range(len(shape['points'])):   #一张图片中的一处标注
                shape['points'][i][0] = float(shape['points'][i][0])    #把标注顶点都变为浮点数
                shape['points'][i][1] = float(shape['points'][i][1])
            points = shape['points']
            anot = list(map(float, getbbox(points, image_h, image_w)))
            for j in anot:
                with open(save_yolo_path+image_name+'.txt', 'a', encoding='utf-8') as fp:
                    fp.write(str(j)+' ')
            with open(save_yolo_path+image_name+'.txt', 'a', encoding='utf-8') as fp:
                fp.write('\n')


if __name__ == "__main__":
    labelme_folder = "C:\\Users\\ning\\Desktop\\val\\"
    save_yolo_path = "C:\\Users\\ning\\Desktop\\cbis_ddsm\\val\\"
    #os.makedirs(save_yolo_path.rstrip('\\'))
    get_yolo(labelme_folder, save_yolo_path)
    #for i in range(10,15):
     #   labelme_folder = "C:\\Users\\ning\\Documents\\learn_document\\mass\\data\\annotations(jsonyolo)\\json\\benigns\\benign_"+str(i)+"\\"
      #  save_yolo_path = "C:\\Users\\ning\\Documents\\learn_document\\mass\\data\\annotations(jsonyolo)\\yolo\\benigns\\benign_"+str(i)+"\\"
       # os.makedirs(save_yolo_path.rstrip('\\'))
        #get_yolo(labelme_folder, save_yolo_path)
