#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os, time
import cv2
import matplotlib.pyplot as plt
import signal
import argparse

from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
from IPython import display

is_interrupted = False

def signal_handler(signal, frame):
    is_interrupted = True

    
class FaceRecog(object):
    
    __cascade_path = "./model/cv2/haarcascade_frontalface_alt2.xml" #用 OpenCV 的 Cascade classifier 來偵測臉部，不一定跟 Facenet 一樣要用 MTCNN。
    __model_path = "./model/keras/facenet_keras.h5" #使用 MS-Celeb-1M dataset pretrained 好的 Keras model
    __faces_path = "./faces/"
    
    def __init__(self, config):
        self.cascade = cv2.CascadeClassifier(FaceRecog.__cascade_path) #用 OpenCV 的 Cascade classifier 來偵測臉部
        self.model = load_model(FaceRecog.__model_path)
        self.margin = 10
        self.imgs_per_person = 5
        self.img_size = 160 #此版 Facenet model 需要的相片尺寸為 160×160
        self.faces_to_find = {}
        self._load_faces()
        self.vc = cv2.VideoCapture(0)
        self.sensitivity = 0.7 #delta值 
        self.show_img = config.showImgWindow

    def __del__(self):
        self.vc.release()
    
    def _load_faces(self):
        files = os.listdir(FaceRecog.__faces_path)
        
        for f in files:
            fullpath = os.path.join(FaceRecog.__faces_path, f)
 
            if os.path.isfile(fullpath):
                if os.path.splitext(f)[1] == '.jpg':
                    aligned = self._align_image(cv2.imread(fullpath), 6) 
                    
                    if aligned is not None:
                        print ("loading " + str(fullpath))
                        faceImg = self._pre_process(aligned)
                        self.faces_to_find[str(f)] = self._l2_normalize(np.concatenate(self.model.predict(faceImg)))
    
    def _prewhiten(self, x):
        #圖像白化（whitening）用於對過度曝光或低曝光的圖片進行處理，處理的方式就是改變圖像的平均像素值為 0 ，改變圖像的方差為單位方差 1。
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError("Dimension should be 3 or 4")

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        
        y = (x - mean) / std_adj
        
        return y
    
    def _l2_normalize(self, x, axis=-1, epsilon=1e-10):
        #使用 L1 或 L2 標準化圖像強化圖像特徵。
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

        return output
    
    def _align_image(self, img, margin):
        #偵測並取得臉孔 area，接著再 resize 為模型要求的尺寸（下方例子並未作alignment）
        faces = self.cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

        if(len(faces)>0):
            (x, y, w, h) = faces[0]
            face = img[y:y+h, x:x+w]
            faceMargin = np.zeros((h+self.margin*2, w+self.margin*2, 3), dtype = 'uint8')
            faceMargin[self.margin:self.margin+h, self.margin:self.margin+w] = face
            aligned = resize(faceMargin, (self.img_size, self.img_size), mode='reflect')

            return aligned
        else:
            return None
    
    def _pre_process(self, img):
        whitenImg = self._prewhiten(img)
        whitenImg = whitenImg[np.newaxis, :]
        return whitenImg
    
    def start(self, gadget=None):
        
        if len(self.faces_to_find) == 0:
            print ("no face file foune")
            return 1
        
        found_times = 0
        the_face = None
    
        if self.vc.isOpened():
            #is_capturing, _ = self.vc.read()
            #print ("capturing")
            pass
        else:
            is_capturing = False
            #print ("not capturing")
            return 1
            
        #while is_capturing:
        is_capturing, frame = self.vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=3)#,minSize=(160, 160))
            
        if len(faces) != 0:
            (x, y, w, h) = faces[0]

            if self.show_img == True:
                left = x - self.margin // 2
                right = x + w + self.margin // 2
                bottom = y - self.margin // 2
                top = y + h + self.margin // 2
                rimg = cv2.rectangle(frame,(left-1, bottom-1),(right+1, top+1),(255, 0, 0), thickness=2)
            
            faceMargin = np.zeros((h+self.margin*2, w+self.margin*2, 3), dtype = 'uint8')
            faceMargin[self.margin:self.margin+h, self.margin:self.margin+w] = frame[y:y+h, x:x+w]
            aligned = resize(faceMargin, (self.img_size, self.img_size), mode='reflect')
        
            if(aligned is not None):
                faceImg = self._pre_process(aligned)
                embs = self._l2_normalize(np.concatenate(self.model.predict(faceImg)))
                    
                lgap = 1
                for key in self.faces_to_find:
                    embs_valid = self.faces_to_find[key]
                    distanceNum = distance.euclidean(embs_valid, embs)
                    print ("diff: " + str(key) + " " + str(distanceNum))

                    if self.show_img:
                        cv2.putText(rimg, "diff: "+str(distanceNum), (x, y-(10*lgap)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        lgap = lgap + 3

                    if distanceNum < self.sensitivity:
                        found_times = found_times + 1
                        the_face = str(key)
                        break

            if self.show_img:            
                plt.imshow(frame)
                plt.title('faceplay')
                plt.xticks([])
                plt.yticks([])
                display.clear_output(wait=True)
        else:
            gadget.play(face_detected=False)
        
        try:
            plt.pause(0.1)
        except Exception:
            pass
        
        if gadget is not None:
            if found_times > 0:
                gadget.play(face_detected=True, detected_time=time.ctime(time.time()), the_face=the_face)
            else:
                gadget.play(face_detected=False)
        
        return 0 
             

def main():
    parser = argparse.ArgumentParser(description='FacePlay')
    parser.add_argument('--showImgWindow', default=False, type=bool, help='show image window with True, default is False')
    args = parser.parse_args()

    from gadget.gadget import FacePlayGadget

    facerecog = FaceRecog(args)
    gadget = FacePlayGadget()
    signal.signal(signal.SIGINT, signal_handler)
    
    print ("starting gadget....")
    
    while True:
        facerecog.start(gadget)
        
        time.sleep(0)
        
        if is_interrupted:
            break
        
    print ("gadget is stopped")

if __name__ == '__main__':
    main()
    
        
    


# In[ ]:





# In[ ]:




