#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import os, time
import cv2
import signal
import argparse
import face_recognition
import platform

from IPython import display

is_interrupted = False

def signal_handler(signal, frame):
    is_interrupted = True

    
class FaceRecog(object):
    __faces_path = "./faces/"
    
    def __init__(self, config):
        self.face_distance = config.distance
        self.show_img = config.display
        self.faces_to_find = []
        self.faces_to_find_imgs = []
        self._load_target_faces()

        if platform.machine() == "aarch64":
            self.vc = cv2.VideoCapture(_get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
        else:
            self.vc = cv2.VideoCapture(0)

    def __del__(self):
        self.vc.release()

    def _get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
        """
        Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
        """
        return (
                f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
                f'width=(int){capture_width}, height=(int){capture_height}, ' +
                f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
                f'nvvidconv flip-method={flip_method} ! ' +
                f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
                'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
                )

    def _load_target_faces(self):
        files = os.listdir(FaceRecog.__faces_path)
        
        for f in files:
            fullpath = os.path.join(FaceRecog.__faces_path, f)
 
            if os.path.isfile(fullpath):
                if os.path.splitext(f)[1] == '.jpg':
                    face = face_recognition.load_image_file(fullpath)

                    print ("loading " + str(fullpath))
                    try:
                        face_encoding = face_recognition.face_encodings(face)[0]
                        self.faces_to_find.append(face_encoding)
                        self.faces_to_find_imgs.append(str(f))
                    except IndexError:
                        #no face found in the loading picture
                        pass
                        
    def _lookup_target_face(self, face_encoding):
        """
        See if this is a face we already have in the face list
        """
        found_face_index = None

        # If known face list is empty, just return nothing since we can't possibly have seen this face.
        if len(self.faces_to_find) == 0:
            return None

        # Calculate the face distance between the unknown face and every face on in our known face list
        # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
        # the more similar that face was to the unknown face.
        face_distances = face_recognition.face_distance(self.faces_to_find, face_encoding)
        #print (face_distances)

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)
        #print (best_match_index)
        print (face_distances[best_match_index])

        # If the face with the lowest distance had a distance under the value defined in self.face_distance, we consider it a face match.
        # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
        # of the same person always were less than 0.6 away from each other.
        if face_distances[best_match_index] < self.face_distance:
            found_face_index = best_match_index
           
        return found_face_index
    
    def start(self, gadget=None):
        
        if len(self.faces_to_find) == 0:
            print ("no target faces loaded")
            return 1
        
        the_face = None
        stop = False
    
        if self.vc.isOpened():
            #print ("capturing")
            pass
        else:
            print ("not capturing")
            return 1
            
        while True:
            is_capturing, frame = self.vc.read()
          
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the face locations and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            found_face_index = None
            found_face_imgs = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                found_face_index = self._lookup_target_face(face_encoding)

                if found_face_index is not None:
                    found_face_imgs.append(self.faces_to_find_imgs[found_face_index])

                if self.show_img:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

                    if found_face_index is not None:
                        cv2.putText(frame, self.faces_to_find_imgs[found_face_index], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, 'not a target face', (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    cv2.imshow('FacePlay', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop = True
                        break
                else:
                    time.sleep(0.5)
                    if is_interrupted:
                        stop = True
                        break

            if gadget is not None:
                if found_face_index is not None:
                    gadget.play(face_detected=True, detected_time=time.ctime(time.time()), the_face=found_face_imgs)
                else:
                    gadget.play(face_detected=False)

            if stop:
                break

        return 0 
             

def main():
    parser = argparse.ArgumentParser(description='FacePlay')
    parser.add_argument('--display', default=1, type=int, help='display image window, set 1 to display the windown, otherwise set 0. default is 1')
    parser.add_argument('--distance', default=0.5, type=float, help='face distance for match, default is 0.5')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    from gadget.gadget import FacePlayGadget

    facerecog = FaceRecog(args)
    gadget = FacePlayGadget()
    
    print ("starting faceplay...")
    
    facerecog.start(gadget)
        
    print ("faceplay is stopped")

if __name__ == '__main__':
    main()
    
        
    


# In[ ]:





# In[ ]:




