#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import os, time
import cv2
import signal
import argparse
import face_recognition

from IPython import display

is_interrupted = False

def signal_handler(signal, frame):
    is_interrupted = True

    
class FaceRecog(object):
    __faces_path = "./faces/"
    
    def __init__(self, config):
        self.face_distance = config.distance
        self.show_img = config.display
        self.faces_to_find = []#{}
        self.faces_to_find_imgs = []
        self.vc = cv2.VideoCapture(0)
        self._load_faces()

    def __del__(self):
        self.vc.release()
        
    def _load_faces(self):
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
        See if this is a face we already have in our face list
        """
        #metadata = None
        found_face_index = None

        # If our known face list is empty, just return nothing since we can't possibly have seen this face.
        if len(self.faces_to_find) == 0:
            return isfound

        # Calculate the face distance between the unknown face and every face on in our known face list
        # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
        # the more similar that face was to the unknown face.
        face_distances = face_recognition.face_distance(self.faces_to_find, face_encoding)
        #print (face_distances)

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)
        #print (best_match_index)
        print (face_distances[best_match_index])

        # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
        # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
        # of the same person always were less than 0.6 away from each other.
        # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
        # people will come up to the door at the same time.
        if face_distances[best_match_index] < self.face_distance:
            found_face_index = best_match_index
            # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
            #metadata = known_face_metadata[best_match_index]

            # Update the metadata for the face so we can keep track of how recently we have seen this face.
            #metadata["last_seen"] = datetime.now()
            #metadata["seen_frames"] += 1

            # We'll also keep a total "seen count" that tracks how many times this person has come to the door.
            # But we can say that if we have seen this person within the last 5 minutes, it is still the same
            # visit, not a new visit. But if they go away for awhile and come back, that is a new visit.
            #if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=5):
                #metadata["first_seen_this_interaction"] = datetime.now()
                #metadata["seen_count"] += 1

        #return metadata
        return found_face_index
    
    def start(self, gadget=None):
        
        if len(self.faces_to_find) == 0:
            print ("no target faces loaded")
            return 1
        
        #isfound = False
        the_face = None
        stop = False
    
        if self.vc.isOpened():
            #is_capturing, _ = self.vc.read()
            #print ("capturing")
            pass
        else:
            is_capturing = False
            #print ("not capturing")
            return 1
            
        #while is_capturing:
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

            #for face_location, face_encoding in zip(face_locations, face_encodings):
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                found_face_index = self._lookup_target_face(face_encoding)

                if found_face_index is not None:
                    #print ("found")
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




