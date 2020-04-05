
class FacePlayGadget(object):
    
    def play(self, **kwargs):
        
        if kwargs.get("face_detected") is not None:
            if kwargs.get("face_detected") == True:
                print (">>>>>>>>>>>Kid is watching  TV! Turn it off<<<<<<<<<<<<<<<<")
                print (kwargs.get("detected_time"))
                print (kwargs.get("the_face"))
            else:
                print (">>>>>>>>>>>no kid found<<<<<<<<<<<<<")
        
