import subprocess

class FacePlayGadget(object):
    #def __init__(self):
        #subprocess.run(['sudo', 'uhubctl', '-l', '2-1', '-p', '1', '-a', 'off']) #turn off the usb port by by default

    #def __del__(self):
        #return

    def play(self, **kwargs):
        
        if kwargs.get("face_detected") is not None:
            if kwargs.get("face_detected") == True:
                print ('the target face %s is detected, time: %s' % (kwargs.get("the_face"), kwargs.get("detected_time")))
                #subprocess.run(['sudo', 'uhubctl', '-l', '2-1', '-p', '1', '-a', 'on']) #turn off the usb port
            else:
                print ('no target faces are detected...')
                #subprocess.run(['sudo', 'uhubctl', '-l', '2-1', '-p', '1', '-a', 'off']) #turn off the usb port
        
