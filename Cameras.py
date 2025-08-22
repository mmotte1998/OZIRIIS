from astropy.io import fits
import os
import time
import numpy as np
import zmq
import matplotlib.pyplot as plt
import dao

class Cred2():
    """
    Class to control cblue object
    """
    def __init__(self,simu = 1, nFrames=1):
        # shm
        self.shm = dao.shm('/tmp/cred2.im.shm')
        self.shmFps = dao.shm('/tmp/cred2Fps.im.shm')
        self.shmDit = dao.shm('/tmp/cred2Dit.im.shm')
        self.simulation_mode = simu
        if self.simulation_mode == 0:
            os.system('daoHwCamCtrl.py -s /tmp/cred2.im.shm &')
        self.nFrames = nFrames
        self.nPx_x = 512
        self.nPx_y = 640
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        
    def get(self,nFrames=None):
        # Send RTC command
        if nFrames is None:
            nFrames = self._nFrames
        data = 0
        if self.simulation_mode == 0:
            for k in range(0,self._nFrames):
                data += self.shm.get_data(check=True).astype(np.float64)
            data /= self._nFrames
        else:
            data = np.random.randn(self.nPx_x, self.nPx_y)
        return data - self.dark

    def get_dark(self):
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        self.dark = self.get()
        print('dark updated')

    def set_dark(self, dark_user):

        self.dark = dark_user
        print('dark updated')
    @property
    def nFrames(self):
        return self._nFrames

    @nFrames.setter
    def nFrames(self, val):
        self.set_nFrames(val)
    
    def set_nFrames(self, val):
        if val < 1:
            raise ValueError("Value cannot be zero or negative!")
        self._nFrames = round(val)

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, val):
        self.set_fps(val)
    
    def set_fps(self, val):
        if self.simulation_mode == 0:
            self.shmFps.set_data(self.shmFps.get_data()+val)
        self._fps = val
        print('FPS changed, do not forget to update dark !')

    @property
    def tint(self):
        return self._tint

    @tint.setter
    def tint(self, val):
        self.set_exp(val)
    
    def set_exp(self, val):
        if self.simulation_mode == 0:
            self.shmDit.set_data(self.shmDit.get_data()*0+val)
        self._tint = val
        print('INTEGRATION TIME changed, do not forget to update dark !')
    
    def shutdown(self):
        pass
              
    def show(self,img = None):
        if img is None:
            img = self.get()
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.show(block=False)

class Cblue():
    """
    Class to control cblue object
    """
    def __init__(self,simu = 1, nFrames=1):
        # shm
        self.shm = dao.shm('/tmp/cblue.im.shm')
        self.shmFps = dao.shm('/tmp/cblueFps.im.shm')
        self.simulation_mode = simu
        if self.simulation_mode == 0:
            os.system('daoImageRTD -s /tmp/cblue.im.shm')
        self.nFrames = nFrames
        self.nPx_x = 816
        self.nPx_y = 624
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        
    def get(self,nFrames=None):
        # Send RTC command
        if nFrames is None:
            nFrames = self._nFrames
        data = 0
        if self.simulation_mode == 0:
            for k in range(0,self._nFrames):
                data += self.shm.get_data(check=True)
            data /= self._nFrames
        else:
            data = np.random.randn(self.nPx_x, self.nPx_y)
        return data - self.dark

    def get_dark(self):
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        self.dark = self.get()
        print('dark updated')
    
    @property
    def nFrames(self):
        return self._nFrames

    @nFrames.setter
    def nFrames(self, val):
        self.set_nFrames(val)
    
    def set_nFrames(self, val):
        if val < 1:
            raise ValueError("Value cannot be zero or negative!")
        self._nFrames = round(val)

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, val):
        self.set_fps(val)
    
    def set_fps(self, val):
        if self.simulation_mode == 0:
            self.shmFps.set_data(self.shmFps.get_data()+val)
        print('FPS changed, do not forget to update dark !')

    def shutdown(self):
        pass
              
    def show(self,img = None):
        if img is None:
            img = self.get()
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.show(block=False)


class Ocam2k():
    """
    Class to control OCAM2K object
    """
    def __init__(self,simu = 1, nFrames=1):
        # shm
        self.simulation_mode = simu
        if self.simulation_mode == 0:
            self.shm = dao.shm('/tmp/papyrus_cal_pix.im.shm')
        self.nFrames = nFrames
        self.nPx_x = 240
        self.nPx_y = 240
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        
    def get(self, nFrames = None):
        # Send RTC command
        if nFrames is None:
            nFrames = self._nFrames
        data = 0
        if self.simulation_mode == 0:
            for k in range(0,self._nFrames):
                data += self.shm.get_data(check=True)
            data /= self._nFrames
        else:
            data = np.random.randn(self.nPx_x, self.nPx_y)
        return data
    
    @property
    def nFrames(self):
        return self._nFrames

    @nFrames.setter
    def nFrames(self, val):
        self.set_nFrames(val)
    
    def set_nFrames(self, val):
        if val < 1:
            raise ValueError("Value cannot be zero or negative!")
        self._nFrames = round(val)

    def shutdown(self):
        pass
              
    def show(self,img = None):
        if img is None:
            img = self.get()
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.show(block=False)


class Cred3():
    """
    Class to control CRED3 object through ssh
    """
    def __init__(self,simu = 1, nFrames=1):
        # Launch sender on DAO2 imager computer
        # Step 1: Start (or ensure) tmux session exists
        os.system("ssh manip@172.20.150.67 'tmux new-session -d -s sender'")
        # Step 2: Send command to the tmux session
        os.system("ssh manip@172.20.150.67 'tmux send-keys -t sender \"python /home/manip/dao/daoPapyrus/sendCommand.py\" C-m'")
        self.simulation_mode = simu
        if self.simulation_mode == 0:
            os.system("ssh manip@172.20.150.67 '~/test/daoHwCamCtrlSR.py -s /tmp/cred3.im.shm'")
        self.nFrames = nFrames
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://172.20.150.67:5558")
        self.nPx_x = 512
        self.nPx_y = 640
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        self.timeout = 100
        
    def get(self):
        # Send RTC command
        if self.simulation_mode == 0:
            self.socket.send_string(str(self._nFrames))
            data_socket = self.socket.recv()
            data = np.frombuffer(data_socket, dtype=np.float32).reshape((self.nPx_x, self.nPx_y)).astype(np.float32)
        else:
            data = np.random.randn(self.nPx_x, self.nPx_y)
        return data-self.dark
    
    @property
    def nFrames(self):
        return self._nFrames

    @nFrames.setter
    def nFrames(self, val):
        self.set_nFrames(val)
    
    def set_nFrames(self, val):
        if val < 1:
            raise ValueError("Value cannot be zero or negative!")
        self._nFrames = round(val)
    
    def get_dark(self):
        self.dark = np.zeros((self.nPx_x,self.nPx_y))
        self.dark = self.get()
        print('dark updated')

    def shutdown(self):
        os.system("ssh manip@172.20.150.67 'tmux kill-session -t sender'")
              
    def show(self,img = None):
        if img is None:
            img = self.get()
        plt.figure()
        plt.imshow(img)
        plt.colorbar()
        plt.show(block=False)

    '''
    NOT YET IMPLEMENTED - not sure how to do that with RTC commands    
    @property
    def exp_ms(self):
        return 

    @exp_ms.setter
    def exp_ms(self, val):
        self.set_exp_ms(val)
    
    def set_exp_ms(self, val):
  
    @property
    def gain(self):
        return

    @gain.setter
    def gain(self, val):
  
    @property
    def temp(self):
        return 

    @fps.setter
    def temp(self, val):
        self.set_temp(val)
    
    def set_temp(self, val):
    '''
