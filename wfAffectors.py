import dao
import numpy as np
import os
import astropy.io.fits as fits
import time
import scipy as scp

class DM:
	def __init__(self,simu=1 ,model='ALPAO97'):
		""" Initialize the DM class.
		:param model: DM model, currently only 'ALPAO' is supported
		"""
		self.model = model
		#self.simulated_dm = Papytwin.dm
		if model == 'ALPAO241':
			nAct = 17 # 17x17 actuators
			nAct_radius = 8.7 # correspond to 241 valid actuators
			maxCmd = 1
			minCmd = -1
			self.simu = simu
			# All other shared memories to 0
			if self.simu == 0:
				self.dmShm = dao.shm('/tmp/dmCmd00.im.shm') # shared memory for DM commands
				dmShmOffset = dao.shm('/tmp/dmCmdOffset.im.shm') 
				dmShm1 = dao.shm('/tmp/dmCmd01.im.shm') 
				dmShm2 = dao.shm('/tmp/dmCmd02.im.shm') 
				dmShm3 = dao.shm('/tmp/dmCmd03.im.shm') 
				dmShmOffset.set_data(dmShmOffset.get_data()*0)
				dmShm1.set_data(dmShmOffset.get_data()*0)
				dmShm2.set_data(dmShmOffset.get_data()*0)
				dmShm3.set_data(dmShmOffset.get_data()*0)
			# --------- Modal Basis ---------
			self.M2C = fits.getdata(os.getenv('DAODATA')+'/inputs/Papyrus/M2C.fits').astype(np.float32)
			# --------- Flat ---------
			flat = self.map_cmd(np.squeeze(fits.getdata(os.getenv('DAODATA')+'/inputs/Papyrus/flat_dm.fits').astype(np.float32)))
		elif model == 'ALPAO97':
			nAct = 11 # 11x11 actuators
			nAct_radius = 5.5 # correspond to 97 valid actuators
			maxCmd = 1
			minCmd = -1
			self.simu = simu
			# All other shared memories to 0
			if self.simu == 0:
				self.dmShm = dao.shm('/tmp/dmCmd.im.shm',np.zeros((97,1)).astype(np.float32)) # shared memory for DM commands
				time.sleep(1)
				# Launch DM
				os.system("tmux new-session -d -s dmloop")
				os.system("tmux send-keys -t dmloop 'daodmAlpaoCtrl -L BAX418 /tmp/dmCmd.im.shm' C-m")
			# --------- Modal Basis ---------
			self.M2C = scp.io.loadmat('/home/manip/dao/daoZernike/OZIRIIS/BAX418-Z2C.mat')['Z2C'].T
			self.IF = scp.io.loadmat('/home/manip/dao/daoZernike/OZIRIIS/BAX418-IF.mat')['influenceMatrix'].T
			self.IF_volume = scp.io.loadmat('/home/manip/dao/daoZernike/OZIRIIS/IF_volume.mat')['IF_volume'].T
			# --------- Flat ---------
			flat = np.squeeze(scp.io.loadmat('/home/manip/dao/daoZernike/OZIRIIS/DM_flat_command.mat')['flat'])

		else:
			raise ValueError('This DM model does not exist')
		# -------- Valid Actuator grid -----
		self.nAct = nAct
		grid=np.mgrid[0:self.nAct,0:self.nAct]
		rgrid=np.sqrt((grid[0]-self.nAct/2+0.5)**2+(grid[1]-self.nAct/2+0.5)**2)
		self.valid_actuators_map = np.zeros((self.nAct,self.nAct)).astype(bool)
		self.valid_actuators_map[np.where(rgrid<nAct_radius)]=1
		# --------- Max/Mion Values ---------
		self.minCmd = minCmd # clipping at 3 microns - maximum hardware value is 3.4 microns
		self.maxCmd = maxCmd
		# ---- Valid acuator with referencing number map ------
		self.valid_actuators_number = np.copy(self.valid_actuators_map)
		self.valid_actuators_number[self.valid_actuators_map==1] = np.linspace(1,int(np.sum(self.valid_actuators_map)),int(np.sum(self.valid_actuators_map)))
		# ----- Valid acutators position -----------
		X = np.round(np.linspace(0,self.nAct-1,self.nAct))
		[xx_dm,yy_dm] = np.meshgrid(X,X)
		self.xx_dm = xx_dm[self.valid_actuators_map==1]
		self.yy_dm = yy_dm[self.valid_actuators_map==1]
		# --------- Load Flat surf ----	
		self.flat_surf = self.map_cmd(flat)
		self.current_surf = self.flat_surf.copy()
		self.set_flat_surf()
	
	def poke_mode(self,amplitude,modeNumber,bias=None):
		""" 
		poke mode : amplitude in PtV in DM units
		"""
		if np.size(amplitude) == 1: # Case with just one mode mode
			amplitude = np.array([amplitude])
			modeNumber = np.array([modeNumber])
		if bias is None:
			bias = np.copy(self.flat_surf)
		dm_shape = np.copy(bias)
		for k in range(0,np.size(amplitude)):
				zer_cmd = amplitude[k]*self.M2C[:,modeNumber[k]]
				zer_map = self.map_cmd(zer_cmd)
				dm_shape = dm_shape + zer_map
		# --- Send commands ---
		self.set_surf(dm_shape)
		return dm_shape
		
	def poke_act(self,amplitude,position,bias=None):
		""" Poke actuator: take its position in X and Y """
		if bias is None:
			bias = np.copy(self.flat_surf)
		dm_shape = np.copy(bias)
		if len(position) == 2:
			dm_shape[position[0]][position[1]] = dm_shape[position[0]][position[1]] + amplitude
		else:
			print('please enter valid position in x and y according to obj.valid_actuators_map')
		# --- Send commands ---
		self.set_surf(dm_shape)
		return dm_shape
		
	def poke_all_act(self,amplitude = 1,timestop = 0.1,bias=None):
		""" Loop through poking all actuators """
		if bias is None:
			bias = np.copy(self.flat_surf)
		# ---- Poking all actuators ----
		for i in range(0,self.nAct):
			for j in range(0,self.nAct):
				if self.valid_actuators_map[i,j] == 1:
					self.poke_act(amplitude,[i,j],bias)
					time.sleep(timestop)
		self.set_surf(bias) # return to initial position
	
	def poke_vector(self,vector,bias=None):
		""" Poke actuator: take its position in X and Y """
		if bias is None:
			bias = np.copy(self.flat_surf)
		dm_shape = np.copy(bias)
		dm_shape += self.map_cmd(vector)
		# --- Send commands ---
		self.set_surf(dm_shape)
		return dm_shape
		

	def poke_waffle(self,amplitude,bias=None):
		''' Poking Waffle patern - Amplitude in PtV '''
		if bias is None:
			bias = np.copy(self.flat_surf)
		# Create waffle patern
		a = -np.ones((self.nAct,self.nAct)).astype(np.float32)
		a[::2,:] = 1
		a[:,::2] = 1
		b = -np.ones((self.nAct,self.nAct)).astype(np.float32)
		b[1::2,:] = 1
		b[:,1::2] = 1
		waffle = bias+amplitude/2*a*b
		self.set_surf(waffle)
		return waffle

	def poke_waffle_large(self,amplitude,bias=None):
			""" Poking Large Waffle patern - Amplitude in PtV """
			if bias is None:
				bias = np.copy(self.flat_surf)
			# Create waffle patern
			a = -np.zeros((self.nAct,self.nAct)).astype(np.float32)
			for i in range(0,self.nAct-2):
				for j in range(0,self.nAct):
					if np.mod(i,4)==0 and np.mod(j,2)==0:
						a[i,j] = 1
					if np.mod(i,4)==0 and np.mod(j,2)==1:
						a[i+2,j] = 1
			waffle = bias+amplitude*a
			self.set_surf(waffle)
			return waffle
	
	def cross(self,amplitude,bias = None):
		if bias is None:
			bias = np.copy(self.flat_surf)
		cmd_cross = np.zeros((self.nAct,self.nAct))
		cmd_cross[int((self.nAct-1)/2),:] = amplitude
		cmd_cross[:,int((self.nAct-1)/2)] = amplitude
		self.set_surf(cmd_cross+bias)
		return dm_shape
  
	def stripes_x(self,amplitude,bias=None):
		if bias is None:
			bias = np.copy(self.flat_surf)
		C = np.zeros((self.nAct,self.nAct))
		# Build cross
		C[1::2,:] = amplitude
		self.set_surf(C+bias)
		return dm_shape
		
	def stripes_y(self,amplitude,bias=None):
		if bias is None:
			bias = np.copy(self.flat_surf)
		C = np.zeros((self.nAct,self.nAct))
		# Build cross
		C[:,1::2] = amplitude
		self.set_surf(C+bias)
		return dm_shape

	def letter_L(self,amplitude,bias=None):
		if bias is None:
			bias = np.copy(self.flat_surf)
		C = np.zeros((self.nAct,self.nAct))
		# Build L
		C[int(self.nAct/5):int(4*self.nAct/5),int(self.nAct/3)] = amplitude
		C[int(self.nAct/5),int(self.nAct/3):2*int(self.nAct/3)] = amplitude
		self.set_surf(C+bias)
		return dm_shape

	def set_flat_surf(self):
		""" Apply flat """
		self.set_surf(self.flat_surf)
		print('Flat surface applied')

	def zero_all(self):
		self.set_surf(0*self.flat_surf)
		print('Zeroing all actuators')
  
	def new_flat(self,dm_map):
		""" Change flat and apply it """
		self.flat_surf = dm_map
		print('Flat surface updated')
		self.set_flat_surf()
		return dm_map

	def get_surf(self):
		""" Check if current surf is consistant with SHM and return it"""
		if self.simu == 0:
			if np.squeeze(self.dmShm.get_data()) != self.current_surf:
				raise ValueError('current surf is not the same shared memory one')
			else:
				return self.current_surf
		else:
			return self.current_surf
		
	def map_cmd(self,cmd_vector):
		""" map command vector to surface command """
		c_map = np.zeros((self.nAct,self.nAct)).astype(np.float32)
		c_map[self.valid_actuators_map==1] = cmd_vector
		return c_map
	def cmd_map(self,c_map):
		cmd = np.zeros(np.where(self.valid_actuators_map)[0].shape)
		cmd = c_map[np.where(self.valid_actuators_map)]
		return cmd

	def set_surf(self,cmd_map):
		""" Apply DM map command """
		# Clipping
		if np.max(cmd_map)>self.maxCmd or np.min(cmd_map)<self.minCmd:
			print("CLIPPING COMMANDS")
			cmd_map = np.clip(cmd_map,self.minCmd,self.maxCmd)
		# Set the current surface
		if self.simu == 0:
			self.dmShm.set_data(np.expand_dims(cmd_map[self.valid_actuators_map==1],axis = 1).astype(np.float32))
		# Simulation
		#self.simulated_dm.coefs = cmd_map[self.valid_actuators_map==1]
		# Update current flat
		self.current_surf = cmd_map
		return cmd_map
	
	def zeroAll(self):
		if self.simu == 0:
			self.dmShm.set_data(self.dmShm.get_data()*0)# shared memory for DM commands
			if self.model == 'ALPAO241':
				# All other shared memories to 0
				dmShmOffset = dao.shm('/tmp/dmCmdOffset.im.shm') 
				dmShm1 = dao.shm('/tmp/dmCmd01.im.shm') 
				dmShm2 = dao.shm('/tmp/dmCmd02.im.shm') 
				dmShm3 = dao.shm('/tmp/dmCmd03.im.shm') 
				dmShmOffset.set_data(dmShmOffset.get_data()*0)
				dmShm1.set_data(dmShmOffset.get_data()*0)
				dmShm2.set_data(dmShmOffset.get_data()*0)
				dmShm3.set_data(dmShmOffset.get_data()*0)

	def shutdown(self):
		#self.simulated_dm.coefs *= 0
		self.zeroAll()
		if self.model == 'ALPAO97':
			os.system("tmux kill-session -t dmloop")
