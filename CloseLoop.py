

from .Cameras import Cred2  # Import camera class
from .wfAffectors import DM  # Import deformable mirror class
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Plotting library (currently unused)
from scipy.ndimage import binary_fill_holes  # For image post-processing (unused here)
from skimage import measure  # For image measurements (unused here)
from plot_functions.plot_func import *  # Custom plotting functions (assumed external)
from skimage.io import imread  # For image loading (unused here)
from skimage.color import rgb2gray  # Convert images to grayscale (unused here)
from OOPAO.Source import Source  # Light source model
from OOPAO.Telescope import Telescope  # Telescope model
from OOPAO.Zernike import Zernike  # Zernike polynomials (unused here)
from OOPAO.ZWFS import ZWFS  # Zernike Wavefront Sensor
from OOPAO.ZWFS2 import ZWFS2  # Second version of ZWFS (unused here)
import time  # Used for timing/waiting
from .tools_zwfs import *  # ZWFS-related tools (assumed external)

from .interaction_matrix import *  # Modal/Zonal interaction matrix tools
from OOPAO.calibration.CalibrationVault import CalibrationVault  # Reconstructor computation


import logging  # For logging messages
import tqdm  # Progress bar

logging.basicConfig(level=logging.INFO)  # Set up logging
logger = logging.getLogger(__name__)  # Create logger


class CloseLoop:
    def __init__(self, dm: DM, cam: Cred2, IM: np.ndarray = None, gain: float = 0.3, iteration: int = 100,
                 inversion_trunc: int = 0, IM_modal: bool = True, ZWFS_tag: int = 1, ZWFS_shift: float = 0.33,
                 ratio_mask_psf: float = 2.2, precropping_data: list = [50, 400, 50, 400], controlled_modes: int = None, 
                 validpixels_crop = 3e2, zpf: int = 30, reconstructor_type: str = 'asin'):
        self.dm = dm  # Assign deformable mirror
        self.cam = cam  # Assign camera
        self.set_flat()  # Reset DM to flat shape
        self._validpixels_crop = validpixels_crop
        # Save configuration parameters to private attributes
        self._IM_modal = IM_modal
        self._gain = gain
        self._iteration = iteration
        self._inversion_trunc = inversion_trunc
        self._ZWFS_tag = ZWFS_tag
        self.precropping_data = precropping_data  # Cropping window for image capture
        self._controlled_modes = controlled_modes
        self._ZWFS_shift = ZWFS_shift
        self._ratio_mask_psf = ratio_mask_psf
        self._zpf = zpf
        image = self.get_image()  # Get initial image
        self.initialise_reference_signal()  # Compute reference signal
        self._reconstructor_type = reconstructor_type
        self._modal_command = IM_modal
        # Set up telescope and source models
        self.tel = Telescope(self.submask.shape[0], 22.5e-3, pupil=self.submask)
        self.tel.pupilReflectivity = np.sqrt(self.pupil)  # Apply pupil reflectivity
        self.src = Source(optBand='J2', magnitude=-2.5)  # Define source
        self.src * self.tel  # Apply source to telescope
        self.update_ZWFS_class()  # Initialize ZWFS object
   
        # Handle interaction matrix if provided
        if IM is not None:
            
            if IM.ndim == 3:
                self._IM_fullframe = IM
                if IM.shape[-2:] != image.shape:
                    raise ValueError("The dimension of the interaction matrix does not correspond to the image")
                IM1 = np.zeros((self.signalref.size, IM.shape[0]))
                for i in range(IM.shape[0]):
                    IM1[:, i] = self.signal_processing_pupils_optimised(IM[i, ...])
                self.IM = IM1.copy()
                self.inversion_IM()
            elif (IM.ndim == 2) and (IM.shape[0] == self.signalref.size):
                self.IM = IM
                self.inversion_IM()
            else:
                logger.warning("The dimension of the interaction matrix must correspond to the signal size or image size, it will not work. You must change the signal mask so that it corresponds to the IM mask or inject the full frame IM")
        else:
            logger.info("No interaction matrix entered, computing one")
            self.IM = self.compute_new_IM()  # Compute default IM
    def _interaction_matrix(self,stroke, dm, cam,nmodes = None, precropping = [20,-20,20,-20], nmeasurement:int = 1):
        if nmodes is None:
            M2C = dm.M2C
            nmodes = M2C.shape[1] #define the nbr of modes
        # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
        
        out = 0
        for _ in range(nmeasurement):
            interactionmatrix=[]
            for i in tqdm.tqdm(range(nmodes)):
                dm.poke_mode(stroke, i)
                time.sleep(0.05)

                image = cam.get()
                image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
                normalisation = np.sum(image)
                image /=normalisation
            
                signal_plus = np.copy(image)   
                #print(f'the shape is {signals_plus[1].shape}')
                dm.poke_mode(-stroke,i)
                time.sleep(0.05)
        
                image = cam.get()
                image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
                image/=image.sum()
                signal_minus = np.copy(image)  
                #print(f'the shape is {signals_minus[1].shape}')
                interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
            dm.set_flat_surf()
            out += np.array(interactionmatrix)
        out /= nmeasurement
        
        # Masque uniquement de cette région
        return out
    def _interaction_matrix_Zonal(self, stroke, dm, cam, nmodes = None, precropping = [20,-20,20,-20], nmeasurement:int = 1):

        if nmodes is None:
            M2C = dm.M2C
            nmodes = M2C.shape[0] #define the nbr of modes
        # nsignals = [submasks[0][submasks[0]].size, submasks[1][submasks[1]].size]
        
        out = 0
        for _ in range(nmeasurement):
            interactionmatrix=[]
            for i in tqdm.tqdm(range(nmodes)):
                vector = np.zeros(nmodes)
                vector[i] = stroke
                dm.poke_vector(vector)
                time.sleep(0.01)

                image = cam.get()
                image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
                normalisation = np.sum(image)
                image /=normalisation
                
                signal_plus = np.copy(image)   
                #print(f'the shape is {signals_plus[1].shape}')
                vector[i] = -stroke
                dm.poke_vector(vector)
                time.sleep(0.01)

                image = cam.get()
                image = image[precropping[0]:precropping[1],precropping[2]:precropping[3]]
                image/=image.sum()
                signal_minus = np.copy(image)  
                #print(f'the shape is {signals_minus[1].shape}')
                interactionmatrix.append(0.5*(signal_plus-signal_minus)/stroke)
            dm.set_flat_surf()
            out += np.array(interactionmatrix)
            out /= nmeasurement

            # Masque uniquement de cette région
            return out
    def compute_new_IM(self, stroke: float = 0.008, nmeasurements: int = 10, cam_nFrames: int = None) -> np.ndarray:
        # Compute interaction matrix either modal or zonal
        self._IM_stroke = stroke
        self._IM_nmeasurement = nmeasurements
        if cam_nFrames is not None:
            self.update_nFrames(np.abs(int(cam_nFrames)))
        self._IM_nframes = self.cam.nFrames
        if self.IM_modal:
            IM = self._interaction_matrix(stroke, self.dm, self.cam, precropping=self.precropping_data, nmeasurement=nmeasurements)
        else:
            IM = self._interaction_matrix_Zonal(stroke, self.dm, self.cam, precropping=self.precropping_data, nmeasurement=nmeasurements)
        self.IM_fullframe = IM
        IM1 = np.zeros((self.signalref.size, IM.shape[0]))
        for i in range(IM.shape[0]):
            IM1[:, i] = self.signal_processing_pupils_optimised(IM[i, ...])
        self.IM = IM1
        self.inversion_IM(IM=IM1)
        return IM1
    def reference_intensities(self,crop=3e2, phase_shift_ZWFS=[-0.6*np.pi,0.3*np.pi], size_psf = 10.45, precropping = [20,-20,20,-20]):

    ######### Off-Mask signal ###########
        self.dm.poke_mode(2,0)
        time.sleep(2)



        off_mask_image = self.cam.get()[precropping[0]:precropping[1],precropping[2]:precropping[3]]
        normalisation = np.sum(off_mask_image)
        off_mask_image/=normalisation

        self.dm.set_flat_surf()
        ######## Cropping Mask to Keep Only the Pupil ###########
        mask = off_mask_image>=crop/normalisation
        labels = measure.label(mask)


        pupilles = []
        submasks = []
        position = []
        for region in measure.regionprops(labels):
            minr, minc, maxr, maxc = region.bbox
            position.append([minr, minc, maxr, maxc])
            sub_image = off_mask_image[minr:maxr, minc:maxc]
            sub_mask = mask[minr:maxr, minc:maxc]
            labeled = measure.label(sub_mask)
            regions = measure.regionprops(labeled)
            largest_region = max(regions, key=lambda r: r.area).label
            pup_clean = labeled == largest_region
            sub_mask = binary_fill_holes(pup_clean)
        # Masque uniquement de cette région
            pup_clean = labeled == largest_region
            sub_image_masked = sub_image * sub_mask#
            pupilles.append(sub_image_masked)
            submasks.append(sub_mask)
        np.array(position)
        pupilles[0], cr= pad_to_square(pupilles[0])#np.pad(pupilles[0], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        position[0]+= np.array(cr)
        pupilles[1], cr= pad_to_square(pupilles[1])#np.pad(pupilles[1], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        position[1]+= np.array(cr)
        submasks[0], _= pad_to_square(submasks[0])#np.pad(submasks[0], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        submasks[1], _= pad_to_square(submasks[1])#np.pad(submasks[1], pad_width=((1, 1), (0, 0)), mode='constant', constant_values=0)
        ######## Cropping Mask to Keep Only the Pupil ###########
        tel1 = Telescope(submasks[0].shape[0], 22.5e-3, pupil=submasks[0])
        tel1.pupilReflectivity = np.sqrt(pupilles[0])
        tel2 = Telescope(submasks[1].shape[0], 22.5e-3, pupil=submasks[1])
        tel2.pupilReflectivity = np.sqrt(pupilles[1])
        src = Source(optBand   = 'J2', magnitude = -2.5)
        src*tel1
        src*tel2
        zwfs1 = ZWFS(tel=tel1,diameter = size_psf, phase_shift = phase_shift_ZWFS[0], zpf = 30, propagation_method='MFT')
        src*tel1*zwfs1
        zwfs2 = ZWFS(tel=tel2,diameter = size_psf, phase_shift = phase_shift_ZWFS[1], zpf = 30, propagation_method='MFT')
        src*tel2*zwfs2
        imref1 = zwfs1.im_ref
        imref2 = zwfs2.im_ref
        signalref1 = zwfs1.ref_signal
        signalref2 = zwfs2.ref_signal
        return [imref1,imref2],[signalref1,signalref2], position, pupilles, submasks
    def initialise_reference_signal(self, crop: float = None):
        if crop is None:
            crop = self.validpixels_crop
        # Capture reference images and extract signals
        _, signalrefs, position, pupils, submasks = self.reference_intensities(crop=crop, precropping=self.precropping_data, phase_shift_ZWFS = [self.ZWFS_shift-np.pi, self.ZWFS_shift], size_psf=self.ratio_mask_psf
        )
        self.pupil = pupils[self.ZWFS_tag]
        self.signalref = signalrefs[self.ZWFS_tag]
        self.submask = submasks[self.ZWFS_tag]
        self.position_signal = position[self.ZWFS_tag]

    def signal_processing_pupils_optimised(self, image: np.ndarray) -> np.ndarray:
        minr, minc, maxr, maxc = self.position_signal
        return image[minr:maxr, minc:maxc][self.submask]  # Extract masked signal

    def inversion_IM(self, IM: np.ndarray = None, filtering: int = None, controlled_modes: int = None) -> None:
        if IM is None:
            IM = self.IM
        if filtering is None:
            filtering = self.inversion_trunc
        if controlled_modes is None:
            controlled_modes = self.controlled_modes
        
        if controlled_modes is not None:
            if controlled_modes > IM.shape[1]:
                logger.warning(f"Requested {controlled_modes} controlled modes, but IM only has {IM.shape[1]}. Using all available modes.")
                controlled_modes = IM.shape[1]
            IM = IM[:, :controlled_modes]  # Truncate IM

        reconstructor_object = CalibrationVault(IM, invert=True, nTrunc=filtering)
        self.reconstructor = reconstructor_object.Mtrunc  # Get reconstructor matrix
        self.eigenmodes_reconstructor = reconstructor_object.U.T  # Eigenmodes
        self.eigenvalues_reconstructor = reconstructor_object.eigenValues  # Eigenvalues

    # [remaining methods omitted for brevity, but would be similarly annotated]

    def get_image(self) -> np.ndarray:
        y0, y1, x0, x1 = self.precropping_data  # Unpack cropping bounds
        image = self.cam.get()[y0:y1, x0:x1]  # Crop image
        image /= image.sum()  # Normalize
        return image

    def get_signal(self) -> np.ndarray:
        image = self.get_image()  # Capture normalized image
        return self.signal_processing_pupils_optimised(image) - self.signalref  # Compute signal deviation

    def apply_dm_command(self, command: np.ndarray, modal_command:bool = None) -> None:
        if modal_command is not None:
           self.modal_command = modal_command
        if self.modal_command:
            self.dm.poke_mode(command, np.arange(command.size))  # Apply modal command
        else:
            self.dm.poke_vector(command)  # Apply zonal command

    def update_flat(self, flat_surf=None):
        # Set new flat DM shape
        self.dm.new_flat(flat_surf if flat_surf is not None else self.dm.current_surf)
    def set_flat(self, flat_surf=None):
        # Set flat DM 
        self.dm.set_flat_surf()
    def update_nFrames(self, nframes):
        # Update number of camera frames for averaging
        self.cam.nFrames = nframes

    def close_loop(self, gain: float = None, iteration: int = None, inversion_trunc: int = None,
                   cam_nFrames: int = None, injected_perturbation: np.ndarray = None,
                   controlled_modes: int = None, leakage: float = 1) -> tuple[list[np.ndarray], list[np.ndarray]]:
        gain = gain if gain is not None else self.gain  # Use default gain if none provided
        iteration = iteration if iteration is not None else self.iteration  # Use default iterations
    
        # Update reconstruction if new parameters provided
        if inversion_trunc is not None:
            self.inversion_IM(filtering=inversion_trunc)
        if cam_nFrames is not None:
            self.update_nFrames(cam_nFrames)
        if controlled_modes is not None and controlled_modes != self.controlled_modes:
            self.controlled_modes = controlled_modes
            self.inversion_IM()

        # Ensure leakage value is valid
        if leakage < 0 or leakage > 1:
            logger.warning("Leakage must be between 0 and 1. Setting it to 1.")
            leakage = 1
        self._leakage = leakage
        self._close_loop_nFrames= self.cam.nFrames

        self.set_flat()  # Start with flat DM
        time.sleep(0.01)  # Allow DM to settle

        command = np.zeros(self.IM.shape[1])  # Initialize DM command vector

        # Handle perturbation input
        if injected_perturbation is None:
            injected_perturbation = 0
        else:
            if injected_perturbation.shape[0] != command.shape[0]:
                raise ValueError("Injected perturbation shape mismatch")
            if injected_perturbation.ndim == 2 and injected_perturbation.shape[1] != iteration:
                raise ValueError("Injected perturbation iteration mismatch")
        self._injected_perturbation = injected_perturbation
        history_cmd = [command.copy()]  # Store command history
        history_signal = []  # Store signal history

        for _ in tqdm.tqdm(range(iteration)):
            signal = self.get_signal()  # Get current signal
            rec1 = np.matmul(self.reconstructor, signal)  # Compute correction
            command[:rec1.shape[0]] = leakage * command[:rec1.shape[0]] - gain * rec1  # Update command

            # Safety check on DM limits
            if np.max(self.dm.current_surf) > self.dm.maxCmd or np.min(self.dm.current_surf) < self.dm.minCmd:
                logger.warning("Diverging: closing everything")
                break

            self.apply_dm_command(command + injected_perturbation)  # Apply DM command
            time.sleep(0.01)  # Allow DM to settle

            history_cmd.append(command.copy())  # Log command
            history_signal.append(signal.copy())  # Log signal
        self._history_cmd = np.array(history_cmd)  # Set as private attribute
        self._history_signal = np.array(history_signal)  # Set as private attribute
        return history_cmd, history_signal  # Return history


    def reconstruct_phase(self, iteration: int = 10, nFrames: int = None, project: bool = False, nmodes_projection: int = None, reconstruct:str=None):
        if reconstruct is not None:
            self.reconstructor_type = reconstruct
        self.update_ZWFS_class()  # Update the ZWFS class (resets internal state)
        if nFrames is not None:
            self.update_nFrames(nFrames)  # Update number of frames if specified
        signal = self.get_signal()  # Get current signal
        signal2D = np.zeros(self.submask.shape)  # Prepare 2D signal image
        signal2D[self.submask] = signal + self.signalref  # Combine signal and reference
        self.zwfs.img_ZWFS = signal2D  # Assign to ZWFS object
        phase = self.zwfs.reconstructor(reconstr=self.reconstructor_type, iteration=iteration)  # Reconstruct phase with arcsin method
        phase[self.submask] -= phase[self.submask].mean()  # Remove piston (average phase)
        self._reconstructed_phase = phase
        if project:
            return phase, self.project_phase(phase, nmodes_projection)
        else:
            return phase

    def project_phase(self, phase, nmodes=None):
        if nmodes is None:
            nmodes = self.IM.shape[1]  # Default to number of modes in IM
        z = Zernike(self.tel, nmodes)  # Create Zernike object
        z.computeZernike(self.tel, remove_piston=1)  # Generate Zernike modes
        modes = z.modesFullRes  # Extract full-resolution Zernike modes
        modes = modes.reshape((self.tel.resolution**2, modes.shape[-1]))  # Flatten for projection
        cov_modes = modes.T @ modes  # Compute mode covariance
        proj_Zern = np.diag(1 / np.diag(cov_modes)) @ modes.T  # Pseudo-inverse projection matrix
        projected_phase = proj_Zern @ phase.reshape(self.pupil.shape[0]**2)  # Project phase onto modes
        return projected_phase
    def update_ZWFS_class(self):
        # Recreate the ZWFS object after parameter update
        self.zwfs = ZWFS(tel=self.tel, diameter=self.ratio_mask_psf,
                         phase_shift=self.ZWFS_shift * np.pi, zpf=self.zpf, propagation_method='MFT')
        self.signalref = self.zwfs.ref_signal
    # Properties for controlled access and automatic update
    @property
    def zpf(self) -> int:
        return self._zpf

    @zpf.setter
    def zpf(self, val: int) -> None:
        if not isinstance(val, int) or val < 1:
            raise ValueError("zpf must be a positive integer.")
        self._zpf = val
        self.update_ZWFS_class() 
    @property
    def IM_fullframe(self) -> int:
        return getattr(self, "_IM_fullframe", None)
    @IM_fullframe.setter
    def IM_fullframe(self, val: float) -> None:
        self._IM_fullframe = val
    @property
    def leakage(self) -> int:
        return getattr(self, "_leakage", None)
    @property
    def close_loop_nFrames(self) -> float:
        return getattr(self, "_close_loop_nFrames", None)
    @property
    def injected_perturbation(self) -> float:
        return getattr(self, "_injected_perturbation", None)
    @property
    def IM_nframes(self) -> int:
        return getattr(self, "_IM_nframes", None)
    @property
    def IM_stroke(self) -> float:
        return getattr(self, "_IM_stroke", None)
    @property
    def validpixels_crop(self) -> float:
        return self._validpixels_crop
    @validpixels_crop.setter
    def validpixels_crop(self, val: float) -> None:
        self._validpixels_crop = val
        self.initialise_reference_signal()
    @property
    def IM_nmeasurement(self) -> int:
        return getattr(self, "_IM_nmeasurement", None)
    @property
    def reconstructed_phase(self) -> np.ndarray:
        return self._reconstructed_phase
    @property
    def history_cmd(self) -> np.ndarray:
        return self._history_cmd

    @property
    def history_signal(self) -> np.ndarray:
        return self._history_signal

    @property
    def ZWFS_shift(self) -> float:
        return self._ZWFS_shift

    @ZWFS_shift.setter
    def ZWFS_shift(self, val: float) -> None:
        self._ZWFS_shift = val
        self.update_ZWFS_class()

    @property
    def ratio_mask_psf(self) -> float:
        return self._ratio_mask_psf

    @ratio_mask_psf.setter
    def ratio_mask_psf(self, val: float) -> None:
        self._ratio_mask_psf = val
        self.update_ZWFS_class()

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, val: float) -> None:
        if not 0 <= val <= 1:
            raise ValueError("Gain must be in [0, 1]")
        self._gain = val

    @property
    def IM_modal(self) -> bool:
        return self._IM_modal

    @IM_modal.setter
    def IM_modal(self, val: bool) -> None:
        if not isinstance(val, bool):
            raise ValueError("IM_modal must be a boolean")
        self._IM_modal = val

    @property
    def signalref(self) -> np.ndarray:
        return self._signalref

    @signalref.setter
    def signalref(self, val: np.ndarray) -> None:
        self._signalref = val

    @property
    def ZWFS_tag(self) -> int:
        return self._ZWFS_tag

    @ZWFS_tag.setter
    def ZWFS_tag(self, val: int) -> None:
        if val not in [0, 1]:
            raise ValueError("ZWFS_tag must be 0 or 1")
        self._ZWFS_tag = val

    @property
    def IM(self) -> np.ndarray:
        return self._IM

    @IM.setter
    def IM(self, val: np.ndarray) -> None:
        self._IM = val

    @property
    def iteration(self) -> int:
        return self._iteration

    @iteration.setter
    def iteration(self, val: int) -> None:
        if val < 1:
            raise ValueError("Iteration must be >= 1")
        self._iteration = int(val)

    @property
    def inversion_trunc(self) -> int:
        return self._inversion_trunc

    @inversion_trunc.setter
    def inversion_trunc(self, val: int) -> None:
        if val < 0:
            raise ValueError("Inversion truncation must be >= 0")
        self._inversion_trunc = int(val)

    @property
    def controlled_modes(self) -> int:
        return self._controlled_modes

    @controlled_modes.setter
    def controlled_modes(self, val: int) -> None:
        if val is not None:
            if val < 0:
                raise ValueError("Controlled modes must be >= 0")
        self._controlled_modes = int(val)

    @property
    def reconstructor_type(self):
        return self._reconstructor_type

    @reconstructor_type.setter
    def reconstructor_type(self, val):
        self._reconstructor_type = val

    @property
    def modal_command(self):
        return self._modal_command

    @modal_command.setter
    def modal_command(self, val):
        self._modal_command = val