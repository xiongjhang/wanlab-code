''''''''

# __all__ = [CellTable, TrajTable]

from typing import Tuple, Union, Literal, Dict
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import scipy.optimize

from utils.util import coordinate_to_mask, mask_to_coordinate, linear_sum_assignment, coord_reg_to_raw


def basename(path: Union[str, "os.PathLike[str]"]) -> str:
    """Returns the basename removing path and extension."""
    return os.path.splitext(os.path.basename(path))[0]


class CellTable:
    '''Data and operator for statistic data for all trajectory result'''

    EXTENSIONS = ('csv')
    COLUMNS = [
        'x', 'y', 'frame', 'particle'
    ]

    def __init__(
            self,
            path: str,
        ):
        # prop
        assert os.path.isdir(path), 'Check your input path'
        self.path = path
        self.cell_idx = os.path.basename(os.path.normpath(self.path))

        raw_mask_stack = tiff.imread(os.path.join(self.path, 'imgs_raw_mask.tif'))
        raw_mask_reg_stack = tiff.imread(os.path.join(self.path, 'imgs_raw_mask_reg_rcs.tif'))
        self.det_data_raw_mask = raw_mask_stack[1]

        self.num_frames = raw_mask_stack.shape[1]

        self.type = ['traj', 'patch']
        traj_data_path = os.path.join(self.path, 'trajectories_data.csv')
        patch_data_path = os.path.join(self.path, 'trajectories_data_raw.csv')
        self.traj_data = self._prepare(traj_data_path) if os.path.exists(traj_data_path) else None
        self.patch_data = self._prepare(patch_data_path) if os.path.exists(patch_data_path) else None

        self.label = os.path.join(self.path, self.cell_idx+'_mask.tif')
        self.traj = os.path.join(self.path, '0_traj_0_raw.tif')
        self.complete_traj = os.path.join(self.path, '0_complete_traj.tif')

        det_path = os.path.join(self.path, 'cell_mask_reg.csv')
        self.det_data_reg = self._prepare(det_path) if os.path.exists(det_path) else None

    def _prepare(self, data_path: str) -> pd.DataFrame:
        '''Read csv from given path'''
        return pd.read_csv(data_path, index_col=False, na_values='NAN')
    
    def get_mask_from_coord(
            self,
            type: Literal['traj', 'patch'] = 'patch',
            img_size: int = 128,
        ):
        '''Turning `trajectories_data.csv` or `trajectories_data_raw.csv` (reg) into mask array (reg)'''
        video_mask = np.zeros((self.num_frames, img_size, img_size))
        site_data = self.patch_data if type == 'patch' else self.traj_data
        grouped_data = site_data.groupby('frame')

        for frame, group in grouped_data:
            if 0 <= frame and frame < self.num_frames:
                coords = group[['y', 'x']].to_numpy().astype(int)

                video_mask[frame] += coordinate_to_mask(coords, img_size, n=1)

        return video_mask
    
    def get_raw_det_mask(self, transform_list, img_size: int = 128):
        '''Turning `cell_mask_reg.csv`(reg) into mask array (raw)'''

        grouped_data = self.det_data_reg.groupby('frame')

        video_mask = np.zeros((self.num_frames, img_size, img_size))
        for frame, group in grouped_data:
            if 0 <= frame and frame < self.num_frames:
                coords = group[['y', 'x']].to_numpy()

                if frame == 0:
                    video_mask[frame] += coordinate_to_mask(coords.astype(int), img_size, n=1)
                else:
                    transform = transform_list[frame - 1]
                    for i, (reg_y, reg_x) in enumerate(coords):
                        org_y, org_x = coord_reg_to_raw(reg_y, reg_x, transform)
                        coords[i, 0], coords[i, 1] = org_y, org_x
                    video_mask[frame] += coordinate_to_mask(coords.astype(int), img_size, n=1)

        return video_mask

    
    def evaluate(self, opt: Literal['det', 'track', 'traj'] = 'track'):
        if not(os.path.exists(self.label)):
            raise ValueError(f'{self.label} is not existing!')
        
        label_stack = tiff.imread(self.label)
        num_frame = label_stack.shape[0]
        if opt == 'track':
            if os.path.exists(self.traj):
                det_stack = tiff.imread(self.traj)[1]
            else:
                det_stack = np.zeros((num_frame, 128, 128))
        elif opt == 'det':
            det_stack = self.det_data_raw_mask
        else:
            if os.path.exists(self.complete_traj):
                det_stack = tiff.imread(self.complete_traj)[1]
            else:
                det_stack = np.zeros((num_frame, 128, 128))

        label_stack = np.clip(label_stack, 0, 1)
        det_stack = np.clip(det_stack, 0, 1)

        assert label_stack.shape == det_stack.shape
        cutoff = 3

        tp_list = []; fn_list = []; fp_list = []
        for i in range(num_frame):
            true = mask_to_coordinate(label_stack[i])
            pred = mask_to_coordinate(det_stack[i])

            if len(true) == 0:
                tp = 0; fn = 0; fp = len(pred)
            elif len(pred) == 0:
                tp = 0; fn = len(true); fp = 0
            else:
                matrix = scipy.spatial.distance.cdist(pred, true, metric="euclidean")
                pred_true_r, _ = linear_sum_assignment(matrix, cutoff)
                true_pred_r, true_pred_c = linear_sum_assignment(matrix.T, cutoff)

                # Calculation of tp/fn/fp based on number of assignments
                tp = len(true_pred_r)
                fn = len(true) - len(true_pred_r)
                fp = len(pred) - len(pred_true_r)

            tp_list.append(tp); fn_list.append(fn); fp_list.append(fp)

        tp = sum(tp_list); fn = sum(fn_list); fp = sum(fp_list)

        return tp, fn, fp


class TrajTable:
    '''Data and operator for statistic data correspongding to a trajectory of single cell seqences'''


    EXTENSIONS = ('csv')
    COLUMNS = [
        'particle_index', 'POSITION_T', 'Reg_X', 'Reg_Y', 'Org_X', 'Org_Y',
        'local_maxima', 'Fit_X', 'Fit_Y', 'Fit_amp', 'Fit_offset',
        'photon_number', 'TP_Flag'
    ]

    def __init__(
            self, 
            path: str,
            extensions: Tuple[str, ...] = EXTENSIONS,
        ):
        # prop
        assert os.path.isfile(path) and path.endswith(extensions), f'Check your input path: {path}'
        self.path = path
        self.folder_path = os.path.dirname(path)
        self.file_name = basename(path)
        
        match = re.search(r'dataAnalysis_tj_([a-zA-Z0-9]+)_withBg', self.file_name)
        if match:
            item: str = match.group(1)
            self.traj_id = int(item) if item != 'empty' else None

        # data
        self.cell_data = self._prepare()
        self.column_list = list(self.cell_data)

    def _prepare(self) -> pd.DataFrame:
        '''Read csv from given path'''
        return pd.read_csv(self.path, index_col=False, na_values='NAN')

    def get_frame_num(self) -> int:
        '''Return total frame of the cell sequence'''
        return len(self.cell_data)

    def get_site_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Return positive/site frame of the cell sequence'''
        assert 'TP_Flag' in self.column_list    
        site_index = self.cell_data['TP_Flag'].values
        site_index[np.isnan(site_index)] = 0 
        site_index = site_index.astype(int)
        site_frame = np.where(site_index == 1)
        
        return site_index, site_frame[0]

    def get_coordinate(self) -> Dict[str, np.ndarray]:
        '''Return kinds of coordinate'''
        coord_list = ['Org_X', 'Org_Y', 'Reg_X', 'Reg_Y', 'Fit_X', 'Fit_Y']
        coord_dict = {
            coord: self.cell_data[coord].values for coord in coord_list
        }

        return coord_dict
    
    def get_sudo_stack(self, type: Literal['raw', 'reg'] = 'raw', only_tp: bool = True) -> np.ndarray:
        '''Return binary mask from statistic data result'''
        coord_dict = self.get_coordinate()
        if type == 'raw':
            x_coord, y_coord = coord_dict['Org_X'], coord_dict['Org_Y']
        elif type == 'reg':
            x_coord, y_coord = coord_dict['Reg_X'], coord_dict['Reg_Y']
        
        site_index, site_frame = self.get_site_frame()
        
        sudo_stack = []
        for idx, (y, x) in enumerate(zip(x_coord, y_coord)):
            if only_tp:
                if idx in list(site_frame):
                    coords = np.array([[np.round(x), np.round(y)]], dtype=np.int16)
                    mask = coordinate_to_mask(coords, image_size=128)
                    sudo_stack.append(mask)
                else:
                    sudo_stack.append(np.zeros((128, 128)))
            else:
                coords = np.array([[np.round(x), np.round(y)]], dtype=np.int16)
                mask = coordinate_to_mask(coords, image_size=128)
                sudo_stack.append(mask)

        return np.array(sudo_stack)

    # def get_complete_traj(self):
    #     coord_dict = self.get_coordinate()
    #     x_coord, y_coord = coord_dict['Org_X'], coord_dict['Org_Y']

    #     sudo_stack = []
    #     for idx, (y, x) in enumerate(zip(x_coord, y_coord)):

    
    def get_intensity(self) -> np.ndarray:
        '''Return intensity changes over time of single site trajectory'''
        return self.cell_data['photon_number'].values