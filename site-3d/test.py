import os
import shutil
from pathlib import Path

import numpy as np
import tifffile as tiff
import SimpleITK as sitk

from siteflow.dataset.data_obj import CellTable, TrajTable
from utils.util import get_global_transform, nrrd_to_tif, tif_to_nrrd


EPS = 1e-12

# human label
# root_dir = Path('/home/xiong/Desktop/wan_label/train')
# raw_dir = root_dir / 'tiff'
# tgt_dir = root_dir / 'nrrd'

# for path in raw_dir.rglob('*.tif'):
#     file_name = path.name
#     differ_path = path.relative_to(raw_dir)  # yangye/.tif

#     tgt_path = tgt_dir / differ_path
#     tgt_path = tgt_path.with_suffix('.nrrd')
#     tgt_path.parent.mkdir(parents=True, exist_ok=True)

#     tif_to_nrrd(path, tgt_path)

# det
# cell_idx_person = {}
# ref_dir = Path('/home/xiong/Desktop/site_test_data/test_stack_xjh/human/tif')

# src_dir = Path('/home/xiong/Desktop/site_test_data/test_stack/data')
# tgt_dir = Path('/home/xiong/Desktop/site_test_data/test_stack_xjh/detect/nrrd')

# for path in ref_dir.rglob('*.tif'):
#     if 'mask' in str(path):
#         continue
#     person = path.parts[-2]
#     cell_idx = path.stem

#     raw_tgt_path = tgt_dir / person / (cell_idx + '.nrrd')
#     mask_tgt_path = tgt_dir / person / (cell_idx + '_mask.nrrd')
#     raw_tgt_path.parent.mkdir(parents=True, exist_ok=True)
#     mask_tgt_path.parent.mkdir(parents=True, exist_ok=True)

#     raw_mask_stack_path = src_dir / cell_idx / 'imgs_raw_mask.tif'
#     raw_mask_stack = tiff.imread(raw_mask_stack_path)
#     raw_stack = raw_mask_stack[0]
#     mask_stack = raw_mask_stack[1]

#     sitk.WriteImage(sitk.GetImageFromArray(raw_stack), raw_tgt_path)
#     sitk.WriteImage(sitk.GetImageFromArray(mask_stack), mask_tgt_path)

# root = Path('/home/xiong/Desktop/site_test_data/xjh_test_data')
# nrrd_dir = root / 'nrrd'
# tif_dir = root / 'tif'

# for nrrd_path in nrrd_dir.rglob('*.nrrd'):
#     cell_idx = nrrd_path.stem
#     person = nrrd_path.parts[-2]

#     cell_idx_tif = cell_idx + '.tif'
#     tif_path = tif_dir / person / cell_idx_tif
#     tif_path.parent.mkdir(parents=True, exist_ok=True)

#     nrrd_to_tif(nrrd_path, tif_path)


# filter to null
# root_dir = Path('/home/xiong/Desktop/wan_label/label_tiff_32bit_stack_val')
# for path in root_dir.rglob('*filter.tif'):
#     new_path = Path(str(path).replace('_filter', ''))
#     path.replace(new_path)

# ====
# test
# ====

# cell_path = Path('/home/xiong/Desktop/site_test_data/test_stack/data/cellraw_3167')

# cell_res = CellTable(str(cell_path))
# num_frames = cell_res.num_frames

# transform_file = cell_path / 'rigid_transforms_series.pkl'
# transform_list = get_global_transform(str(transform_file), num_frames)

# raw_det_mask = cell_res.get_raw_det_mask(transform_list).astype(np.float32)
# tgt_path = cell_path / '0_det_raw_mask.tif'
# tiff.imwrite(str(tgt_path), raw_det_mask, imagej=True)


# ==========
# evaluation
# ==========

root = '/home/xiong/Desktop/wan_label/test/det_data/f1/1'

# root = Path(root)
# for path in root.iterdir():
#     if path.is_dir():
#         cell_idx = path.parts[-1]
#         mask = cell_idx + '_mask.tif'
        
#         src = root / mask
#         tgt = root / cell_idx / mask

#         shutil.copy(src, tgt)

tp_dict = {}; fp_dict = {}; fn_dict = {}

remove = ['cellraw_15743', 'cellraw_15757', 'cellraw_32566', 'cellraw_35335', 'cellraw_36988']
remove = []

for dir in os.listdir(root):
    root_dir = os.path.join(root, dir)

    if os.path.isfile(root_dir) or dir in remove:
        continue

    # raw_stack = tiff.imread(os.path.join(root_dir, 'imgs_raw_mask.tif'))[0, ...]
    # reg_stack = tiff.imread(os.path.join(root_dir, 'imgs_raw_mask_reg_rcs.tif'))[0, ...]

    # # traj_data
    # traj_path = os.path.join(root_dir, 'dataAnalysis_tj_0_withBg.csv')
    # if os.path.exists(traj_path):
    #     traj_res = TrajTable(traj_path)

    #     site_index, site_frame = traj_res.get_site_frame()
    #     sudo_stack = traj_res.get_sudo_stack()
    #     intensity = traj_res.get_intensity()

    #     traj_tgt_path = os.path.join(root_dir, '0_traj_0_raw.tif')
    #     # traj_stack = np.stack((raw_stack, sudo_stack.astype(np.uint16)))
    #     # tiff.imwrite(traj_tgt_path, traj_stack, imagej=True)

    #     complete_traj_path = os.path.join(root_dir, '0_complete_traj.tif')
    #     complete_stack_mask = traj_res.get_sudo_stack(only_tp=False)
    #     complete_stack = np.stack((raw_stack, complete_stack_mask.astype(np.uint16)))

    #     tiff.imwrite(complete_traj_path, complete_stack, imagej=True)

    # # cell_res
    # cell_path = root_dir
    # cell_res = CellTable(cell_path)

    # det_site = cell_res.get_mask_from_coord('traj')

    # traj_tgt_path = os.path.join(root_dir, '0_det_site_reg.tif')
    # det_stack = np.stack((reg_stack, det_site.astype(np.uint16)))
    # tiff.imwrite(traj_tgt_path, det_stack, imagej=True)



    # ========
    # analysis
    # ========
    cell_path = root_dir
    cell_res = CellTable(cell_path)

    tp, fn, fp = cell_res.evaluate('traj')
    tp_dict[cell_res.cell_idx] = tp
    fn_dict[cell_res.cell_idx] = fn
    fp_dict[cell_res.cell_idx] = fp
    
    # print(cell_res.cell_idx, tp, fn, fp)

tp = sum(tp_dict.values())
fn = sum(fn_dict.values())
fp = sum(fp_dict.values())

recall = tp / (tp + fn + EPS)
precision = tp / (tp + fp + EPS)
f1_value = (2 * precision * recall) / (precision + recall + EPS)

print(f'recall: {recall:.2f}, precision: {precision:.2f}, f1: {f1_value:.2f}')


# fn_dict = dict(sorted(fn_dict.items(), key=lambda x: x[1], reverse=True))
# print('print(fn_dict)')
# print(fn_dict)

# fp_dict = dict(sorted(fp_dict.items(), key=lambda x: x[1], reverse=True))
# print('print(fp_dict)')
# print(fp_dict)
