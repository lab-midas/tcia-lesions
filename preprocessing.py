import pathlib as plb
import shutil
import tempfile 

import dicom2nifti
import nibabel as nib
import nilearn
import numpy as np
import pydicom

from misc import reorient_nifti, convert_pet, calculate_suv_factor


def resample_nifti(ct, suv, target_orientation, target_spacing):
    # TODO move file access to main routine
    # load nifti data
    ct, suv = plb.Path(ct), plb.Path(suv)
    ct, suv = nib.load(ct), nib.load(suv)

    # reorient niftis
    ct = reorient_nifti(ct, target_orientation=target_orientation)
    suv = reorient_nifti(suv, target_orientation=target_orientation)

    # resample and align pet/ct
    orig_spacing = np.array(ct.header.get_zooms())
    orig_shape = ct.header.get_data_shape()
    target_affine = np.copy(ct.affine)
    target_affine[:3, :3] = np.diag(target_spacing / orig_spacing) @ ct.affine[:3, :3]
    target_shape = (orig_shape*(orig_spacing/target_spacing)).astype(int)

    ct_rs = nilearn.image.resample_img(ct, target_affine, target_shape,
                                    interpolation='continuous',
                                    fill_value=-1024)
    ct_rs.set_data_dtype(np.float32)

    suv_rs = nilearn.image.resample_to_img(suv, ct_rs,
                                            interpolation='continuous',
                                            fill_value=0)
    suv_rs.set_data_dtype(np.float32)

    nib.save(ct_rs, ct.parent/('rsCT.nii.gz'))
    nib.save(suv_rs, suv.parent/('rsSUV.nii.gz')) 


def main():
    # TODO add argparse
    # TODO adapt for tcia file structure
    target_orientation = 'LAS'
    target_spacing = [2.0, 2.0, 3.0]    
    ct_nii_pattern = '*gk*.nii.gz'
    suv_nii_pattern = 'suv*.nii.gz'

    # specify root directory containing all dicom directories (one per subject)
    dicom_root = plb.Path('/mnt/data/datasets/Stanford/dicom')
    dicom_dirs = list(dicom_root.glob('*'))
    nifti_dir = dicom_root.parent/'nifti'
    nifti_dir.mkdir(exist_ok=True)

    print(f'found {len(dicom_dirs)} dicom directories')
    print(dicom_dirs)

    print(f'niftis will be stored in {nifti_dir}')

    for dicom_dir in dicom_dirs:
        subject = dicom_dir.name
        print(f'processing {dicom_dir}')
        for d in (dicom_dir/'FinalDicoms').glob('*'):
            first_dcm = next(d.glob('*.dcm'))
            ds = pydicom.read_file(first_dcm)
            modality = ds.Modality
            print(f'  converting {modality} ...')
            # create nifti sub-directory 
            # (dirname = subject dicom dirname)
            nii_d = nifti_dir/subject
            nii_d.mkdir(exist_ok=True)

            if modality == 'PT':
                # select sample pet dicom file to read header
                suv_corr_factor = calculate_suv_factor(first_dcm)

            with tempfile.TemporaryDirectory() as tmp:
                tmp = plb.Path(str(tmp))
                # convert dicom directory to nifti
                # (store results in temp directory)
                dicom2nifti.convert_directory(d, str(tmp), 
                                            compression=True, reorient=True)
                nii = next(tmp.glob('*nii.gz'))
                # copy niftis to output folder with consistent naming
                f = nii_d/f'{modality}.nii.gz'
                shutil.copy(nii, nii_d/f'{modality}.nii.gz')
                print(f'    -> stored as {f}')

                if modality == 'PT':
                    print(f'  converting SUV ...')
                    g = nii_d/'SUV.nii.gz'
                    suv_pet_nii = convert_pet(nib.load(f), suv_factor=suv_corr_factor)
                    nib.save(suv_pet_nii, g)
                    print(f'    -> stored as {g}')
            
        print(f'  resample and align CT/SUV ...')
        resample_nifti(nii_d/'CT.nii.gz', nii_d/'SUV.nii.gz', target_orientation, target_spacing)


if __name__ == '__main__':
    main()
