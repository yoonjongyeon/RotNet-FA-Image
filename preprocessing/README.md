# Preprocessing diffusion MRI (dMRI) and Extracting Fractional Anisotropy (FA) map

## Overview
This script is for preprocessing raw dMRI file and extracting FA map from preprocessed dMRI file.  
The preprocessing includes following:
- `Motion Correction`: Corrects for subject motion during the scan to ensure accurate alignment of diffusion-weighted images.
- `Skull Stripping`: Removes non-brain tissues such as the skull and scalp to focus on brain structures for analysis.
- `Bias Field Correction`: Compensates for intensity inhomogeneities caused by variations in the magnetic field, improving the uniformity of the MR signal.

## Installation
Following librarys are required to preprocess the data and extract FA map:
* [MRTrix3](https://www.mrtrix.org/download/)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation#Installing_FSL)
* [ANTs](https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS)

## Running the script
Run `preproc_fa.sh` with following code:
```bash
./preproc_fa.sh <out_dir> <dMRI_file> <bvec_file> <bval_file>
```
* **`<out_dir>`** Path on the host machine to the *directory* in which the output FA map (in NIfTI) will be saved.
* **`<dMRI_file>`** Path on the host machine to the diffusion MRI in NIfTI or other supported formats.
* **`<bvec_file>`** Path on the host machine to the b-vector file, which describes gradient directions used during diffusion imaging.
* **`<bval_file>`** Path on the host machine to the b-value file, which contains gradient strength values corresponding to the b-vectors.