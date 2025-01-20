#!/bin/bash

out_dir=$1
dmri_file=$2
bvec_file=$3
bval_file=$4

echo "<<<<<Obtaining FA map>>>>>"

echo "<<<<Preprocessing dMRI data>>>>"

echo "**It is recommended to use isotropice resolution**"
echo "*If your data is anisotropic, you can use the mrgrid command from MRTrix3 to get isotropic voxel spacing*"

# Distortion and Motion Correction
echo "<<Motion Correction>>"
corrected_dmri_file="$out_dir/corrected_dmri.nii.gz"
cmd="mcflirt -in $dmri_file -out $corrected_dmri_file -refvol 0 -plots"
[ ! -f $corrected_dmri_file ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

# Skull Stripping
echo "<<Skull Stripping>>"
skull_stripped_file="$out_dir/skull_stripp_dmri.nii.gz"
skull_mask_file="$out_dir/skull_stripp_dmri_mask.nii.gz"
skull_stripped_file_4d="$out_dir/skull_stripp_dmri_4d.nii.gz"
cmd="bet $corrected_dmri_file $skull_stripped_file -m"
[ ! -f $skull_stripped_file ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"
cmd="mrcalc $corrected_dmri_file $skull_mask_file -mult $skull_stripped_file_4d"
[ ! -f $skull_stripped_file_4d ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

# Bias Field Correction
echo "<<Bias Field Correction>>"
processed_dmri_file="$out_dir/processed_dmri.nii.gz"
cmd="dwibiascorrect ants -mask $skull_mask_file  -fslgrad $bvec_file $bval_file $skull_stripped_file_4d $processed_dmri_file"
[ ! -f $processed_dmri_file ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

echo "<<<<Extracting FA map from preprocessed dMRI data>>>>"

echo "<<Tensor Fitting>>"
tensor_file="$out_dir/tensor.mif"
cmd="dwi2tensor $processed_dmri_file $tensor_file -fslgrad $bvec_file $bval_file"
[ ! -f $out_dir/tensor.mif ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

echo "<<FA Calculation>>"
fa_map_mif="$out_dir/fa_map.mif"
fa_map_nii="$out_dir/fa_map.nii.gz"
cmd="tensor2metric $tensor_file -fa $fa_map_mif"
[ ! -f $out_dir/fa_map.mif ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"
cmd="mrconvert $fa_map_mif $fa_map_nii"
[ ! -f $out_dir/fa_map.nii.gz ] && (echo $cmd && eval $cmd) || echo "Output exists, skipping!"

echo "<<<Cleaning up intermediate files>>>"
rm -f $corrected_dmri_file
rm -f $skull_stripped_file
rm -f $skull_mask_file
rm -f $skull_stripped_file_4d
rm -f $processed_dmri_file
rm -f $tensor_file
rm -f $fa_map_mif
rm -f $out_dir/corrected_dmri.nii.gz.par

echo "<<<<<Obtaining FA map finished>>>>>"


