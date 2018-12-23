rm lattice_filter.so
mkdir genLatticeFilterOp_multiBatch/build_dir
cd genLatticeFilterOp_multiBatch/build_dir

setenv CUDA_COMPILER /usr/local/cuda/bin/nvcc
setenv CXX_COMPILER /usr/bin/g++-5
setenv CMAKE /mnt/ssd/REM/liverProject/crfrnn3D/permutohedral_lattice/cmake-3.13.2/bin/cmake

set SPATIAL_DIMS=2
set INPUT_CHANNELS=2
set REFERENCE_CHANNELS=1
set MAKE_TESTS=False

$CMAKE -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
                                -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
                                -D CMAKE_CUDA_HOST_COMPILER=${CXX_COMPILER} \
                                -D SPATIAL_DIMS=${SPATIAL_DIMS} \
                                -D INPUT_CHANNELS=${INPUT_CHANNELS} \
                                -D REFERENCE_CHANNELS=${REFERENCE_CHANNELS} \
                                -D MAKE_TESTS=${MAKE_TESTS} \
                                -G "CodeBlocks - Unix Makefiles" ../


make

cp lattice_filter.so ../../

cd ../../
rm -r genLatticeFilterOp_multiBatch/build_dir

