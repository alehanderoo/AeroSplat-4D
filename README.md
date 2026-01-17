# Installation
export TORCH_CUDA_ARCH_LIST="9.0"
conda env create -f environment.yml
conda activate aeroSplat



# TODO

## isaacsim simulation
- run simulation headless using the yaml config file
- run simulation with all birds and drones in different environments from assets directory
- Create more flight (dynamic behavior, patterns etc.) paths to run simulations
- 

## stage-0_input-data
- move /home/sandro/thesis/aeroSplat-4D/stage-2_3DGS/depthsplat/inference/scripts/start_rtsp_simulator.sh to /home/sandro/thesis/aeroSplat-4D/stage-0_input-data
- make it a docker container / service
- 

## stage-1_fg-seg
- run voxeliser on pure pixel difference
    - output the 2D position, 3D position and confidence, masks per view, 2D bbox of object in each view
- record metrics for inference
    - PSNR, SSIM, LPIPS, (See Gaussian Mamba metrics)
    - memory
    - fps
    - latency
    - inference time
    - bottlenecks for acceleration
- make it a docker container / service

## stage-2_3DGS
- fix dependency issues in aeroSplat env. it does work in depthsplat env
- record the .ply models per frame for stage 3
- fine-tune depth anything model on objects in the air using the objaverse dataset and get metric depth: https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
- compare to : LGM, GRM, TriplaneGaussian, AGG, Splatter Image, Gamba
- record performance for inference
    - memory
    - fps
    - latency
    - inference time
    - bottlenecks for acceleration
- convert the checkpoint to onnx format so it can be optimized by deepstream in TensorRT
- run the 3DGS model on the fg-seg output
- make it a docker container / service


## stage-3_4D-classify
- generate input data for training
- run the 4D-classify model on the 3DGS output
- record metrics for inference
    - memory
    - fps
    - latency
    - inference time
    - bottlenecks for acceleration
- compare classification results with existing methods:
    - winner drone vs. bird challenge
    - yolo models
    - combine multiple camera view classifications
- make it a docker container / service

## abliation studies
- simulation
    - default / finetuned depth anything model
    - different weather conditions
    - different lighting conditions
    - different object types
    - different camera setups
    - objects outside of overlapping view frustum
- stage-1_fg-seg
    - compare to other fg-seg methods
    - 
- stage-2_3DGS
    - compare to other 3DGS methods
    - 
- stage-3_4D-classify
    - compare to other 4D-classify methods
    - compare transformer with mamba (do we need the memory reduction from mamba?)



## overall pipeline
- use deepstream for the data processing pipeline
- 
