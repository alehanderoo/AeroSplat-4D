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
- get TODOs from report
- run the 4D-classify model on the 3DGS output
- record metrics for inference
    - memory
    - fps
    - latency
    - inference time
    - bottlenecks for acceleration
- make it a docker container / service

## overall pipeline
- use deepstream for the data processing pipeline
- 
