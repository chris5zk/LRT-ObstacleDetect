# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:46:12 2022

@author: chrischris
"""

from importpackage import *

########## Datasets ##########
dataset_base_path = './dataset'
target = 'images'     # images / videos

### train ###
# train_dataset_path = f"{dataset_base_path}/train"

### test ###
test_dataset_path = f"{dataset_base_path}/test"
test_path = f"{test_dataset_path}/{target}"

# original dataset
test_org_path = f"{test_path}/original"
test_org_images = f"{test_org_path}/rail"
org_video = "wulai_short.mp4"
test_org_video = f"{test_org_path}/{org_video}"

# segmentation dataset
test_seg_path = f"{test_path}/seg"    
test_seg_images = f"{test_seg_path}/rail"
seg_video = "out_short.mp4"
test_seg_video = f"{test_seg_path}/{seg_video}"

########## Models ##########
### yolact-edge ###
# weights
yolact_edge_pt = './my_yolact/weights/yolact_edge_32_20000.pth'

# parse_args
class parse_arguments:
    # key arguments
    trained_model = f'{os.path.abspath(yolact_edge_pt)}'                                                                # Trained state_dict file path to open. If "interrupt", this will open the interrupt file.
    image = None                                                                                                        # A path to an image to use for display.
    images = f'{os.path.abspath(test_org_images)}:{os.path.abspath(test_seg_images)}' if target=='images' else None     # An input folder of images and output folder to save detected images. Should be in the format input:output.
    video = f'{os.path.abspath(test_org_video)}:{os.path.abspath(test_seg_video)}' if target=='videos' else None        # A path to a video to evaluate on. Passing in a number will use that index webcam.
    video_multiframe = 1                                                                                                # The number of frames to evaluate in parallel to make videos play at higher fps.
    
    top_k = 1                       # Further restrict the number of predictions to parse.
    score_threshold = 0.1           # Detections with a score under this threshold will not be considered. This currently only works in display mode.
    
    cuda = True                     # Use cuda to evaulate model.
    config = None                   # The config object to use.
    dataset = None                  # If specified, override the dataset specified in the config with this one (example: coco2017_dataset).
    max_images = -1                 # The maximum number of images from the dataset to consider. Use -1 for all.
    
    display_masks = True            # Whether or not to display masks over bounding boxes.
    display_bboxes = False          # Whether or not to display bboxes around masks.
    display_text = False            # Whether or not to display text (class [score]).
    display_scores = True           # Whether or not to display scores in addition to classes.
    display_lincomb = False         # If the config uses lincomb masks, output a visualization of how those masks are created.
    
    # other arguments
    fast_nms = True                 # Whether to use a faster, but not entirely correct version of NMS.
    fast_eval = False               # Skip those warping frames when there is no GT annotations.
    eval_stride = 5                 # The default frame eval stride.
    drop_weights = None             # Drop specified weights (split by comma) from existing model.

    deterministic = False           # Whether to enable deterministic flags of PyTorch for deterministic results.
    seed = None                     # The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.
    no_crop = False                 # Do not crop output masks with the predicted bounding box.
    
    calib_images = None                 # Directory of images for TensorRT INT8 calibration, for explanation of this field, please refer to `calib_images` in `data/config.py`.
    trt_batch_size = 1                  # Maximum batch size to use during TRT conversion. This has to be greater than or equal to the batch size the model will take during inferece.
    disable_tensorrt = True             # Don't use TensorRT optimization when specified.
    use_fp16_tensorrt = False           # This replaces all TensorRT INT8 optimization with FP16 optimization when specified.
    use_tensorrt_safe_mode = False      # This enables the safe mode that is a workaround for various TensorRT engine issues.
    
    # some path
    ap_data_file = 'results/ap_data.pkl'                    # In quantitative mode, the file to save detections before calculating mAP.
    bbox_det_file = 'results/bbox_detections.json'          # The output file for coco bbox results if --coco_results is set.
    mask_det_file = 'results/mask_detections.json'          # The output file for coco mask results if --coco_results is set.
    web_det_path = 'web/dets/'                              # If output_web_json is set, this is the path to dump detections into.
 
    # set_default
    benchmark = False   # Equivalent to running display mode but without displaying an image.
    crop = False         
    display = False     # Display qualitative results instead of quantitative ones.
    detect = False      # Don't evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.
    resume = False      # If display not set, this resumes mAP calculations from the ap_data_file.
    shuffle = False     # Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.
    
    no_bar = False      # Do not output the status bar. This is useful for when piping to a file.
    no_sort = False     # Do not sort images by hashed image ID.
    no_hash = False     
    
    output_coco_json = True         # If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.
    output_web_json =  False        # If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.
    mask_proto_debug = False        # Outputs stuff for scripts/compute_mask.py.
     

### yolov5 ###
# weights
yolov5_pt = 'object_5.pt'

# data
batch_size = 4

# video
mode = 'sec'    # pts / sec
start = 0
end = 3

########## Output ##########
output_base_path = 'runs/output'