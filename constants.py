from pathlib import Path

global_min_prob = 0.30
picture_saved_box_version = 2

ptb_hash_len = 4

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
VID_EXT = (".mp4", ".avi", ".gif")

project_root_dir = Path(__file__).parent.absolute().resolve()

data_root = project_root_dir / 'data'
uncensored_path = data_root / 'uncensored'
censored_path = data_root / 'censored'
stickers_root_path = data_root / 'stickers'
debug_path = data_root / 'debug'
assets_path = data_root / 'assets'
temp_path = data_root / 'temp'
configs_path = temp_path / 'configs'
log_path = data_root / 'logs'

init_screen_image = assets_path / 'init_screen.png'
failed_screen = assets_path / 'failed_screen.png'

cache_root = project_root_dir / 'cache'
image_cache_path = cache_root / 'images'
video_cache_path = cache_root / 'videos'
trash_bin_cache_path = cache_root / 'trashbin'

model_root = project_root_dir / 'models'


# includes the RGB debug square color
classes = [
    [ 'exposed_anus',     ( 47,  79,  79), ],  # darkslategray
    [ 'exposed_armpits',  (139,  69,  19), ],  # saddlebrown
    [ 'covered_belly',    (  0, 100,   0), ],  # darkgreen
    [ 'exposed_belly',    (  0,   0, 139), ],  # darkblue
    [ 'covered_buttocks', (255,   0,   0), ],  # red
    [ 'exposed_buttocks', (255, 165,   0), ],  # orange
    [ 'face_femme',       (255, 255,   0), ],  # yellow
    [ 'face_masc',        (199,  21, 133), ],  # mediumvioletred
    [ 'covered_feet',     (  0, 255,   0), ],  # lime
    [ 'exposed_feet',     (  0, 250, 154), ],  # mediumspringgreen
    [ 'covered_breast',   (  0, 255, 255), ],  # aqua
    [ 'exposed_breast',   (  0,   0, 255), ],  # blue
    [ 'covered_vulva',    (216, 191, 216), ],  # thistle
    [ 'exposed_vulva',    (255,   0, 255), ],  # fuchsia
    [ 'exposed_chest',    ( 30, 144, 255), ],  # dodgerblue
    [ 'exposed_penis',    (240, 230, 140), ],  # khaki
    [ 'eyes_femme',       (255, 100, 255)  ],
    [ 'eyes_masc',        (255, 255, 100)  ],
]

bv_ss_timestamp1_name = 'bv_ss_timestamp1_name'
bv_ss_timestamp2_name = 'bv_ss_timestamp2_name'
bv_detect_timestamp1_name = 'bv_detect_timestamp1_name'
bv_detect_timestamp2_name = 'bv_detects_timestamp2_name'

bv_detect_shm_0_name = "bv_detect_shm_0_name"
bv_detect_shm_1_name = "bv_detect_shm_1_name"
bv_detect_shm_2_name = "bv_detect_shm_2_name"

model_outputs = [
        'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',    # boxes
        'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',  # scores
        'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0',  # classes
]

model_input = 'input_1:0'
