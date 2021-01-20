learning_rate = 1e-3
num_epochs = 14
img_save_path = "./results/"
lr_schular = [1e-3, 5e-4, 1e-4, 2e-5, 1e-6]
training_schedule = [4, 8, 10, 12, 14]
device_id = [0, 1]

crop_height, crop_width = (256, 256)
train_batch_size = 8   # 24
val_batch_size = 8

mean = [0.5, 0.5, 0.5]
std  = [1, 1, 1]

train_data_dir = '/home/zhihao/data/vimeo_septuplet/'
val_data_dir = '/home/zhihao/data/vimeo_septuplet/'
from datas.val_data_vimeo import ValData

save_freq = 4

mode = 'poly'
delta = 0.5

if mode == 'poly':
    model_save_path = "./modeldict/poly/"
elif mode == '1axis':
    model_save_path = "./modeldict/inverse_1axis_same/"
elif mode == '3axis':
    model_save_path = "./modeldict/inverse_3axis_same/"

