import torch.utils.data as data
from PIL import Image, ImageFile
from configs.config import train_data_dir, crop_height, crop_width
from random import randrange
ImageFile.LOAD_TRUNCATED_IMAGES = True
from util.utils import transform

class TrainData(data.Dataset):
    def __init__(self, flip=True):
        super().__init__()
        train_list = train_data_dir + 'sep_trainlist.txt'
        with open(train_list) as f:
            contents = f.readlines()
            img_path = [i.strip() for i in contents]

        self.img_path = img_path
        self.forward_trans = transform()

    def __getitem__(self, index):
        img_path = self.img_path[index]

        img1 = Image.open(train_data_dir + 'sequences/' + img_path + '/im1.png')
        img2 = Image.open(train_data_dir + 'sequences/' + img_path + '/im3.png')
        img3 = Image.open(train_data_dir + 'sequences/' + img_path + '/im4.png')
        img4 = Image.open(train_data_dir + 'sequences/' + img_path + '/im5.png')
        img5 = Image.open(train_data_dir + 'sequences/' + img_path + '/im7.png')

        transpose = randrange(0, 3)
        if transpose!=0:
            aug_list = ['FLIP_LEFT_RIGHT', 'FLIP_TOP_BOTTOM']
            aug_ind = randrange(0, len(aug_list))

            aug_method = getattr(Image, aug_list[aug_ind])

            img1 = img1.transpose(aug_method)
            img2 = img2.transpose(aug_method)
            img3 = img3.transpose(aug_method)
            img4 = img4.transpose(aug_method)
            img5 = img5.transpose(aug_method)

        rev_ind = randrange(0, 2)

        if rev_ind:
            img5, img4, img2, img1 = img1, img2, img4, img5


        width, height = img1.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        img1 = img1.crop((x, y, x + crop_width, y + crop_height))
        img2 = img2.crop((x, y, x + crop_width, y + crop_height))
        img3 = img3.crop((x, y, x + crop_width, y + crop_height))
        img4 = img4.crop((x, y, x + crop_width, y + crop_height))
        img5 = img5.crop((x, y, x + crop_width, y + crop_height))

        img1 = self.forward_trans(img1)
        img2 = self.forward_trans(img2)
        img4 = self.forward_trans(img4)
        img5 = self.forward_trans(img5)
        img3 = self.forward_trans(img3)

        return img1, img2, img4, img5, img3

    def __len__(self):
        return len(self.img_path)


