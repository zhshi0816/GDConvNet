import torch.utils.data as data
from PIL import Image, ImageFile
from configs.config import val_data_dir
ImageFile.LOAD_TRUNCATED_IMAGES = True
from util.utils import transform


class ValData(data.Dataset):
    def __init__(self, mode='full'):
        super().__init__()
        val_list = val_data_dir + 'sep_testlist.txt'
        with open(val_list) as f:
            contents = f.readlines()
            img_path = [i.strip() for i in contents]

        if mode != 'full':
            tmp = []
            for i, value in enumerate(img_path):
                if i%38 == 0:
                    tmp.append(value)
            img_path = tmp
        self.img_path = img_path

        self.forward_trans = transform()

    def __getitem__(self, index):
        img_path = self.img_path[index]

        img1 = Image.open(val_data_dir + 'sequences/' + img_path + '/im1.png')
        img2 = Image.open(val_data_dir + 'sequences/' + img_path + '/im3.png')
        img3 = Image.open(val_data_dir + 'sequences/' + img_path + '/im4.png')
        img4 = Image.open(val_data_dir + 'sequences/' + img_path + '/im5.png')
        img5 = Image.open(val_data_dir + 'sequences/' + img_path + '/im7.png')

        img1 = self.forward_trans(img1)
        img2 = self.forward_trans(img2)
        img4 = self.forward_trans(img4)
        img5 = self.forward_trans(img5)
        img3 = self.forward_trans(img3)

        img_name = img_path + '/im4.png'
        img_name = img_name.replace('/', '_')

        return img1, img2, img4, img5, img3, img_name

    def __len__(self):
        return len(self.img_path)

if __name__ == '__main__':
    dataset = ValData()
    print(dataset[0][0].shape)
