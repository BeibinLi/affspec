import os
import numpy as np
import pandas as pd

import torch
import torch.utils.data

from skimage import io

import pdb

# %%
MISSING_VAL = 999


# %%
class DISFAplusDataset(torch.utils.data.Dataset):
    def __init__(self, csv_name, transform=None,
                 base_path="/m-patho1/ferg/DISFA+/"):
        self.csv_name = csv_name
        df = pd.read_csv(csv_name)
        df = df.reset_index()
        self.df = df

        self.transform = transform
        self.base_path = base_path

        self.column_names = ["AU%d" % i for i in range(1, 44)]

        for colname in self.column_names:
            if colname not in self.df.keys():
                self.df[colname] = MISSING_VAL  # means: NA

        self.calculate_au_active_ratio()

        print("DISFAplus DF size:", self.df.shape)

    def calculate_au_active_ratio(self):
        """
        Get the activation ratio for each action units.
        If there are 43 action units, then the self.label_activate_ratio should
        be an array with 43 numbers.
        """
        self.label_activate_ratio = []
        for lab in self.column_names:
            labels = self.df[lab]
            if labels[0] == MISSING_VAL:  # this column is missing
                self.label_activate_ratio.append(np.nan)
            else:
                self.label_activate_ratio.append(
                    np.sum(labels > 0) / self.df.shape[0])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        try:
            imgpath = self.df.path[idx]
#            imgpath = imgpath.replace("/homes/gws/beibin/Emotionet/",  self.base_path )
            imgpath = imgpath.replace(
                "O:/unix/projects/grail/deepalianb/DISFA+/",
                self.base_path)

            t, r, b, l = self.df.t[idx], self.df.r[idx], self.df.b[idx], self.df.l[idx]

            img = io.imread(os.path.join(imgpath))

            input_data = {
                "image_path": imgpath,
                'image': img,
                "top": t,
                "right": r,
                "bottom": b,
                "left": l}

            if self.transform:
                data = self.transform(input_data)
            else:
                raise("unknown transform function!")

            # possibility of expression
            label = self.df.loc[idx, self.column_names].tolist()
            label = np.array(label)

            weights = []
            for i in range(len(label)):
                if label[i] == MISSING_VAL:
                    w = 0
                else:
                    w = 1 / \
                        self.label_activate_ratio[i] if label[i] else 1 / \
                        (1 - self.label_activate_ratio[i])
                weights.append(w)

            weights = np.array(weights).astype(np.float)
#
            label = {"expression": -100,
                     "action units": label.reshape(-1),
                     "valence": -100.0,
                     "arousal": -100.0,
                     }  # wrap in a dictionary

            weights = {"action units": weights.reshape(-1),
                       "expression": 0.0,
                       "valence": 0.0,
                       "arousal": 0.0,
                       }  # wrap in a dictionary

            return data, label, weights

        except Exception as e:
            print(e)
            print(t, r, b, l, imgpath)
            # print( idx )
            # pdb.set_trace()
            # raise( 'dam' )
            # Return Another Random Choice
            return self.__getitem__(np.random.randint(0, self.__len__()))


if __name__ == "__main__":
    fname = "DISFAplus_test.csv"

    from torchvision import transforms

    from custom_transforms import FaceCrop

    transformer = transforms.Compose([
        FaceCrop(scale=1.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=224, scale=(0.9, 1.1)),
        transforms.RandomAffine(5, shear=20),
        transforms.ToTensor()
    ])

    train_dataset = DISFAplusDataset(fname, transform=transformer)

#
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True)

    # %%
    import tqdm
    count = 0

    for inputs, labels, weights in tqdm.tqdm(trainloader):
        x = inputs.numpy()
        x = x.reshape(3, 224, 224)
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 0, 1)

#        io.imsave( "test_%d.jpg" % count, x )

        print(inputs.shape, labels, weights)

        count += 1

        if count > 100:
            break
#        break
        pass
