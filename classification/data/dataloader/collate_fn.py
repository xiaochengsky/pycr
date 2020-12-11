import torch

import torch.utils.data as Data
import numpy as np


# [N, C, H, W] -> [N*C, H, W]
def collate_function(batch_list):
    image_list = list()
    label_list = list()
    for i in range(len(batch_list)):
        image, label = batch_list[i]
        image_list.append(image.squeeze(0))
        label_list.append(label)

    # print(image_list.data)
    # print(label_list.data)
    return torch.cat(image_list, dim=0), torch.cat(label_list, dim=0)


test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
test = np.expand_dims(test, 0).repeat(12, axis=0)
test = np.expand_dims(test, 0).repeat(12, axis=0)

inputing = torch.tensor(np.array([test[i:i + 3] for i in range(10)]))
target = torch.tensor(np.array([test[0][0][i:i + 1] for i in range(10)]))

torch_dataset = Data.TensorDataset(inputing, target)
batch = 3

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch,  # 批大小
    # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
    # collate_fn=collate_function
)

for (i, j) in loader:
    print(i.shape)
    print(j.shape)


