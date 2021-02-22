import torch


# [N, C, H, W] -> [N*C, H, W]
def collate_function(batch_list):
    # image_list = list()
    # label_list = list()
    # for i in range(len(batch_list)):
    #     image, labels = batch_list[i]
    #     image_list.append(image.squeeze(0))
    #     label_list.append(labels.squeeze(0))
    # return torch.cat(image_list, dim=0), torch.cat(label_list, dim=0)

    image_list = list()
    label_list = list()
    image_name_list = list()
    for i in range(len(batch_list)):
        image, labels, image_name = batch_list[i]
        image_list.append(image.squeeze(0))
        label_list.append(labels.squeeze(0))
        image_name_list.append(image_name)
    return torch.cat(image_list, dim=0), torch.cat(label_list, dim=0), image_name_list

