from argparse import ArgumentParser


def load_args():
    parser = ArgumentParser(description="PyTorch Retrieval")
    parser.add_argument('--config-file', help="train config file path", required=True, type=str)
    parser.add_argument("-tag", "--TAG", type=str)
    parser.add_argument('--gpus', help="gpu nums", type=int, default=1)

    # model
    parser.add_argument('--load-path', help="path of pre-train model", type=str)
    parser.add_argument("-device", type=int, nargs='+', help="list of device_id, e.g. [0,1]")

    arg = parser.parse_args()
    return arg


def merge_from_arg(config, arg):  # --> dict{},dict{}
    if arg['TAG']:
        config['tag'] = arg['TAG']
    else:
        config['tag'] = (((arg['config_file']).split('/')[-1]).split('.'))[0]
    print("TAG : ", config['tag'])

    # if arg['max_num_devices']:
    #     config['max_num_devices'] = arg['max_num_devices']

    return config


if __name__ == "__main__":
    pass
