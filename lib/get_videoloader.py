from torch.utils.data import DataLoader
from lib.datasets.mixed_dataset import MixedVidDataset
from lib.datasets.video_dataset import VideoDataset
from lib.core.data_loader import CheckpointDataLoader

def get_dataloaders(cfg=None):

    train_bs = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.NUM_WORKERS
    crop_size = cfg.IMG_RES
    dataset_list = cfg.DATASET.LIST
    valid_set = cfg.DATASET.TEST
    partition = cfg.DATASET.PARTITION

    train_seqlen = 16
    train_stride = train_seqlen
    stride = cfg.DATASET.STRIDE

    sequence_bs = cfg.TRAIN.SEQUENCE_BS
    test_bs = cfg.TEST.BATCH_SIZE


    print('Num of data loading workers:', num_workers)
    print('Sequence length:', train_seqlen)
    print('Sequence stride:', train_stride)

    print('Datasets:', dataset_list)
    print('Partition:', partition)
    
    train = MixedVidDataset(dataset_list, partition, is_train=True, use_augmentation=True,
                            normalization=True, cropped=True, crop_size=crop_size,
                            seqlen=train_seqlen, stride=train_stride) 
                            # seqlen=stride should be batch_size+2, set the real batch_size=1
                            # then reshape the input to B, window_size=3, ...
                            # --> make sure the frames are from continous seqs
    train_loader = CheckpointDataLoader(train, shuffle=True, batch_size=sequence_bs, num_workers=num_workers)
    
    test = VideoDataset(valid_set, is_train=False, use_augmentation=False, 
                    normalization=True, cropped=True, crop_size=crop_size, seqlen=16, stride=16) 
    test_loader = DataLoader(test, batch_size=8, shuffle=False, num_workers=num_workers)

    # ----------- debug ----------- #
    # import matplotlib.pyplot as plt
    # import torch
    # mean = torch.tensor([0.485, 0.456, 0.406]) # imagenet
    # std = torch.tensor([0.229, 0.224, 0.225])

    # for g in range(train_bs):
    #     batch = train[g]
    #     img = batch['img']

    #     kpts = batch['keypoints'][:, -24:].clone() # batch['keypoints'].shape  N, J, D ([3, 49, 3]) TODO(yiwen) check what is the previous 25
    #     valid = kpts[:,:,-1] > 0 # confidence > 0
    #     kpts = (kpts+1) * 256 / 2 # NOTE(yiwen) from range [-1,1] to real coords in 256*256 image
    #     kpts[~valid] = 0

    #     plt.rcParams['figure.figsize'] = 8, 5
    #     fig, axes = plt.subplots(1, 3)

    #     for i in range(seqlen):
    #         ax = axes[i%3]
    #         img_denorm = img[i] * std[:, None, None] + mean[:, None, None]
    #         ax.imshow(img_denorm.permute(1, 2, 0).clip(0, 1).numpy()) # stride=2
    #         ax.axis('off')
    #         ax.scatter(kpts[i,:,0], kpts[i,:,1], s=10)
    #     fig.tight_layout()
    #     plt.savefig(f'noaug_bedlambs{g}.png')
    #     plt.close()
    # ----------- debug ----------- #

    
    return [train_loader, test_loader]


