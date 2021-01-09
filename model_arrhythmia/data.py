import os
import torch
import numpy as np
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['cifar10']:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': False}

        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

        dataset = {}
        dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)

        if opt.task == 'anomaly_detect':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_cifar_anomaly_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                abn_cls_idx=classes[opt.anomaly_class]
            )
        elif opt.task == 'random_walk':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_sub_cifar10_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
            )
        elif opt.task == 'llk_trend':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_sub_cifar10_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                abn_cls_idx=[0]
            )
        elif opt.task == 'rw_llk': ##for simplication, let's do it together
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_cifar_rwllk_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                abn_cls_idx=classes[opt.anomaly_class]
            )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batch_size,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader
    elif opt.dataset in ['mnist']:
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        ##the second one is good for mnist
        transform = transforms.Compose([transforms.Resize((opt.isize, opt.isize)), transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        if opt.task == 'anomaly_detect':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_mnist_anomaly_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                abn_cls_idx=opt.anomaly_class
            )
        elif opt.task == 'random_walk':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_sub_mnist_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
            )

        elif opt.task == 'llk_trend':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_sub_mnist_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                abn_cls_idx = [0]
            )
        elif opt.task == 'rw_llk':  ##for simplication, let's do it together
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_mnist_rwllk_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                abn_cls_idx=opt.anomaly_class
            )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batch_size,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}

        return dataloader

    elif opt.dataset in ['svhn']:
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        ##it seems that the transform in our GAN, we can use identical transform for both cifar10 and mnist
        ##the following is not working well at all.
        # transform = transforms.Compose(
        #     [
        #         transforms.Scale(opt.isize),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ]
        # )
        ##the second one is good for mnist
        transform = transforms.Compose([transforms.Resize((opt.isize, opt.isize)), transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        dataset = {}
        dataset['train'] = SVHN(root='./data', split='train', download=True, transform=transform)
        dataset['test'] = SVHN(root='./data', split='test', download=True, transform=transform)

        if opt.task == 'anomaly_detect':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_cifar_anomaly_dataset( ##not sure if we need to write a get_svhn_anomaly_dataset yet.
                trn_img=dataset['train'].data,
                trn_lbl=dataset['train'].labels,
                tst_img=dataset['test'].data,
                tst_lbl=dataset['test'].labels,
                abn_cls_idx=opt.anomaly_class
            )


        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batch_size,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

    elif opt.dataset in ['mnist2']:
        opt.anomaly_class = int(opt.anomaly_class)

        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}

        transform = transforms.Compose(
            [
                transforms.Scale(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )

        dataset = {}
        dataset['train'] = MNIST(root='./data', train=True, download=True, transform=transform)
        dataset['test'] = MNIST(root='./data', train=False, download=True, transform=transform)

        if opt.task == 'anomaly_detect':
            dataset['train'].train_data, dataset['train'].train_labels, \
            dataset['test'].test_data, dataset['test'].test_labels = get_mnist2_anomaly_dataset(
                trn_img=dataset['train'].train_data,
                trn_lbl=dataset['train'].train_labels,
                tst_img=dataset['test'].test_data,
                tst_lbl=dataset['test'].test_labels,
                nrm_cls_idx=opt.anomaly_class,
                proportion=opt.proportion
            )

        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batch_size,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x]) for x in splits}
        return dataloader

    # elif opt.dataset in ['celebA']:
    #     ##not for abnormal detection but not for classification either
    #     splits = ['train', 'test']
    #     drop_last_batch = {'train': True, 'test': False}
    #     shuffle = {'train': True, 'test': True}
    #     transform = transforms.Compose([transforms.Scale(opt.isize),
    #                                     transforms.CenterCrop(opt.isize),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    #     print(os.path.abspath('./data/celebA'))
    #     dataset = datasets.ImageFolder(os.path.abspath('./data/celebA'), transform)
    #     dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    #     return dataloader
    elif opt.dataset in ['celebA']:
        ##not for abnormal detection but not for classification either
        # import helper
        # helper.download_extract('celeba', opt.dataroot)
        # splits = ['train', 'test']
        # drop_last_batch = {'train': True, 'test': False}
        # shuffle = {'train': True, 'test': True}
        # transform = transforms.Compose([transforms.Scale(opt.isize),
        #                                 transforms.CenterCrop(opt.isize),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        #
        # # transform = transforms.Compose([
        # #     transforms.CenterCrop(160),
        # #     transforms.Scale(opt.isize),
        # #     transforms.ToTensor(),)
        #
        # dataset = ImageFolder(root=image_root, transform=transforms.Compose([
        #     transforms.CenterCrop(160),
        #     transforms.Scale(scale_size),
        #     transforms.ToTensor(),
        #     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ]))
        #
        # dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
        # dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
        #                                              batch_size=opt.batch_size,
        #                                              shuffle=shuffle[x],
        #                                              num_workers=int(opt.workers),
        #                                              drop_last=drop_last_batch[x]) for x in splits}

        dataloader = get_loader(
            './data/celebA', 'train', opt.batch_size, opt.isize)
        return dataloader


def get_loader(root, split, batch_size, scale_size, num_workers=2, shuffle=True):
    dataset_name = os.path.basename(root)
    #image_root = os.path.join(root, 'splits', split)
    image_root = os.path.join(root, split)
    print('image_root=%s' % (image_root))

    if dataset_name in ['celebA']:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    else:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader

##
def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


## Random_walk and llk_trend are done together
def get_cifar_rwllk_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == 1)[0] ##let's do it for simplity
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == 1)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


##temporarily, we use abnormal function for now.
def get_sub_cifar10_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=[0,1,2], manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # Convert train-test labels into numpy array.
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)
    # --
    # Find normal abnormal indexes.

    sub_nrm_trn_ind = []
    sub_abn_trn_ind = []
    sub_nrm_tst_ind = []
    sub_abn_tst_ind = []
    for ind in abn_cls_idx:
        sub_nrm_trn_ind.extend(np.where(trn_lbl == ind)[0])
        sub_abn_trn_ind.extend(np.where(trn_lbl == ind)[0])
        sub_nrm_tst_ind.extend(np.where(tst_lbl == ind)[0])
        sub_abn_tst_ind.extend(np.where(tst_lbl == ind)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[sub_nrm_trn_ind]    # Normal training images
    abn_trn_img = trn_img[sub_abn_trn_ind]    # Abnormal training images.
    nrm_tst_img = tst_img[sub_nrm_tst_ind]    # Normal training images
    abn_tst_img = tst_img[sub_abn_tst_ind]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[sub_nrm_trn_ind]    # Normal training labels
    abn_trn_lbl = trn_lbl[sub_abn_trn_ind]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[sub_nrm_tst_ind]    # Normal training labels
    abn_tst_lbl = tst_lbl[sub_abn_tst_ind]    # Abnormal training labels.

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##
def get_mnist_rwllk_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == 1)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == 1)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # --
    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = torch.cat((nrm_trn_img, nrm_tst_img), dim=0)
        nrm_lbl = torch.cat((nrm_trn_lbl, nrm_tst_lbl), dim=0)
        abn_img = torch.cat((abn_trn_img, abn_tst_img), dim=0)
        abn_lbl = torch.cat((abn_trn_lbl, abn_tst_lbl), dim=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl

##temporarily, we use abnormal function for now.
def get_sub_mnist_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=[0,1,2], manualseed=-1):
    """[summary]

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    sub_nrm_trn_ind = []
    sub_abn_trn_ind = []
    sub_nrm_tst_ind = []
    sub_abn_tst_ind = []
    for ind in abn_cls_idx:
        nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == ind)[0])
        abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == ind)[0])
        nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == ind)[0])
        abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == ind)[0])
        sub_nrm_trn_ind.append(nrm_trn_idx)
        sub_abn_trn_ind.append(abn_trn_idx)
        sub_nrm_tst_ind.append(nrm_tst_idx)
        sub_abn_tst_ind.append(abn_tst_idx)

    sub_nrm_trn_ind,sorted_nrm_trn_indices = torch.sort(torch.cat(sub_nrm_trn_ind))
    sub_abn_trn_ind,sorted_abn_trn_indices = torch.sort(torch.cat(sub_abn_trn_ind))
    sub_nrm_tst_ind,sorted_nrm_tst_indices = torch.sort(torch.cat(sub_nrm_tst_ind))
    sub_abn_tst_ind,sorted_abn_tst_indices = torch.sort(torch.cat(sub_abn_tst_ind))

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[sub_nrm_trn_ind]    # Normal training images
    abn_trn_img = trn_img[sub_abn_trn_ind]    # Abnormal training images.
    nrm_tst_img = tst_img[sub_nrm_tst_ind]    # Normal training images
    abn_tst_img = tst_img[sub_abn_tst_ind]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[sub_nrm_trn_ind]    # Normal training labels
    abn_trn_lbl = trn_lbl[sub_abn_trn_ind]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[sub_nrm_tst_ind]    # Normal training labels
    abn_tst_lbl = tst_lbl[sub_abn_tst_ind]    # Abnormal training labels.

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl



##
def get_mnist2_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, nrm_cls_idx=0, proportion=0.5):
    """ Create mnist 2 anomaly dataset.

    Arguments:
        trn_img {np.array} -- Training images
        trn_lbl {np.array} -- Training labels
        tst_img {np.array} -- Test     images
        tst_lbl {np.array} -- Test     labels

    Keyword Arguments:
        nrm_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [tensor] -- New training-test images and labels.
    """
    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == nrm_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != nrm_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == nrm_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != nrm_cls_idx)[0])

    # Get n percent of the abnormal samples.
    abn_tst_idx = abn_tst_idx[torch.randperm(len(abn_tst_idx))]
    abn_tst_idx = abn_tst_idx[:int(len(abn_tst_idx) * proportion)]


    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl