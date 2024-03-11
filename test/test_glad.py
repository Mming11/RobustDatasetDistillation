import os
import numpy as np
from math import ceil
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset
from train import define_model, train, validate
from data import TensorDataset, ImageFolder, MultiEpochsDataLoader, ImageFolder_subset
from torchvision import datasets, transforms
from data import save_img, transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion, transform_tinyimagenet
import models.resnet as RN
import models.densenet_cifar as DN
from coreset import randomselect, herding
from efficientnet_pytorch import EfficientNet
from sklearn.cluster import KMeans


DATA_PATH = "./results"

def return_data_path(args):
    if args.factor > 1:
        init = 'mix'
    else:
        init = 'random'

    #You can put the results at ./results.
    if 'glad' in args.slct_type:
        name = args.name
        if name == '':
            if args.dataset == 'cifar10':
                name = f'glad_0'

            elif args.dataset == 'cifar100':
                name = f'cifar100/rdd'

            elif args.dataset == 'imagenet':
                if args.nclass == 10:
                    name = f'rdd'

            elif args.dataset == 'svhn':
                name = f'rdd'
                if args.factor == 1 and args.ipc == 1:
                    args.mixup = 'vanilla'
                    args.dsa_strategy = 'color_crop_cutout_scale_rotate'

        path_list = [f'{name}_ipc{args.ipc}']

    elif args.slct_type == 'dsa':
        path_list = [f'cifar10/dsa/res_DSA_CIFAR10_ConvNet_{args.ipc}ipc']
    elif args.slct_type == 'kip':
        path_list = [f'cifar10/kip/kip_ipc{args.ipc}']
    else:
        path_list = ['']

    return path_list


def resnet10_in(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'instance', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: instance")
    return model


def resnet10_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 10, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-10, norm: batch")
    return model


def resnet18_bn(args, nclass, logger=None):
    model = RN.ResNet(args.dataset, 18, nclass, 'batch', args.size, nch=args.nch)
    if logger is not None:
        logger(f"=> creating model resnet-18, norm: batch")
    return model


def densenet(args, nclass, logger=None):
    if 'cifar' == args.dataset[:5]:
        model = DN.densenet_cifar(nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating DenseNet")
    return model


def efficientnet(args, nclass, logger=None):
    if args.dataset == 'imagenet':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    else:
        raise AssertionError("Not implemented!")

    if logger is not None:
        logger(f"=> creating EfficientNet")
    return model


def load_ckpt(model, file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    model.load_state_dict(checkpoint)

    if verbose:
        print(f"\n=> loaded checkpoint '{file_dir}'")


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary


def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target

def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    save_img('./results/test_dec.png', data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec

def load_data_path(args):
    """Load condensed data from the given path
    """
    if args.pretrained:
        args.augment = False

    print()
    if args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        train_transform, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)
        # Load condensed dataset
        if 'glad' in args.slct_type:
     
            data = torch.load(os.path.join(f'{args.save_dir}', 'images_best.pt')) #change
            target = torch.load(os.path.join(f'{args.save_dir}', 'labels_best.pt'))
                

            print("Load condensed data ", data.shape, args.save_dir)

            train_transform, _ = transform_imagenet(augment=args.augment,
                                                    from_tensor=True,
                                                    size=args.size,
                                                    rrc=args.rrc)
            train_dataset = TensorDataset(data, target, train_transform)
        else:
            train_dataset = ImageFolder(traindir,
                                        train_transform,
                                        nclass=args.nclass,
                                        seed=args.dseed,
                                        slct_type=args.slct_type,
                                        ipc=args.ipc,
                                        load_memory=args.load_memory)
            print(f"Test {args.dataset} random selection {args.ipc} (total {len(train_dataset)})")

        if args.subset == "f":
            val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)
        else:
            val_dataset = ImageFolder_subset(valdir,
                                  test_transform,
                                  subset=args.subset,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory
                                  )
    else:
        if args.dataset[:5] == 'cifar':
            transform_fn = transform_cifar
        elif args.dataset == 'svhn':
            transform_fn = transform_svhn
        elif args.dataset == 'mnist':
            transform_fn = transform_mnist
        elif args.dataset == 'fashion':
            transform_fn = transform_fashion
        elif args.dataset == 'tiny':
            transform_fn = transform_tinyimagenet
        train_transform, test_transform = transform_fn(augment=args.augment, from_tensor=False)

        # Load condensed dataset
        if 'glad' in args.slct_type:

            data = torch.load(os.path.join(f'{args.save_dir}', 'images_best.pt')) #change
            target = torch.load(os.path.join(f'{args.save_dir}', 'labels_best.pt'))
 
            print("Load condensed data ", args.save_dir, data.shape)
       
            train_transform, _ = transform_fn(augment=args.augment, from_tensor=True)
            train_dataset = TensorDataset(data, target, train_transform)

        elif args.slct_type in ['dsa', 'kip']:
            condensed = torch.load(f'{args.save_dir}.pt')
            try:
                condensed = condensed['data']
                data = condensed[-1][0]
                target = condensed[-1][1]
            except:
                data = condensed[0].permute(0, 3, 1, 2)
                target = torch.arange(args.nclass).repeat_interleave(len(data) // args.nclass)

            if args.factor > 1:
                data, target = decode(args, data, target)
            # These data are saved as the normalized values!
            train_transform, _ = transform_fn(augment=args.augment,
                                              from_tensor=True,
                                              normalize=False)
            train_dataset = TensorDataset(data, target, train_transform)
            print("Load condensed data ", args.save_dir, data.shape)

        else:
            if args.dataset == 'cifar10':
                train_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                             train=True,
                                                             transform=train_transform)
            elif args.dataset == 'cifar100':
                train_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                              train=True,
                                                              transform=train_transform)
            elif args.dataset == 'svhn':
                train_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                          split='train',
                                                          transform=train_transform)
                train_dataset.targets = train_dataset.labels
            elif args.dataset == 'mnist':
                train_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                           train=True,
                                                           transform=train_transform)
            elif args.dataset == 'fashion':
                train_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                                  train=True,
                                                                  transform=train_transform)

            indices = randomselect(train_dataset, args.ipc, nclass=args.nclass)
            train_dataset = Subset(train_dataset, indices)
            print(f"Random select {args.ipc} data (total {len(indices)})")

        # Test dataset
        if args.dataset == 'cifar10':
            val_dataset = torchvision.datasets.CIFAR10(args.data_dir,
                                                       train=False,
                                                       download=True,
                                                       transform=test_transform)
        elif args.dataset == 'cifar100':
            val_dataset = torchvision.datasets.CIFAR100(args.data_dir,
                                                        train=False,
                                                        download=True,
                                                        transform=test_transform)
        elif args.dataset == 'svhn':
            val_dataset = torchvision.datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                                    split='test',
                                                    download=True,
                                                    transform=test_transform)
        elif args.dataset == 'mnist':
            val_dataset = torchvision.datasets.MNIST(args.data_dir,
                                                     train=False,
                                                     transform=test_transform)
        elif args.dataset == 'fashion':
            val_dataset = torchvision.datasets.FashionMNIST(args.data_dir,
                                                            train=False,
                                                            download=True,
                                                            transform=test_transform)
        elif args.dataset == 'tiny':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            val_dataset = datasets.ImageFolder(os.path.join(args.tinyimagenet_dir, "val"), transform=transform_test)

    # For sanity check
    print("Training data shape: ", train_dataset[0][0].shape)
    os.makedirs('./results', exist_ok=True)
    save_img('./results/test.png',
             torch.stack([d[0] for d in train_dataset]),
             dataname=args.dataset)
    print()

    return train_dataset, val_dataset


class Robust_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = torch.tensor([item.cpu().detach().numpy() for item in images]).detach().cpu().float()
        self.targets = torch.tensor(labels).detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]
    

def load_imagenet(args):

    if args.pretrained:
        args.augment = False
   
    if args.dataset == 'imagenet':

        valdir_blur = os.path.join(f'{args.imagenet_dir}/ImageNet-blur', 'val')
        valdir_invert = os.path.join(f'{args.imagenet_dir}/ImageNet-invert', 'val')
        valdir_noise = os.path.join(f'{args.imagenet_dir}/ImageNet-noise', 'val')

        _, test_transform = transform_imagenet(augment=args.augment,
                                                             from_tensor=False,
                                                             size=args.size,
                                                             rrc=args.rrc)

        val_dataset_blur = ImageFolder(valdir_blur,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)
        val_dataset_invert = ImageFolder(valdir_invert,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)
        val_dataset_noise = ImageFolder(valdir_noise,
                                  test_transform,
                                  nclass=args.nclass,
                                  seed=args.dseed,
                                  load_memory=args.load_memory)
        
        return val_dataset_blur, val_dataset_invert, val_dataset_noise 


def load_cifar(args):
    """Load condensed data from the given path
    """
    if args.pretrained:
        args.augment = False

    if args.dataset[:5] == 'cifar':
        transform_fn = transform_cifar

    _, test_transform = transform_fn(augment=args.augment, from_tensor=True)

    data_noise,target_noise = torch.load(f'{args.processed_cifar_dir}/noise_cifar10_test.pt')
    test_dataset_noise = Robust_Dataset(data_noise,target_noise, test_transform)

    data_blur,target_blur = torch.load(f'{args.processed_cifar_dir}/blur_cifar10_test.pt')
    test_dataset_blur = Robust_Dataset(data_blur,target_blur, test_transform)

    data_invert,target_invert = torch.load(f'{args.processed_cifar_dir}/invert_cifar10_test.pt')
    test_dataset_invert = Robust_Dataset(data_invert,target_invert, test_transform)

    return test_dataset_noise, test_dataset_blur, test_dataset_invert


def test_data(args,
              train_loader,
              val_loader,
              test_resnet=False,
              model_fn=None,
              repeat=1,
              logger=print,
              num_val=4):
    """Train neural networks on condensed data
    """
    repeat = args.repeat

    args.epoch_print_freq = args.epochs // num_val

    if model_fn is None:
        model_fn_ls = [define_model]
        if test_resnet:
            model_fn_ls = [resnet10_bn]
    else:
        model_fn_ls = [model_fn]
    
    criterion = nn.CrossEntropyLoss().cuda()

    for model_fn in model_fn_ls:
        
        best_acc_l = []
        acc_l = []
        mask_acc_5 = []
        mask_acc_l = []
        robust_acc_5_ = []
        robust_acc_l_ = []
        mask_acc_5_ = []
        mask_acc_l_ = []
        best_acc_l = []
        acc_l = []
        min_cluster_acc = []
        mean_cluster_acc = []
        for it in range(repeat):
            model = model_fn(args, args.nclass, logger=logger)
            best_acc, acc = train(args, model, train_loader, val_loader, logger=logger)
            mean, min = cluster_loss(val_loader, model, args, n_clusters = 10, random_state = 0)
            if args.dataset == 'imagenet':
                test_dataset_blur, test_dataset_invert, test_dataset_noise = load_imagenet(args)

                val_loader_robust = MultiEpochsDataLoader(test_dataset_noise,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
            
                acc1_mask, acc5_mask, _ = validate(args, val_loader_robust, model, criterion, it, logger)

                val_loader_robust = MultiEpochsDataLoader(test_dataset_blur,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
            
                acc1_, acc5_, _ = validate(args, val_loader_robust, model, criterion, it, logger)

                val_loader_robust = MultiEpochsDataLoader(test_dataset_invert,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
            
                acc1_mask_, acc5_mask_, _ = validate(args, val_loader_robust, model, criterion, it, logger)

            else:

                test_dataset_noise, test_dataset_blur, test_dataset_invert = load_cifar(args)

                val_loader_robust = MultiEpochsDataLoader(test_dataset_noise,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
            
                acc1_mask, acc5_mask, _ = validate(args, val_loader_robust, model, criterion, it, logger)

                val_loader_robust = MultiEpochsDataLoader(test_dataset_blur,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
            
                acc1_, acc5_, _ = validate(args, val_loader_robust, model, criterion, it, logger)

                val_loader_robust = MultiEpochsDataLoader(test_dataset_invert,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)
            
                acc1_mask_, acc5_mask_, _ = validate(args, val_loader_robust, model, criterion, it, logger)

            min_cluster_acc.append(min)
            mean_cluster_acc.append(mean)

            best_acc_l.append(best_acc)
            acc_l.append(acc)

            mask_acc_l.append(acc1_mask)
            mask_acc_5.append(acc5_mask)

            robust_acc_l_.append(acc1_)
            robust_acc_5_.append(acc5_)

            mask_acc_l_.append(acc1_mask_)
            mask_acc_5_.append(acc5_mask_)
        logger(
            f'Repeat {repeat} =>[MEAN] Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}\n')
        logger(
            f'Repeat {repeat} =>[BEST] Best, last acc: {np.max(best_acc_l):.1f} {np.max(acc_l):.1f}\n')

        logger(
            f'[ROBUST]Repeat {repeat} =>[MEAN] Current noise accuracy (top-1 and 5): {np.mean(mask_acc_l):.1f} {np.mean(mask_acc_5):.1f}\n')
        logger(
            f'[ROBUST]Repeat {repeat} =>[BEST] Best noise accuracy (top-1 and 5): {np.max(mask_acc_l):.1f} {np.max(mask_acc_5):.1f}\n')
        logger(
            f'[ROBUST]Repeat {repeat} =>[MEAN] Current blur accuracy (top-1 and 5): {np.mean(robust_acc_l_):.1f} {np.mean(robust_acc_5_):.1f}\n')
        logger(
            f'[ROBUST]Repeat {repeat} =>[BEST] Best blur accuracy (top-1 and 5): {np.max(robust_acc_l_):.1f} {np.max(robust_acc_5_):.1f}\n')
        logger(
            f'[ROBUST]Repeat {repeat} =>[MEAN] Current invert accuracy (top-1 and 5): {np.mean(mask_acc_l_):.1f} {np.mean(mask_acc_5_):.1f}\n')
        logger(
            f'[ROBUST]Repeat {repeat} =>[BEST] Best invert accuracy (top-1 and 5): {np.max(mask_acc_l_):.1f} {np.max(mask_acc_5_):.1f}\n')

        logger(
            f'Repeat {repeat} =>[BEST] min cluster acc, mean cluster acc: {(np.max(min_cluster_acc)*100):.1f} {(np.max(mean_cluster_acc)*100):.1f}\n')


def cluster_loss(dataloader, net, args, n_clusters = 10, random_state = 0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    features = []
    labels = []

    for batch_data, batch_labels in dataloader:
        features.append(batch_data)
        labels.append(batch_labels)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=-1)

    features = (features.view(features.size(0), -1))
    clusters = kmeans.fit_predict(features.detach().numpy())

    cluster_precisions = []
    for i in range(n_clusters):
        cluster_indices = torch.nonzero(torch.from_numpy(clusters) == i).squeeze()
        cluster_data = features[cluster_indices]
        cluster_labels = (torch.tensor(labels)[cluster_indices]).to(device)
  
        if args.dataset == 'cifar10':
            cluster_data = cluster_data.view(-1, 3, 32, 32)
        elif args.dataset.startswith("imagenet"):
            cluster_data = cluster_data.view(-1, 3, 128, 128)

        cluster_data = cluster_data.to(device)
        outputs = model(cluster_data)
        _, predicted = torch.max(outputs, 1)

        precision = (predicted == cluster_labels).sum().item() / cluster_labels.size(0)
        cluster_precisions.append(precision)

        mean = np.mean(cluster_precisions)
        min = np.min(cluster_precisions)
    
    return mean, min


if __name__ == '__main__':
    from argument import args
    import torch.backends.cudnn as cudnn
    import numpy as np
    cudnn.benchmark = True

    if args.same_compute and args.factor > 1:
        args.epochs = int(args.epochs / args.factor**2)

    path_list = return_data_path(args)
    for p in path_list:
        #init path
        args.save_dir = os.path.join(DATA_PATH, p)
        if args.slct_type == 'herding':
            train_dataset, val_dataset = herding(args)
        else:
      
            train_dataset, val_dataset = load_data_path(args)

        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.workers if args.augment else 0,
                                             persistent_workers=args.augment > 0)
        val_loader = MultiEpochsDataLoader(val_dataset,
                                           batch_size=args.batch_size // 2,
                                           shuffle=False,
                                           persistent_workers=True,
                                           num_workers=4)

        test_data(args, train_loader, val_loader, repeat=args.repeat, test_resnet=False)
