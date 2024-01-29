from dataloaders.datasets import spacenet, spacenet_crop, deepglobe, deepglobe_crop_zw
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def make_data_loader(args, **kwargs):

    if args.dataset == 'DeepGlobe':
        train_set = deepglobe_crop_zw.Segmentation(args, split='train')
        val_set = deepglobe_crop_zw.Segmentation(args, split='val')
        #test_set = deepglobe.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, num_class

    else:
        raise NotImplementedError

