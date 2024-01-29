import argparse
import os
import numpy as np
import time
from model.coanet import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid #, save_image
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from utils.loss import SegmentationLosses
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_image(tensor, filename, nrow=8, padding=0,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    #im = im.resize([1300, 1300])
    im.save(filename)

def main():
    parser = argparse.ArgumentParser(description="PyTorch CoANet Training")
    parser.add_argument('--out-path', type=str, default='./img/',
                        help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default="./weigth/model_best.pth.tar",
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='arcade',
                        help='dataset name')
    # parser.add_argument('--base-size', type=int, default=512,
                        # help='base image size. spacenet:1280, DeepGlobe:1024.')
    # parser.add_argument('--crop-size', type=int, default=512,
                        # help='crop image size. spacenet:1280, DeepGlobe:1024.')
    # parser.add_argument('--sync-bn', type=bool, default=False,
                        # help='whether to use sync bn')
    # parser.add_argument('--freeze-bn', type=bool, default=False,
                        # help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    kwargs = {'num_workers': args.workers, 'pin_memory': False}
    torch.manual_seed(args.seed)
    train_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    model = CoANet(num_classes=1,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    model = model.cuda()
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])
    out_path = os.path.join(args.out_path, 'result/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    evaluator = Evaluator(2)
    model.eval()
    evaluator.reset()
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target = sample[0]['image'], sample[0]['label']
        img_name = sample[1][0].split('.')[0]
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output, out_connect, out_connect_d1 = model(image)

        target = torch.unsqueeze(target, 1)

        target_n = target.cpu().numpy()
        pred = output.data.cpu().numpy()
        pred[pred > 0.4] = 1
        pred[pred < 0.4] = 0

        evaluator.add_batch(target_n, pred.astype(int))#

        out_image = make_grid(image.clone().cpu().data, 3, normalize=True,padding=0)
        out_GT = make_grid(decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(),
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255),padding=0)
        out_pred_label_sum = make_grid(decode_seg_map_sequence(np.squeeze(pred, 1),
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255),padding=0)
        num=int(out_image.shape[2]/512)

        for i in range(num):
            save_image(out_image[:,:,i*512:(i+1)*512], out_path + str(int(img_name)+i) + '_.png')
            save_image(out_GT[:,:,i*512:(i+1)*512], out_path + str(int(img_name)+i) + '_GT' + '.png')
            save_image(out_pred_label_sum[:,:,i*512:(i+1)*512], out_path + str(int(img_name)+i) + '_pred' + '.png')

    # Fast test during the training
    mIoU = evaluator.Mean_Intersection_over_Union()
    IoU = evaluator.Intersection_over_Union()
    dice = evaluator.Dice_coefficient()  
    print('Validation:')
    print('[numImages: %5d]' % (i * args.batch_size + image.data.shape[0]))
    print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}, dice:{}"
          .format(0, 0, mIoU, IoU, 0, 0, 0, dice))

if __name__ == "__main__":
   main()