from .btcv import BTCV
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def get_dataloader(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_train_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test_seg = transforms.Compose([
        transforms.Resize((args.out_size,args.out_size)),
        transforms.ToTensor(),
    ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        lvis_train_dataset = BTCV(args, args.data_path, transform = transform_train, transform_msk= transform_train_seg, mode = 'Training', prompt=args.prompt)
        lvis_test_dataset = BTCV(args, args.data_path, transform = transform_test, transform_msk= transform_test_seg, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(lvis_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(lvis_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader