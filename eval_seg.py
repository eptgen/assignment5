import numpy as np
import argparse

import pytorch3d
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes = args.num_seg_class)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    N = 10000
    replace = True
    # UNCOMMENT BELOW TWO LINES FOR NPOINTS TEST
    # N = 10
    # replace = True
    ind = np.random.choice(N,args.num_points, replace=replace)
    num_obj, _, _ = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device).shape
    # print("num_obj", num_obj)
    batch_size = 16
    test_data = torch.zeros(num_obj, args.num_points, 3).to(args.device)
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind]).to(args.device)
    pred_label = torch.zeros(num_obj, args.num_points).to(args.device)
    # UNCOMMENT FOR ROTATION TEST
    # rot = pytorch3d.transforms.euler_angles_to_matrix(torch.asarray([45, 0, 45]), "XYZ").to(args.device)
    
    for batch in range(0, num_obj, batch_size):
        test_data[batch:batch+batch_size] = torch.from_numpy((np.load(args.test_data))[batch:batch+batch_size,ind,:]).to(args.device)
        # UNCOMMENT FOR ROTATION TEST
        # test_data = test_data @ rot

        # ------ TO DO: Make Prediction ------
        _, pred_label[batch:batch+batch_size] = model(test_data[batch:batch+batch_size]).max(dim = 2)

    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    ith_accuracy = pred_label[args.i].eq(test_label[args.i].data).cpu().sum().item() / (test_label[args.i].reshape((-1,1)).size()[0])
    print(f"accuracy of {args.i}th object: {ith_accuracy}")

    test_data = test_data.to(args.device)
    test_label = test_label.to(args.device)
    pred_label = pred_label.to(args.device)
    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/seg_gt_{}.gif".format(args.output_dir, args.i), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/seg_pred_{}.gif".format(args.output_dir, args.i), args.device)
