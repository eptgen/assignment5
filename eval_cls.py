import numpy as np
import argparse

import imageio
import pytorch3d
import torch
from models import cls_model
from utils import create_dir, get_points_renderer

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(args.num_cls_class)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    N = 10000
    replace = False
    # UNCOMMENT BELOW TWO LINES FOR NPOINTS TEST
    # N = 10
    # replace = True
    ind = np.random.choice(N,args.num_points, replace=replace)
    num_obj, _, _ = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device).shape
    batch_size = 32
    test_label = torch.from_numpy(np.load(args.test_label)).to(args.device)
    pred_label = torch.zeros(num_obj).to(args.device)
    point_color = torch.asarray([0.7, 0.0, 0.0]).unsqueeze(0).to(args.device)
    # UNCOMMENT FOR ROTATION TEST
    # rot = pytorch3d.transforms.euler_angles_to_matrix(torch.asarray([45, 0, 45]), "XYZ").to(args.device)
    # print("rot", rot)
    
    for batch in range(0, num_obj, batch_size):
        test_data = torch.from_numpy((np.load(args.test_data))[batch:batch+batch_size,ind,:]).to(args.device)
        # UNCOMMENT FOR ROTATION TEST
        # test_data = test_data @ rot

        # ------ TO DO: Make Prediction ------
        _, pred_label[batch:batch+batch_size] = model(test_data).max(dim = 1)
        i = min(num_obj - 1, batch)
        print("item", i, ": predicted", pred_label[i], ", actually", test_label.data[i])
        renderer = get_points_renderer(device = args.device)
        points = test_data[i - batch]
        rgba = torch.ones_like(points).to(args.device) * point_color
        pointcloud = pytorch3d.structures.Pointclouds(
            points=points.unsqueeze(0),
            features=rgba.unsqueeze(0),
        ).to(args.device)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=args.device)

        rends = []
        num_povs = 15
        for j in range(num_povs):
            theta = 360 * (j / num_povs)
            R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist = 3., azim = theta, elev = 30)
            # Prepare the camera:
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R, T=T, fov=60, device=args.device
            )

            rend = renderer(pointcloud, cameras=cameras, lights=lights)
            rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            rends.append((rend * 255).astype(np.uint8))
        imageio.mimsave(f"{args.output_dir}/rend_{i}.gif", rends, fps = 15, loop = 0)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

