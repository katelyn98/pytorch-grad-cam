import argparse
import cv2
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random

def random_choice(selected_image):
  imagelist = ['dog','parrot','frog','tiger','bee','kingSnake','jellyfish','turtle','starfish',
              'mudTurtle','seaAnemone','pineapple','strawberry','pomegranate','fig','broccoli',
              'corn','artichoke','bellPepper','cucumber','violin','piano','acousticGuitar','flute',
              'harmonica','bicycle','car','train','canoe','tractor','warplane','laptop','electricFan',
              'camera']

  selected_image = "tiger"
  random_choice = selected_image

  while random_choice == selected_image:
    random_choice = random.choice(imagelist)

  return random_choice

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    # model = models.resnet50(pretrained=True)
    model = models.googlenet(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    # target_layers = [model.layer4]
    target_layers = [model.inception5b.branch4]

    image_name = args.image_path
    image_name = image_name.split('/')[8]
    image_name = image_name.split('.')[0]
    print("IMAGE NAME: " + image_name)
    random_name = random_choice(image_name)
    print("RANDOM NAME: " + random_name)
    path="./Documents/GitHub/eyeintoai-code/frontend/src/images/samples/"

    rand_img = cv2.imread(path+random_name+".jpg", 1)[:, :, ::-1]
    rand_img = np.float32(rand_img) / 255
    rand_tensor = preprocess_image(rand_img,
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    image = cv2.imread(args.image_path)
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        # cam.batch_size = 1
        grayscale_cam = cam(input_tensor=rand_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    mask_2 = cam_mask > np.percentile(cam_mask, 85)
    mask2_img = mask_2.astype(np.uint8)  #convert to an unsigned byte
    mask2_img*=255

    mask_cv2 = cv2.cvtColor(mask2_img, cv2.COLOR_BGR2GRAY)
    print(mask2_img.shape)
    
    print(image.shape)
    masked = cv2.bitwise_and(image, image, mask=mask_cv2)

    cv2.imwrite("masked_test.jpg", mask2_img)
    cv2.imwrite("masked_image.jpg", masked)
    cv2.imwrite('grad_cam.jpg', cam_image)
