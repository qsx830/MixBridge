from cleanfid import fid
from cleanfid.inception_torchscript import InceptionV3W
import torch

def myfid(dir1, dir2):
    device = torch.device('cuda:0')
    model = InceptionV3W(path='InceptionV3W-model', download=False, resize_inside=False).to(device)
    model.eval()
    model = torch.nn.DataParallel(model)
    def model_fn(x): return model(x)

    return fid.compute_fid(dir1, dir2, custom_feat_extractor=model_fn)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Missing Parameters")
        sys.exit(1)

    input_str = sys.argv[1]
    print('fid:', myfid(input_str+"clean", input_str+"recon"))