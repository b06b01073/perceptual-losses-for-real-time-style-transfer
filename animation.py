import argparse 
import torch
from tqdm import tqdm

from model import TransformNet_V2
import utils
from torchvision.utils import save_image
from torchvision import transforms 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    frames, fps = utils.read_video(args.input) # (frames, h, w, c)
    frames = torch.permute(frames, (0, 3, 1, 2)).float().to(device) # (frames, c, h, w)
    frames /= 255
    batch_size = args.batch_size
    batch = frames.shape[0] // batch_size
    if frames.shape[0] % batch_size != 0:
        batch += 1

    transform_net = TransformNet_V2().to(device)
    utils.load_model(transform_net, args.model_path)
    transform_net.eval()
    output = []

    with torch.no_grad():
        for i in tqdm(range(batch)):
            index = i * batch_size
            if i != batch - 1:
                input = frames[index:index+batch_size]
            else:
                input = frames[index:]
            input = transforms.Resize((256, 256))(input)
            stylized_frames = transform_net(input) 
            output.append(stylized_frames)

    output = torch.concat(output, dim=0)
    utils.save_gif(output, args.output_path, fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./video/dance.mp4')
    parser.add_argument('--model_path', type=str, default='./model_params/night/epoch3.pth')
    parser.add_argument('--output_format', type=str, default='gif')
    parser.add_argument('--output_path', type=str, default='./output/videos/dance.gif')
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    main(args)