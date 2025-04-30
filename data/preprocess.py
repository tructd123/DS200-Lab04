import os, argparse, pickle
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to dataset/training or dataset/test')
parser.add_argument('--output', required=True, help='Output folder for batches')
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

os.makedirs(args.output, exist_ok=True)
classes = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input,d))])

images, labels = [], []
batch_idx = 0

for cls_idx, cls in enumerate(classes):
    cls_dir = os.path.join(args.input, cls)
    for fn in tqdm(os.listdir(cls_dir), desc=f'Preproc {cls}'):
        path = os.path.join(cls_dir, fn)
        try:
            img = Image.open(path).convert('RGB')
        except:
            continue
        tensor = tf(img).flatten().tolist()
        images.append(tensor)
        labels.append(cls_idx)
        if len(images) == args.batch_size:
            with open(f"{args.output}/batch_{batch_idx}.pkl","wb") as f:
                pickle.dump({'data': images,'labels': labels}, f)
            images, labels = [], []
            batch_idx += 1

# lưu phần dư cuối cùng
if images:
    with open(f"{args.output}/batch_{batch_idx}.pkl","wb") as f:
        pickle.dump({'data': images,'labels': labels}, f)