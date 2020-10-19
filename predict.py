'''
# PROGRAMMER: Junghwa C.
# DATE CREATED: 2020-10-16
# REVISED DATE: 2020-10-19

Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
• Basic usage: python predict.py /path/to/image checkpoint
• Options:
    • Return top K most likely classes: 
        python predict.py input checkpoint --top_k 3
    • Use a mapping of categories to real names: 
        python predict.py input checkpoint --category_names cat_to_name.json
    • Choose a processor for inference: cpu or gpu
        python predict.py input checkpoint --device gpu

Example call: python predict.py flowers/test/1/image_06743.jpg checkpoint --device gpu
'''
import argparse
from PIL import Image
import torch
from torchvision import transforms, models
import json

def get_input_args_predict():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_pth', type=str, help='/path/to/image')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names', type=str, default='cat_to_name.json')
    
    return parser.parse_args()

def predict(img_pth, checkpoint_dir, device, top_k, category_names):

    img_pil = Image.open(img_pth)    
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)
    
    checkpoint_dir = 'checkpoint.pth'
    processors = {'cpu': 'cpu', 'gpu': 'cuda'}
    processor = processors[device]
    
    if processor == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        checkpoint = torch.load(checkpoint_dir) # to load the model on gpu 
    else:
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_dir, map_location=lambda storage, loc: storage) # to load the model on cpu 
   
    if checkpoint['model'] == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.state_dict = checkpoint['state_dict'] 
        model.optimizer = checkpoint['optimizer_dict']
        model.class_to_idx = checkpoint['class_to_idx']        
    else:
        model = models.resnet50(pretrained=True)
        model.fc = checkpoint['fc']
        model.state_dict = checkpoint['state_dict'] 
        model.optimizer = checkpoint['optimizer_dict']
        model.class_to_idx = checkpoint['class_to_idx'] 
    
    model.to(device)
    model = model.eval()
    img_tensor = img_tensor.to(device) # img_tensor.to(device) gets RuntimeError: torch.FloatTensor -> torch.cuda.FloatTensor
    img_tensor = img_tensor.requires_grad_(False)

    logps = model(img_tensor)
    ps = torch.exp(logps)
    top_p, idx = ps.topk(top_k, dim=1)
    top_p = top_p[0].tolist()
    top_p = [round(i, 3) for i in top_p]
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    idx_to_class = [model.idx_to_class[i] for i in idx[0].tolist()]
    class_to_name = [cat_to_name[i] for i in idx_to_class]
    flower_names = [i.title() for i in class_to_name]
          
    print(  dict(  zip(flower_names, top_p)  )  )
    
def main():
    
    in_arg = get_input_args_predict()
    predict(in_arg.image_pth, in_arg.checkpoint, in_arg.device, in_arg.top_k, in_arg.category_names)
          
if __name__ == '__main__':
    main()