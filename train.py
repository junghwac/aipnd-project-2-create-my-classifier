'''
# PROGRAMMER: Junghwa C.
# DATE CREATED: 2020-10-16
# REVISED DATE: 2020-10-19

Train a new network on a data set with train.py
• Basic usage: python train.py data_directory
• Prints out training loss, validation loss, and validation accuracy as the network trains
• Options:
    • Set directory to save checkpoints: 
        python train.py data_dir --save_dir save_directory
    • Choose architecture: vgg or resnet
        python train.py data_dir --arch vgg
    • Set hyperparameters: 
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    • Choose processor: gpu or cpu
        python train.py data_dir --device gpu

Example call: python train.py flowers --device gpu               
'''
import argparse
from time import time
from classifier import classifier

def get_input_args_train():

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help = 'the data folder')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--arch', type=str, default='vgg')
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help = 'directory to save a trained model to and load from')
     
    return parser.parse_args()

def main():
    
    start_time = time()
    
    in_arg = get_input_args_train()
    classifier(in_arg.data_dir, in_arg.device, in_arg.arch, in_arg.hidden_units, in_arg.learning_rate, in_arg.epochs, in_arg.save_dir)
    
    end_time = time()
    tot_time = end_time - start_time
    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" + \
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + \
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )
    
if __name__ == '__main__':
    main()
    
    