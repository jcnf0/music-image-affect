import os
import sys
import math
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

from models import CVAE, CVAEGAN, CALI, TripleGAN
from datasets import mnist, svhn, artemis
from datasets.datasets import load_data 
models = {
    'cvae': CVAE,
    'cvaegan': CVAEGAN,
    'cali': CALI,
    'triple_gan': TripleGAN
}

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--size', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--datasize', type=int, default=-1)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # Select GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Load datasets
    if args.dataset == 'mnist':
        datasets = mnist.load_data()
    elif args.dataset == 'svhn':
        datasets = svhn.load_data()
    elif args.dataset == 'artemis':
        datasets = artemis.load_data
    
    else:
        datasets = load_data(args.dataset,size=args.size)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    if args.dataset == 'data64.hdf5':
        model = models[args.model](
            input_shape=(64,64,3),
            num_attrs=2,
            z_dims=args.zdims,
            output=args.output
        )
    
    elif args.dataset == 'data64_discrete.hdf5':
        model = models[args.model](
            input_shape=(64,64,3),
            num_attrs=9,
            z_dims=args.zdims,
            output=args.output
        )
        
    elif args.dataset == 'data128.hdf5':
        model = models[args.model](
            input_shape=(128,128,3),
            num_attrs=2,
            z_dims=args.zdims,
            output=args.output
        )

    else :
        model = models[args.model](
            input_shape=(256,256,3),
            num_attrs=2,
            z_dims=args.zdims,
            output=args.output
        )

    if args.resume is not None:
        model.load_model(args.resume)
    # Training loop
    datasets.images = datasets.images * 2.0 - 1.0
    datasets.labels = (datasets.attrs+1)/2
    
    samples = np.random.normal(size=(10, args.zdims)).astype(np.float32)
    model.main_loop(datasets, samples, datasets.attr_names,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc', 'c_loss', 'ae_loss'])

if __name__ == '__main__':
    main()
