#!/bin/python
import argparse

def main():
    
    parser = argparse.ArgumentParser(
                    description='Simulates the dimensional reduction/upscaling for Conv2d and ConvTransposed2d layers.')

    parser.add_argument('-b', '--batch-size',    dest="batch_size",  default=1,                  type=int, help="")
    parser.add_argument('-i', '--initial-dim',   dest="initial_dims",default=[1, 28, 28],        type=int, nargs='+', help="")
    parser.add_argument('-d', '--hidden-dims',   dest="hidden_dims", default=[32, 64, 128, 256], type=int, nargs='+', help="")
    parser.add_argument('-s', '--stride ',       dest="stride",      default=[2],                  type=int, nargs='+', help="")
    parser.add_argument('-p', '--padding',       dest="padding",     default=[1],                  type=int, nargs='+', help="")
    parser.add_argument('-op','--output-padding',dest="out_padding", default=[0],                  type=int, nargs='+', help="")
    parser.add_argument('-k', '--kernel-size',   dest="kernel_size", default=[3],                  type=int, nargs='+', help="")
    parser.add_argument('-l', '--layers',        dest="layers",      default="cccc",             type=str, help="")

    args = parser.parse_args()

    batch_size  = args.batch_size
    initial_dims= args.initial_dims
    hidden_dims = args.hidden_dims
    stride      = args.stride
    padding     = args.padding
    kernel_size = args.kernel_size
    out_padding = args.out_padding
    layers      = args.layers
    n = len(hidden_dims)
    if len(stride)==1:
        stride = stride * n
    if len(padding)==1:
        padding = padding * n
    if len(kernel_size)==1:
        kernel_size = kernel_size * n
    if len(out_padding)==1:
        out_padding = out_padding * n
    print("Starting...\n")

    header = " id,   b,   c,   w,   h"
    explanation = "b: batch size\nc: channels\nw: width\nh: height\n"
    fstring = "%3d, %3d, %3d, %3d, %3d" 
    print(explanation)
    print(header)
    w = initial_dims[1]
    h = initial_dims[2]
    print(fstring % (0, batch_size, *initial_dims))
    idx = 0
    for i, (t, h_dim, s, p, k, op) in enumerate(zip(layers, hidden_dims, stride, padding, kernel_size, out_padding)):
        idx += 1
        if t=="c":
            w = (w - k + 2 * p) / s + 1
            h = (h - k + 2 * p) / s + 1
        elif t=="t":
            w = (w - 1) * s - 2 * p + k + op
            h = (h - 1) * s - 2 * p + k + op
        print(fstring % (idx, batch_size, h_dim, w, h))

    print("\nFinished!")


if __name__=="__main__":
    main()