#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "dubnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
tensor forward_maxpool_layer(layer *l, tensor x)
{
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = tensor_copy(x);

    assert(x.n == 4);

    tensor y = tensor_vmake(4,
        x.size[0],  // same # data points and # of channels (N and C)
        x.size[1],
        (x.size[2]-1)/l->stride + 1, // H and W scaled based on stride
        (x.size[3]-1)/l->stride + 1);

    // This might be a useful offset...
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int out_c = x.size[1];
    int out_h = (x.size[2]-1)/l->stride + 1;
    int out_w = (x.size[3]-1)/l->stride + 1;
    int n, c, h, w, i, j;
    for (n = 0; n < x.size[0]; ++n) {
        for (c = 0; c < out_c; ++c) {
            for (h = 0; h < out_h; ++h) {
                for (w = 0; w < out_w; ++w) {
                    int index_1d = w + out_w * (h + out_h *(c + out_c*n));
                    float max_value = -FLT_MAX;
                    for (i = 0; i < (int) l->size; ++i) {
                        for (j = 0; j < (int) l->size; ++j) {
                            int layer_h = pad + h*l->stride + i;
                            int layer_w = pad + w*l->stride + j;
                            int index = layer_w + x.size[3]*(layer_h + x.size[2]*(c + n*x.size[1]));
                            float val = x.data[index];
                            if (layer_h < 0 || layer_h >= x.size[2] || layer_w < 0 || layer_w >= x.size[3]) {
                                val = -FLT_MAX;
                            }
                            max_value = (val > max_value) ? val   : max_value;
                        }
                    }
                    y.data[index_1d] = max_value;
                }
            }
        }
    }

    return y;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
tensor backward_maxpool_layer(layer *l, tensor dy)
{
    tensor x    = l->x;
    tensor dx = tensor_make(x.n, x.size);
    int pad = -((int) l->size - 1)/2;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int out_c = x.size[1];
    int out_h = (x.size[2]-1)/l->stride + 1;
    int out_w = (x.size[3]-1)/l->stride + 1;
    int n, c, h, w, i, j;
    for (n = 0; n < x.size[0]; ++n) {
        for (c = 0; c < out_c; ++c) {
            for (h = 0; h < out_h; ++h) {
                for (w = 0; w < out_w; ++w) {
                    int index_1d = w + out_w * (h + out_h *(c + out_c*n));
                    float max_value = -FLT_MAX;
                    int max_index = -1;
                    for (i = 0; i < (int) l->size; ++i) {
                        for (j = 0; j < (int) l->size; ++j) {
                            int layer_h = pad + h*l->stride + i;
                            int layer_w = pad + w*l->stride + j;
                            int index = layer_w + x.size[3]*(layer_h + x.size[2]*(c + n*x.size[1]));
                            float val = x.data[index];
                            if (layer_h < 0 || layer_h >= x.size[2] || layer_w < 0 || layer_w >= x.size[3]) {
                                val = -FLT_MAX;
                            }
                            if (val > max_value) {
                                max_value = val;
                                max_index = index;
                            }
                        }
                    }
                    dx.data[max_index] += dy.data[index_1d];
                }
            }
        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer *l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(size_t size, size_t stride)
{
    layer l = {0};
    l.size = size;
    l.stride = stride;
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

