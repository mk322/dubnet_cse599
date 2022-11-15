#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "dubnet.h"
#include "matrix.h"

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = xw+b
tensor forward_connected_layer(layer *l, tensor x)
{
    if(x.n > 2){
        x = tensor_vview(x, 2, x.size[0], tensor_len(x)/x.size[0]);
    } else {
        x = tensor_copy(x);
    }
    // Saving our input
    // Probably don't change this
    tensor_free(l->x);
    l->x = x;

    // TODO: 3.0 - run the network forward
    const size_t new_size[2] = {x.size[0], l->w.size[1]};
    tensor y = matrix_multiply(x, l->w);
    for(size_t i = 0; i < new_size[0]; ++i){
        for (size_t j = 0; j < new_size[1]; ++j) {
            y.data[i*new_size[1] + j] += l->b.data[j];
        }
    }
    return y;
}

// Run a connected layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
tensor backward_connected_layer(layer *l, tensor dy)
{
    tensor x = l->x;

    // TODO: 3.1
    // Calculate the gradient dL/db for the bias terms using backward_bias
    // add this into any stored gradient info already in l.db
    tensor db = tensor_sum_dim(dy, 0);
    tensor_axpy_(1, db, l->db);
    tensor_free(db);


    // Then calculate dL/dw. Use axpy to add this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw
    tensor dw = matrix_multiply(matrix_transpose(x), dy);
    tensor_axpy_(1, dw, l->dw);
    tensor_free(dw);

    // Calculate dL/dx and return it
    tensor dx = matrix_multiply(tensor_copy(dy), matrix_transpose(l->w));
    return dx;
}

// Update weights and biases of connected layer
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_connected_layer(layer *l, float rate, float momentum, float decay)
{
    // TODO: 3.2
    // Apply our updates using our SGD update rule
    // assume  l.dw = dL/dw - momentum * update_prev
    // we want l.dw = dL/dw - momentum * update_prev + decay * w
    // then we update l.w = l.w - rate * l.dw
    // lastly, l.dw is the negative update (-update) but for the next iteration
    // we want it to be (-momentum * update) so we just need to scale it a little
    tensor_axpy_(decay, l->w, l->dw);
    tensor_axpy_(-rate, l->dw, l->w);
    tensor_scale_(momentum, l->dw);

    // Do the same for biases as well but no need to use weight decay on biases
    tensor_axpy_(-rate, l->db, l->b);
    tensor_scale_(momentum, l->db);
}

layer make_connected_layer(int inputs, int outputs)
{
    layer l = {0};
    l.w  = tensor_vrandom(sqrtf(2.f/inputs), 2, inputs, outputs);
    l.dw = tensor_vmake(2, inputs, outputs);
    l.b  = tensor_vmake(2, 1, outputs);
    l.db = tensor_vmake(2, 1, outputs);
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

