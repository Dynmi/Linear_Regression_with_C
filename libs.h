#include <stdio.h>
#include <stdlib.h>

#define N_FEATS 20

void init_params(float *W)
{
    for(int i=0; i<N_FEATS; i++)
    {
        W[i] = (rand()%200 -100)*1.0 / 100.0;
    }
}


float predict(float *W, float *x)
{
    float y_pred = 0;
    for(int i=0; i<N_FEATS; i++)
    {
        y_pred+=W[i]*x[i];
    }

    return y_pred;
}


float get_mse_error(float *y_pred, float *y_true, int size)
{   
    float error = 0;
    for(int i=0; i<size; i++)
    {
        error += (y_pred[i]-y_true[i])*(y_pred[i]-y_true[i]);
    }
    error/=size;

    return error;
}

/**
 *  This is the key function of this algorithm
 * */
void cal_gradients(float *weights, float *X, float *Y, float *gradients, int batch_size)
{
    float y_preds;
    for(int p=0; p<batch_size; p++)
    {   
        y_preds=predict(weights, (X+p*N_FEATS));
        for(int i=0; i<N_FEATS; i++)
        {
            gradients[i]+=2 * (*(X+p*N_FEATS+i)) * (y_preds-Y[p]);
        }
    }

    for(int i=0; i<N_FEATS; i++)
    {
        gradients[i] /= batch_size;
    }

}


void update_params(float *W, float *grads, float a)
{
    /**
     * W          array of weights
     * gradients  gradients corresponding to each w in W
     * a          learning rate
     **/
    for(int i=0; i<N_FEATS; i++)
    {
        W[i] = W[i]-a*grads[i];
    }
}
