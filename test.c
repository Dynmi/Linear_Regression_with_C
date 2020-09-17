#include <stdio.h>
#include <string.h>
#include "libs.h"

#define batch_size 8
#define N 49600

float X[N][N_FEATS];
float Y[N];

void inline _train(float *batch_x, float *batch_y, float *W, float lr)
{
    float *grads = malloc(sizeof(float)*N_FEATS);
    cal_gradients(W, batch_x, batch_y, grads, batch_size);
    update_params(W, grads, lr);
    free(grads);
}

float _evaluate(float *X, float *Y, float *W, int size)
{
    float *y_pred = malloc(sizeof(float)*size);
    for(int i=0; i<size; i++)
    {
        y_pred[i] = predict(W,(X+i*N_FEATS));
    }
    float mse_error = get_mse_error(y_pred, Y, size);
    free(y_pred);
    return mse_error;
}

int main(void)
{
    /**
     * initialize all the weights
     * */
    float W[N_FEATS];
    init_params(W);
    
    /**
     * generate dataset
     * */
    for(int p=0; p<N; p++)
    {
        for(int i=0; i<N_FEATS; i++)
        {
            X[p][i] = (rand()%100)*1.0 / 100.0;
            Y[p]+=X[p][i];
        }
    }

    int     idx=0,
            cycles = 400;
    float   batch_x[batch_size][N_FEATS];
    float   batch_y[batch_size];

    printf("Before training, the weights is: ");
    for(int i=0; i<N_FEATS; i++)
    {
        printf("%.4f ",W[i]);    
    }
    printf("\n"); 

    printf("Step -1: mse_error on train_set is %.4f\n",
            _evaluate(&X[0][0], &Y[0], &W[0], N-1));
    for(int step=0; step<cycles; step++)
    {

        for(int cur=idx; cur<idx+batch_size; cur++)
        {
            for(int i=0;i<N_FEATS;i++)
            {
                batch_x[cur-idx][i] = X[idx][i];
            }
            batch_y[cur-idx] = Y[idx];
        }
        idx = (idx+batch_size)%N;

        _train(&batch_x[0][0], &batch_y[0], &W[0], 0.1);
        printf("Step %d: mse_error on batch data is %.4f, on training set is %.4f\n",
                step+1,
                _evaluate(&batch_x[0][0], &batch_y[0], &W[0], batch_size),
                _evaluate(&X[0][0], &Y[0], &W[0], N-1));
    }

    printf("After training, the weights is: ");
    for(int i=0; i<N_FEATS; i++)
    {
        printf("%.4f ",W[i]);    
    }
    printf("\n");  

    return 0;
}