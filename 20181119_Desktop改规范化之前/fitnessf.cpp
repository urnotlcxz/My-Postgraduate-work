/*
CEC17 Test Function Suite for Single Objective Optimization
Noor Awad (email: noor0029@ntu.edu.sg)
Sep. 10th 2016
*/
#include "stdafx.h"
#include <WINDOWS.H>
#include <stdio.h>
#include <math.h>
#include <malloc.h>


void cec17_test_func(double *, double *, int, int, int);

extern double *OShift, *M, *y, *z, *x_bound;
extern int ini_flag, n_flag, func_flag, *SS;
extern int fitcount;

void fitnessf(double *one_x, int positionDim, int func_num, double *one_f)
{
	int i, j, k, n, m;
	double *x, *f;
	FILE *fpt;
	char FileName[30];
	m = 1;    //一次放几个x计算
	n = positionDim;   //维数
	x = (double *)malloc(m*n * sizeof(double));
	f = (double *)malloc(sizeof(double)  *  m);

	for (j = 0; j < n; j++)    //维数
	{
		x[0 * n + j] = one_x[j];
		/*printf("%Lf\n",x[1*n+j]);*/
	}

	cec17_test_func(x, f, n, m, func_num);
	fitcount = fitcount + 1;
	for (j = 0; j < m; j++)  //m个x放进去算
	{
		//  printf(" f%d(x[%d]) = %lf,",func_num,j+1,f[j]);
		*one_f = f[j];
	}
	//  printf("\n");

	free(x);
	free(f);

}


