#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <math.h>
#include <string.h>
#define nn_num 200
#define benchmark_number 10
#define input_num 30
#define hidden_num 40
#define output_num 10
#define positionDim 2
#define section 10
#define total_weight 1600

#define MAX_GENERATION 200 //���ٴ�GA
#define initial_popsize 1000
#define num_recurrent 100   //ѡ100�Σ�100���µ�
#define all_recurrent 30    //����ѭ��90�Σ�90*100=9000���µ�  ��30�Ͳ�����
#define final_popsize initial_popsize+num_recurrent*all_recurrent

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
int fitcount;

double neural_network[nn_num][total_weight];
double nn_fitness[nn_num][benchmark_number];

void initial_NN()    //��ʼ��nn_num���������Ȩֵ
{
	int nn, i;

	//   srand((unsigned)time(NULL));
	for (nn = 0; nn < nn_num; nn++)
	{
		for (i = 0; i < total_weight; i++)
		{
			neural_network[nn][i] = (double)rand() / RAND_MAX * 2 - 1.0;   //�������ʼ��Ȩֵ��-1,1��																		   //	printf("%f\n",neural_network[nn][i]);
		}
	}
}
void swap(double *a, double *b)  //����2����
{
	double temp;
	temp = *a;
	*a = *b;
	*b = temp;
}
void adjust(double* arr, int idx1, int idx2)
{

	if (nullptr == arr || idx1 >= idx2 || idx1 < 0 || idx2 < 0)
		return;
	double tmp = arr[idx1];  //��ʱ���Ҫ����������
	for (int idx = idx1 * 2 + 1; idx <= idx2; idx = idx * 2 + 1)  //��Ҫ���������ݵ����ӿ�ʼ�Ƚ�
	{
		//ѡ�����Һ����е�����
		if (idx + 1 <= idx2 && arr[idx] < arr[idx + 1])
			++idx;
		if (arr[idx] > tmp)  //���������ѣ�����
		{
			arr[idx1] = arr[idx];  //�����������ƻ������������ѵ�����
			idx1 = idx;  //��������Ҫ�����ģ���ʱtmp��ʱ����˳�ʼarr[idx1]��ֵ������ÿ�αȽ϶��Ǻ�tmp�Ƚ�,�ñȽ����ˣ����Կ��Բ����Ƚ���
						 //�������µ�����ֱ����������������
		}
		else
			break;
	}
	arr[idx1] = tmp;
}
//�����������±��Ӧ���ڶ����ݽṹ�еĽ��λ�ã���0��ʼ��ţ�������ȫ������
double heapSort(double* arr, int length)
{
	if (nullptr == arr || length <= 0)
		return -1;
	//������˳���ŵ����ݾͶ�Ӧ��ȫ���������еĶ�Ӧ����ֵ�����ڵ���Ϊ�����
	for (int idx = length / 2 - 1; idx >= 0; --idx)  //�����һ����Ҷ�ӽ�㿪ʼ����Ϊ����
	{
		adjust(arr, idx, length - 1);  //���һ����Ҷ�ӽ������ĺ��ӱȽϵ���
	}
	//���򣬸��������һ����㽻��������
	for (int idx = length - 1; idx > 0; --idx)
	{
		swap(&arr[0], &arr[idx]);  //ÿ��ѡ��һ���������ŵ�ĩβ��Ҳ��������ĩβ
		adjust(arr, 0, idx - 1);  //��������㵽idx-1�����Ϊ�����
	}
	return 0;
}


void nn_compute(int section_fitness_sort[][3], double W_inhidden[][hidden_num], double W_hiddenout[][output_num], double every_rate[section])
{
	double x[input_num];
	double hiddenoutputs[hidden_num];
	double newx[output_num];
	double wh_bias[hidden_num];
	double wo_bias[output_num];
	double sigma0, sigma1, SumNewx;
	int i, j;
	// srand((unsigned)time(NULL));
	//����ά�����еı�����ŵ�һά������
	int num = 0;
	for (i = 0; i < 10; i++)
	{
		for (j = 0; j < 3; j++)
		{
			x[num] = section_fitness_sort[i][j] / (double)initial_popsize;
			num++;
		}
	}

	for (i = 0; i < hidden_num; i++)
	{
		sigma0 = 0.0;
		for (j = 0; j < input_num; j++)
		{
			sigma0 = sigma0 + W_inhidden[j][i] * x[j];
		}
		wh_bias[i] = 2.0*rand() / RAND_MAX - 1.0;  // (rand()%100/50.0) - 1.0;
		hiddenoutputs[i] = sigma0 + wh_bias[i];
	}
	for (i = 0; i < output_num; i++)
	{
		sigma1 = 0.0;
		for (j = 0; j < hidden_num; j++)
		{
			sigma1 = sigma1 + W_hiddenout[j][i] * hiddenoutputs[j];
		}

		wo_bias[i] = 2.0*rand() / RAND_MAX - 1.0; //(rand()%100/50.0) - 1.0;
		newx[i] = exp(sigma1 + wo_bias[i]);
	}
	SumNewx = 0;
	for (i = 0; i < output_num; i++)
	{
		SumNewx += newx[i];
	}
	for (i = 0; i < output_num; i++)
	{
		every_rate[i] = newx[i] / SumNewx;
	}
}

void select_action(double sum_rate[10], int *a_section_index)
{
	double r;
	//   srand((unsigned)time(NULL));

	r = rand() / (RAND_MAX + 1.0);      //����0-1֮������С��
	if (r < sum_rate[0])
	{
		a_section_index[0] = 0;
		a_section_index[1] = 1;
	}
	else
	{
		for (int j = 0; j < section - 1; j++)
		{
			if (r >= sum_rate[j] && r < sum_rate[j + 1])
			{
				a_section_index[0] = j + 1;
				a_section_index[1] = j + 2;
			}
		}
	}
}




void select_f_index_draw_rate(double temp_x[][positionDim], double f[], int *one_d_index_x, int d, int index_f[], double W_inhidden[][hidden_num], double W_hiddenout[][output_num], int condition, int every_section, int *one_d_select_index_x, int *select_index_f)
{
	int randsel;
	//	int *index_x = (int(*))malloc(sizeof(one_d_index_x)); //���ǡ�4�� ��Ϊint��4���Ǿ�˵�� ��1������ʵ��ָ��
	int *index_x = (int(*))malloc(condition * sizeof(int));

	int not_select_index_x;
	int index_1, index_2, index_end;   //ȡ����ֵ��ŵĵ�һ��
	int ArrLength;

	int(*a_index_section)[section];  //����  100*10��Ȼ��10*10��every_section*section
	a_index_section = (int(*)[section])malloc(every_section*section * sizeof(int));

	double(*a_section_fit)[section];
	a_section_fit = (double((*)[section]))malloc(every_section*section * sizeof(double));

	int(*a_section_fit_index)[section];
	a_section_fit_index = (int(*)[section])malloc(every_section*section * sizeof(int));

	int section_fit_index_sort[section][3];
	int max_index[section];
	int min_index[section];
	int mid_index[section];

	int index_f_record;
	double temp_f[final_popsize];
	double boundary[section + 1];  //11���߽�ֵ
	double every_rate[section] = { 0 }; //10��ͨ��NN����ó��ĸ���
	double sum_rate[section];
	int a_section_index[2];   //ѡ�е��Ǹ�����ı߽�ֵ��2�����
							  //  srand((unsigned)time(NULL));
	double *temp_section_fit;
	int *temp_section_fit_index;
	double *temp;
	int *one_section_index_f;
	int k;
	int index1, index2, index3;
	int z;

	temp_section_fit = (double(*))malloc(sizeof(double)*every_section);
	temp = (double(*))malloc(sizeof(double)*every_section);
	temp_section_fit_index = (int(*))malloc(sizeof(int)*every_section);
	one_section_index_f = (int(*))malloc(sizeof(int)*every_section);

	memcpy(index_x, one_d_index_x, sizeof(one_d_index_x)*condition);

	if (condition < input_num)
	{
		randsel = rand() % condition + 1;     //�������
		index_1 = index_x[0];       //ȡ����һ�����
		boundary[0] = temp_x[index_1][d];
		for (int i = 0; i < section - 1; i++)  //�м��ٽ�ƽ����
		{
			index_1 = index_x[i];
			index_2 = index_x[i + 1];
			boundary[i + 1] = (temp_x[index_1][d] + temp_x[index_2][d]) / 2;
		}
		index_end = index_x[condition - 1];
		boundary[section] = temp_x[index_end][d];
		not_select_index_x = index_x[randsel];
	}
	//else if (condition%section == 0)
	else
	{
		for (int i = 0; i < section; i++)
		{
			for (int j = 0; j < every_section; j++)
			{
				a_index_section[j][i] = index_x[i * every_section + j]; //λ����ŷֳ�section�����䣬1��������100��/10��/1��

			}
		}
		free(index_x);
		index_x = NULL;
		for (int j = 0; j < condition; j++) {
			index_f_record = index_f[j];
			temp_f[j] = f[index_f_record];
		}

		for (int i = 0; i < section; i++)
		{
			for (int j = 0; j < every_section; j++)
			{
				a_section_fit[j][i] = temp_f[i * every_section + j]; //fit��ŷֳ�section�����䣬1��������100��/10��/1��
				a_section_fit_index[j][i] = index_f[i * every_section + j];
			}
		}

		for (int i = 0; i < section; i++)   //ÿ��ȡfitness�����ֵ����Сֵ����ֵ
		{
			k = 0;
			for (int j = 0; j < every_section; j++)
			{
				temp_section_fit[j] = a_section_fit[j][i];   //a_section_fit�Ƕ�ά��Ҫ��ÿ�������������
				temp[j] = a_section_fit[j][i];
				temp_section_fit_index[j] = a_section_fit_index[j][i]; //һ�������f����
			}

			ArrLength = every_section;  //һ����������
										//quicksort(temp_section_fit, ArrLength, 0, ArrLength - 1);  //��������һ�������ϵ�fitness������
			heapSort(temp_section_fit, ArrLength);
			for (int w = 0; w < ArrLength; w++)
			{
				for (z = 0; z < ArrLength; z++)
				{
					if (temp_section_fit[w] == temp[z])   //����֮��ĺ�ԭ������ȣ�����֪���±�
					{
						one_section_index_f[k] = z;   //��Ӧԭ�����±�
						k++;
						break;
					}
				}
			}

			index1 = one_section_index_f[k - 1];
			index2 = one_section_index_f[0];
			index3 = (int)floor((one_section_index_f[(int)floor(ArrLength / 2)] + one_section_index_f[(int)floor(ArrLength / 2) + 1]) / 2);

			max_index[i] = temp_section_fit_index[index1];
			min_index[i] = temp_section_fit_index[index2];
			mid_index[i] = temp_section_fit_index[index3];

			section_fit_index_sort[i][0] = max_index[i];
			section_fit_index_sort[i][1] = min_index[i];
			section_fit_index_sort[i][2] = mid_index[i];
		}//i 10���������

		free(temp_section_fit);
		free(temp);
		free(temp_section_fit_index);
		free(one_section_index_f);
		free(a_section_fit);

		temp_section_fit = NULL;
		temp = NULL;
		temp_section_fit_index = NULL;
		one_section_index_f = NULL;
		a_section_fit = NULL;
		//a_section_fit.shrink_to_fit();

		boundary[0] = temp_x[a_index_section[0][0]][d];
		for (int i = 0; i < section; i++)
		{
			index_1 = a_index_section[every_section - 1][i];
			index_2 = a_index_section[0][i + 1];
			boundary[i + 1] = (temp_x[index_1][d] + temp_x[index_2][d]) / 2;
		}
		index_end = a_index_section[every_section - 1][section - 1];
		boundary[section] = temp_x[index_end][d];

		//ͨ��NN����ÿ������ĸ���
		nn_compute(section_fit_index_sort, W_inhidden, W_hiddenout, every_rate);

		//�ۼӸ���
		sum_rate[0] = every_rate[0];
		for (int l = 1; l < section; l++)
		{
			sum_rate[l] = sum_rate[l - 1] + every_rate[l];
		}
		//���̶�ѡ���䵽��һ��������
		select_action(sum_rate, a_section_index);

		for (int m = 0; m < every_section; m++)
		{
			one_d_select_index_x[m] = a_index_section[m][a_section_index[0]]; //ȡ����һ������������У������Ժ��ٷ�
			select_index_f[m] = a_section_fit_index[m][a_section_index[0]];
		}
		free(a_index_section);
		free(a_section_fit_index);
		a_index_section = NULL;
		a_section_fit_index = NULL;
	}
	/*	else
	{
	randsel = rand() % condition;     //�������
	one_d_select_index_x[0] = index_x[randsel];
	select_index_f[0] = index_f[randsel];
	free(index_x);
	index_x = NULL;
	}
	*/
}

void fitnessf(double *, int, int, double *);
double dif_nn(double W_inhidden[][hidden_num], double W_hiddenout[][output_num], int FUNNUM)
{
	double INIT_MIN;
	double INIT_MAX;

	//	int all_recurrent = 20;    //����ѭ��20�Σ�20*100=2000���µ�
	int ArrLength;
	ArrLength = initial_popsize;
	int ArrLength_f;

	ArrLength_f = initial_popsize;
	//	int row;
	int every_section = 100;

	int condition;
	double x[final_popsize][positionDim];
	double temp_x[final_popsize][positionDim];

	double *one_d_x;               //һ��ά���ϵ�x��¼����
	int index_x[final_popsize][positionDim];  //��ά�Ϸֱ�������֮����������±꣩
											  //int re_index_x[final_popsize][positionDim];
	double *temp_test;
	int *one_d_index_x;
	//	double sort_X[final_popsize][positionDim];

	double f[final_popsize];
	double *one_d_f;
	int *index_f;
	int re_index_f[final_popsize];
	double *temp_f;
	int *one_d_index_f;

	int *one_d_select_index_x; //Ҫ���ص�x���������
	int *select_index_f;
	double sel_new_x;
	double add_position[num_recurrent][positionDim];
	int new_x_flag = 0;;

	double nn_best_fit;
	double add_one_f;
	double one_x[positionDim];
	double one_x_add_position[positionDim];
	double one_f;

	switch (FUNNUM)
	{
	case 1:INIT_MIN = -100; INIT_MAX = 100; break;
	case 2:INIT_MIN = -100; INIT_MAX = 100; break;
	case 3:INIT_MIN = -100; INIT_MAX = 100; break;
	case 4:INIT_MIN = -100; INIT_MAX = 100; break;
	case 5:INIT_MIN = -100; INIT_MAX = 100; break;
	case 6:INIT_MIN = -100; INIT_MAX = 100; break;
	case 7:INIT_MIN = -100; INIT_MAX = 100; break;
	case 8:INIT_MIN = -100; INIT_MAX = 100; break;
	case 9:INIT_MIN = -100; INIT_MAX = 100; break;
	case 10:INIT_MIN = -100; INIT_MAX = 100; break;
	}

	srand((unsigned)time(NULL));
	for (int i = 0; i < initial_popsize; i++)
	{
		for (int j = 0; j < positionDim; j++)
		{
			x[i][j] = (INIT_MAX - INIT_MIN)*rand() / (double)RAND_MAX + INIT_MIN;  //��ʼ��x
			one_x[j] = x[i][j];
		}
		fitnessf(one_x, positionDim, FUNNUM, &one_f);    //��x��ÿһ������(��ά)���Ž�ȥ������Ӧֵ
		f[i] = one_f;
	}


	for (int a = 0; a < all_recurrent; a++)     //����ѭ��20��
	{
		ArrLength = initial_popsize + a * num_recurrent;   //���ŵ������Ӷ����ӵ���������POP_SIZE
		one_d_x = (double(*))malloc(ArrLength * sizeof(double));   //�������õ��ģ���̬��С
		temp_test = (double(*))malloc(ArrLength * sizeof(double));
		one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));
		for (int d = 0; d < positionDim; d++)  // ���������ӵĲ�ͬά�Ƚ��д�С�������򣬲���¼ԭ���
		{
			int k;
			k = 0;
			for (int i = 0; i < ArrLength; i++)
			{
				one_d_x[i] = x[i][d];  //һ��ά���ϵ�x��¼����
				temp_test[i] = x[i][d];     //Ϊ������֮���ҵ���֮ǰ��Ӧ��������
			}
			//quicksort(one_d_x, ArrLength, 0, ArrLength - 1);    //��������һ��ά���ϵ�x����
			heapSort(one_d_x, ArrLength);
			for (int i = 0; i < ArrLength; i++)
			{
				//sort_X[i][d] = one_d_x[i];  //��¼����֮���xλ�ã���ʵû��
				for (int z = 0; z < ArrLength; z++)
				{
					if (one_d_x[i] == temp_test[z])
					{
						one_d_index_x[k++] = z;   //ȡԭ�����±����Բ���1�����˵���ǵڼ�λ�������±�ͼ�1
						break;
					}
				}
			}

			for (int i = 0; i < ArrLength; i++)
			{

				index_x[i][d] = one_d_index_x[i];   //����֮���ԭ���
			}
		}//d
		free(one_d_x);
		free(temp_test);
		free(one_d_index_x);

		one_d_x = NULL;
		temp_test = NULL;
		one_d_index_x = NULL;
		//����x����Ӧֵ
		ArrLength_f = initial_popsize + a * num_recurrent;   //���ŵ������Ӷ����ӵ�������
		one_d_f = (double(*))malloc(ArrLength_f * sizeof(double));
		temp_f = (double(*))malloc(ArrLength_f * sizeof(double));
		one_d_index_f = (int(*))malloc(ArrLength_f * sizeof(int));
/*
		for (int l = 0; l < ArrLength_f; l++)  //x��һ��һ�зŽ�ȥ������Ӧֵ
		{
			for (int d = 0; d < positionDim; d++)
			{
				one_x[d] = x[l][d];
			}

			fitnessf(one_x, positionDim, FUNNUM, &one_f);    //��x��ÿһ������(��ά)���Ž�ȥ������Ӧֵ
			f[l] = one_f;
		}
*/
		//��Ӧֵ���� ��¼�������
		for (int i = 0; i < ArrLength_f; i++)
		{
			one_d_f[i] = f[i];
			temp_f[i] = f[i];
		}

		//quicksort(one_d_f, ArrLength_f, 0, ArrLength_f - 1);    //��������
		heapSort(one_d_f, ArrLength_f);
		int k;
		k = 0;
		index_f = (int(*))malloc(ArrLength_f * sizeof(int));

		for (int i = 0; i < ArrLength_f; i++)
		{
			for (int z = 0; z < ArrLength_f; z++) {
				if (one_d_f[i] == temp_f[z]) {
					one_d_index_f[k++] = z;   //ȡԭ����±�����
					break;
				}
			}
			index_f[i] = one_d_index_f[i];   //����֮���ԭ����±�
			re_index_f[i] = index_f[i];
		}
		free(one_d_f);
		free(temp_f);
		free(one_d_index_f);

		one_d_f = NULL;
		temp_f = NULL;
		one_d_index_f = NULL;
		memcpy(temp_x, x, sizeof(x));  //�ݴ�һ��

		 //condition = ArrLength;

		for (int s = 0; s < num_recurrent; s++)
		{
/*ע��
			condition = ArrLength;
			every_section = (int)floor(condition / section);  //���飺100�� 10��...
			one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));
			one_d_select_index_x = (int(*))malloc(every_section * sizeof(int)); //Ҫ���ص�һ��ά���ϵ�x���������
			select_index_f = (int(*))malloc(every_section * sizeof(int));    //Ҫ���ص�f���������
*/
			for (int d = 0; d < positionDim; d++)
			{
/*ע��
				condition = ArrLength;
				every_section = (int)floor(condition / section);  //���飺100�� 10��...
*/
				//ADD one line
				one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));	
				for (int i = 0; i < ArrLength; i++)
				{
					one_d_index_x[i] = index_x[i][d];  //ȡ��һά�����в��� �Ž���һ������
				}

				// !!!!!!!!!!!!!!!!!!!stupid!!!  ��������д����Զ��1 ��
				//			condition = sizeof(one_d_index_x) / sizeof(int);//��������������������30 input_num
				condition = ArrLength;

				while (condition > input_num)
				{
					//  ���ԣ�����������
					//	row = (int)sizeof(one_d_index_x) / sizeof(int); // ֻ����һά����������ֻ��һά����

					//ADD 3 LINES
						every_section = (int)floor(condition / section);  //���飺100�� 10��...
						one_d_select_index_x = (int(*))malloc(every_section * sizeof(int)); //Ҫ���ص�һ��ά���ϵ�x���������
						select_index_f = (int(*))malloc(every_section * sizeof(int));    //Ҫ���ص�f���������
					select_f_index_draw_rate(temp_x, f, one_d_index_x, d, index_f, W_inhidden, W_hiddenout, condition, every_section, one_d_select_index_x, select_index_f);
					//ADD 2 LINES
						free(one_d_index_x); //��ΪҪѭ�����ã���С��ı����Ե��ͷ�
						free(index_f);
					//	one_d_index_x = NULL;
					//	index_f = NULL;

					//ADD 2 LINES
						one_d_index_x = (int(*))malloc(every_section * sizeof(int));
						index_f = (int(*))malloc(every_section * sizeof(int));
					for (int i = 0; i < every_section; i++)
					{
						one_d_index_x[i] = one_d_select_index_x[i]; //��������ʱ����ѡ��֮������������ٴ�ѡ��
						index_f[i] = select_index_f[i];
					}
					//ADD 2 LINES
						free(one_d_select_index_x);//Ҫ��100������Ϊ10��
						free(select_index_f);
					//	one_d_select_index_x = NULL;
					//    select_index_f = NULL;
					//			printf("%d",select_index_f[0]);

					//  !!!!!!!!!!!!!!!!!!!stupid!!!  ��������д����Զ��1 ��
					//		condition = sizeof(one_d_index_x) / sizeof(int);//��������������������30 input_num
					condition = every_section;

					/*ע��
					every_section = (int)floor(condition / section);
					*/

				} //while һ����2�Σ�1000��100�ٵ�10
				  //if (every_section <= 3 &&condition<=30)  //1000-3000��x��
				  //{
				int r1; //С��
				int r2; //���
				int temp;
				r1 = rand() % condition;
				r2 = rand() % condition;
				if (r1 == r2)
				{
					r2 = rand() % condition;
				}
				if (r1 > r2)
				{
					temp = r1;
					r1 = r2;
					r2 = temp;
				}
				//	sel_new_x = temp_x[one_d_index_x[r1]][d] + (temp_x[one_d_index_x[r2]][d] - temp_x[one_d_index_x[r1]][d]) * zero_to_one;

				sel_new_x = (temp_x[one_d_index_x[r2]][d] - (temp_x[one_d_index_x[r1]][d]))* rand() / (double)RAND_MAX + temp_x[one_d_index_x[r1]][d];

				//}
				
				add_position[s][d] = sel_new_x;
				x[ArrLength + s][d] = sel_new_x;  //���Ӳ����µ�һ���㣬
				//ADD 3 LINES
				free(one_d_index_x);
				free(index_f);
				index_f = (int(*))malloc(ArrLength_f * sizeof(int));
				//	one_d_index_x = NULL;
				//	index_f = NULL;
				
				for (int j = 0; j < ArrLength_f; j++)
				{
					index_f[j] = re_index_f[j];
				}
			}//d
			/*ע��
			free(one_d_index_x);
			free(one_d_select_index_x);
			free(select_index_f);
			one_d_index_x = NULL;
			one_d_select_index_x = NULL;
			select_index_f = NULL;
			*/
		}//s

		 		free(index_f);
		

		for (int s = 0; s < num_recurrent; s++)
		{
			for (int d = 0; d < positionDim; d++)
			{
				one_x_add_position[d] = add_position[s][d];
			}
			fitnessf(one_x_add_position, positionDim, FUNNUM, &add_one_f);    //��������ӵ�x����Ӧֵ																	 
			f[ArrLength_f + s] = add_one_f;
		}

	}//all��*num_recurrent = 2000���µ�

	nn_best_fit = f[0];     //����С��Ӧֵ
	for (int m = 0; m < ArrLength; m++)
	{
		if (f[m] < nn_best_fit)
		{
			nn_best_fit = f[m];
		}
	}
	return nn_best_fit;
}




void selection_tournment(double neural_network[nn_num][total_weight], double GA_nn_fitness[nn_num])
{
	int m;
	m = nn_num;
	int select[4] = { 0 };
	int mark2[nn_num] = { 0 };//��Ǹ�����û�б�ѡ��
	int index[nn_num] = { 0 };//
	double min;
	int maxindex;

	for (int i = 0; i < m; i++)// m = nn_num
	{
		for (int j = 0; j < 4; j++)
		{
			int r2 = rand() % m + 1;   //1-m֮���ĸ����壨������
			while (mark2[r2 - 1] == 1)
			{
				r2 = rand() % m + 1;
			}
			mark2[r2 - 1] = 1;
			select[j] = r2 - 1;
		}
		min = GA_nn_fitness[select[0]];
		maxindex = select[0];
		for (int k = 1; k < 4; k++)
		{
			if (GA_nn_fitness[select[k]] < min)
			{
				min = GA_nn_fitness[select[k]];
				maxindex = k;
			}
		}
		index[i] = maxindex;
		for (int n = 0; n < nn_num; n++)
		{
			mark2[n] = 0;
		}

		for (int n = 0; n < 4; n++)
		{
			select[n] = 0;
		}


		for (int l = 0; l < total_weight; l++)
		{
			neural_network[i][l] = neural_network[index[i]][l];
		}
	}

} //selection_tournment����

void crossover(double neural_network[nn_num][total_weight], double pc)
{
	//const double a = 0.0;
	//const double b = 1.0;
	int two;
	int one;
	int first = 0;
	double r;
	int point;
	double t;

	for (two = 0; two < nn_num; two++)
	{
		r = rand() / (RAND_MAX + 1.0);
		if (r < pc)
		{
			++first;
			if (first % 2 == 0)//����
			{
				point = rand() % total_weight + 1;  //���ѡ�񽻲��
				for (int i = 0; i < point; i++)
				{
					t = neural_network[one][i];
					neural_network[one][i] = neural_network[two][i];
					neural_network[two][i] = t;
				}
			}
			else
			{
				one = two;
			}

		}
	}
}

void mutation(double neural_network[nn_num][total_weight], double pm)
{
	double rand_r;
	//  srand((unsigned)time(NULL));
	for (int i = 0; i < nn_num; i++)
	{
		for (int d = 0; d < total_weight; d++)
		{
			rand_r = rand() / (RAND_MAX + 1.0);
			if (rand_r < pm)
			{
				neural_network[i][d] = (double)rand() / RAND_MAX * 2 - 1.0;
			}
		}

	}
}//����

void main()
{
	int FUNNUM;
	double nn_fitness_one_f[nn_num];
	double temp_compare[nn_num];
	double min_fit[MAX_GENERATION];
	double GA_nn_fitness[nn_num];
	//	double selpop[nn_num][total_weight];
	double pc = 0.5;    //�������
	double pm = 0.1;    //�������
	double the_min_fit_inallG = nn_num;
	int record_sort_index[nn_num];   //1�������µ�nn_num��NN ����
	double record_nn_fitness_sort_index[nn_num][benchmark_number + 1];  //10�������µ�nn_num��NN ����
	int record_g = 0;
	double GA_best_NN[total_weight];
	double gbest_NN[total_weight];


	FILE* fp = fopen("GA100results.txt", "a");
	FILE* fw = fopen("GA100bestNN.txt", "a");

	if (fp == NULL || fw == NULL)
	{
		printf("failed to open file\n");
		system("pause");
	}

	initial_NN();   //����nn_num����������
	char fileName[256];
/*	
	FILE *fnn[nn_num];

	if (fnn == NULL)
	{
		printf("error!");
	}
	
	for (int nn = 0; nn < nn_num; nn++)
	{
		sprintf(fileName, "NN/g66/NN_%d.txt", nn);
		fnn[nn] = fopen(fileName, "r");
		for (int i = 0; i < total_weight; i++)
		{
			fscanf(fnn[nn], "%lf", &neural_network[nn][i]);    //����õ��������Ȩֵȡ����
		}
		fclose(fnn[nn]);
	}
	*/

	//  printf("\n");
	
	for (int g = 0; g < MAX_GENERATION; g++)
	{
		FILE* f_temp_NN = fopen("temp_bestNN.txt", "a");
		FILE* f_temp_result = fopen("temp_result.txt", "a");

		FILE* a_generation_NN[nn_num + 1];
		

		for (FUNNUM = 0; FUNNUM < benchmark_number; FUNNUM++)
		{

			for (int nn = 0; nn < nn_num; nn++)      //��Ҫ���У�nn_num��������һ����
			{
				double re_convert1[input_num * hidden_num] = { 0 };
				double re_convert2[hidden_num*output_num] = { 0 };
				double W_inhidden[input_num][hidden_num];
				double W_hiddenout[hidden_num][output_num];

				for (int i = 0; i < input_num * hidden_num; i++)        //��������磺Ϊ�˸��õļ���
				{
					re_convert1[i] = neural_network[nn][i];
				}
				int k = 0;
				for (int i = input_num * hidden_num; i < 1600; i++)
				{
					re_convert2[k] = neural_network[nn][i];
					k++;
				}
				for (int i = 0; i < input_num; i++) {
					for (int j = 0; j < hidden_num; j++) {
						W_inhidden[i][j] = re_convert1[i * hidden_num + j];
					}
				}
				for (int i = 0; i < hidden_num; i++)
				{
					for (int j = 0; j < output_num; j++)
					{
						W_hiddenout[i][j] = re_convert2[i * output_num + j];
					}
				}

				nn_fitness[nn][FUNNUM] = dif_nn(W_inhidden, W_hiddenout, FUNNUM + 1);  //1�������������к������в�ͬ��fitness������λ��x������ģ�

			} //Ӧ���ǲ��в��֣�nn_num��������һ����

		}//FUNNUM


		for (int fun_num = 0; fun_num < benchmark_number; fun_num++)
		{
			for (int nn = 0; nn < nn_num; nn++)
			{
				nn_fitness_one_f[nn] = nn_fitness[nn][fun_num];     //ͬһ�������µ��������������С��Ӧֵ�ó���
				temp_compare[nn] = nn_fitness[nn][fun_num];
			}
			int k = 0;
			//quicksort(nn_fitness_one_f, nn_num, 0, nn_num - 1);  //������������NN��һ�������ϵ�fitness������
			heapSort(nn_fitness_one_f, nn_num);
			for (int w = 0; w < nn_num; w++)
			{
				for (int z = 0; z < nn_num; z++)
				{
					if (nn_fitness_one_f[w] == temp_compare[z])   //����֮��ĺ�ԭ������ȣ�����֪���±�
					{
						record_sort_index[k++] = z + 1;   //��Ӧ��������
						break;
					}
				}
			}
			for (int i = 0; i < nn_num; i++)
			{
				record_nn_fitness_sort_index[i][fun_num] = record_sort_index[i]; //nn_num * fun_num������
			}
		}//10������

		for (int l = 0; l < nn_num; l++)   //ƽ������
		{
			double sum = 0;
			for (int i = 0; i < benchmark_number; i++)
			{
				sum += record_nn_fitness_sort_index[l][i];
			}

			record_nn_fitness_sort_index[l][benchmark_number] = sum / benchmark_number;
			GA_nn_fitness[l] = sum / benchmark_number;
			printf("the %dth NN in all f's avg ranking is :%lf \n", l + 1, GA_nn_fitness[l]);
		}


		min_fit[g] = GA_nn_fitness[0];     //����С��������Ӧ������
		for (int nn = 1; nn < nn_num; nn++)
		{
			if (GA_nn_fitness[nn] < min_fit[g])
			{
				min_fit[g] = GA_nn_fitness[nn];
				for (int i = 0; i < total_weight; i++)
				{
					GA_best_NN[i] = neural_network[nn][i];
				}
			}
			else
			{
				for (int i = 0; i < total_weight; i++)
				{
					GA_best_NN[i] = neural_network[0][i];
				}
			}

		}


		sprintf(fileName, "NN/g%d/GA_best_NN.txt", g);
		a_generation_NN[nn_num] = fopen(fileName, "a");
		if (a_generation_NN[nn_num] == NULL)
		{
			printf("\n Error: Cannot open input file for writing \n");
		}
		for (int i = 0; i < total_weight; i++)
		{
			
			fprintf(a_generation_NN[nn_num], "%lf\n", GA_best_NN[i]);
		}

		printf("the generation %d min ranking is:%lf \n", g + 1, min_fit[g]);


		if (min_fit[g] < the_min_fit_inallG)
		{
			the_min_fit_inallG = min_fit[g];
			for (int i = 0; i < total_weight; i++)
			{
				gbest_NN[i] = GA_best_NN[i];
				fprintf(f_temp_NN, "%lf\n", gbest_NN[i]);
			}
			record_g = g;
		}
		printf("the history min ranking is:%lf \n", the_min_fit_inallG);

		fprintf(fp, "g=%d���ţ�%lf\n", g + 1, min_fit[g]);
		fprintf(fp, "��ʷ������%d��ȡ����%lf\n", record_g, the_min_fit_inallG);

		selection_tournment(neural_network, GA_nn_fitness); //�ı�selpop
		crossover(neural_network, pc);
		mutation(neural_network, pm);
		
		fprintf(f_temp_result, "g=%d���ţ�%lf\n", g + 1, min_fit[g]);
		fprintf(f_temp_result, "��ʷ������%d��ȡ����%lf\n", record_g, the_min_fit_inallG);
		fclose(f_temp_NN);
		fclose(f_temp_result);

		for (int nn = 0; nn < nn_num; nn++) //����ÿһ���ĸ���������
		{
			sprintf(fileName, "NN/g%d/NN_%d.txt", g, nn);
			a_generation_NN[nn] = fopen(fileName, "a");
			if (a_generation_NN[nn] == NULL)
			{
				printf("\n Error: Cannot open input file for writing \n");
			}
			for (int i = 0; i < total_weight; i++)
			{
				fprintf(a_generation_NN[nn], "%lf\n", neural_network[nn][i]);
			}
		}
		for (int i = 0; i < nn_num + 1; i++)
		{
			fclose(a_generation_NN[i]);
		}
	}//GA��
	for (int i = 0; i < total_weight; i++)
	{
		fprintf(fw, "%lf\n", gbest_NN[i]);
	}
	printf("%d\n", fitcount);
	free(y);
	free(z);
	free(M);
	free(OShift);
	free(x_bound);
	fclose(fp);
	fclose(fw);

	getchar();
}//main����

