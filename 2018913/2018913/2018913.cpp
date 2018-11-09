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

#define MAX_GENERATION 200 //多少代GA
#define initial_popsize 1000
#define num_recurrent 100   //选100次：100个新点
#define all_recurrent 30    //整体循环90次：90*100=9000个新点  设30就不行了
#define final_popsize initial_popsize+num_recurrent*all_recurrent

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
int fitcount;

double neural_network[nn_num][total_weight];
double nn_fitness[nn_num][benchmark_number];

void initial_NN()    //初始化nn_num个神经网络的权值
{
	int nn, i;

	//   srand((unsigned)time(NULL));
	for (nn = 0; nn < nn_num; nn++)
	{
		for (i = 0; i < total_weight; i++)
		{
			neural_network[nn][i] = (double)rand() / RAND_MAX * 2 - 1.0;   //神经网络初始化权值【-1,1】																		   //	printf("%f\n",neural_network[nn][i]);
		}
	}
}
void swap(double *a, double *b)  //交换2个数
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
	double tmp = arr[idx1];  //暂时存放要调整的数据
	for (int idx = idx1 * 2 + 1; idx <= idx2; idx = idx * 2 + 1)  //从要调整的数据的左孩子开始比较
	{
		//选出左右孩子中的最大结
		if (idx + 1 <= idx2 && arr[idx] < arr[idx + 1])
			++idx;
		if (arr[idx] > tmp)  //不满足大根堆，调整
		{
			arr[idx1] = arr[idx];  //交换，可能破坏子树满足大根堆的性质
			idx1 = idx;  //本来这里要交换的，但时tmp暂时存放了初始arr[idx1]的值，这里每次比较都是和tmp比较,好比交换了，所以可以不用先交换
						 //继续向下调整，直到树满足大根堆性质
		}
		else
			break;
	}
	arr[idx1] = tmp;
}
//堆排序，数组下标对应其在堆数据结构中的结点位置，从0开始编号，堆是完全二叉树
double heapSort(double* arr, int length)
{
	if (nullptr == arr || length <= 0)
		return -1;
	//数组中顺序存放的数据就对应完全二叉树堆中的对应结点的值，现在调整为大根堆
	for (int idx = length / 2 - 1; idx >= 0; --idx)  //从最后一个非叶子结点开始调整为最大堆
	{
		adjust(arr, idx, length - 1);  //最后一个非叶子结点和它的孩子比较调整
	}
	//排序，根结点后最后一个结点交换，调整
	for (int idx = length - 1; idx > 0; --idx)
	{
		swap(&arr[0], &arr[idx]);  //每次选出一个最大的数放到末尾，也就是数组末尾
		adjust(arr, 0, idx - 1);  //调整根结点到idx-1个结点为大根堆
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
	//将二维数组中的变量存放到一维数组中
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

	r = rand() / (RAND_MAX + 1.0);      //产生0-1之间的随机小数
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
	//	int *index_x = (int(*))malloc(sizeof(one_d_index_x)); //仅是“4” 因为int是4，那就说明 是1个，其实是指针
	int *index_x = (int(*))malloc(condition * sizeof(int));

	int not_select_index_x;
	int index_1, index_2, index_end;   //取索引值序号的第一个
	int ArrLength;

	int(*a_index_section)[section];  //分组  100*10，然后10*10：every_section*section
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
	double boundary[section + 1];  //11个边界值
	double every_rate[section] = { 0 }; //10个通过NN计算得出的概率
	double sum_rate[section];
	int a_section_index[2];   //选中的那个区间的边界值的2个序号
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
		randsel = rand() % condition + 1;     //随机整数
		index_1 = index_x[0];       //取出第一个序号
		boundary[0] = temp_x[index_1][d];
		for (int i = 0; i < section - 1; i++)  //中间临界平均分
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
				a_index_section[j][i] = index_x[i * every_section + j]; //位置序号分成section个区间，1个区间有100个/10个/1个

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
				a_section_fit[j][i] = temp_f[i * every_section + j]; //fit序号分成section个区间，1个区间有100个/10个/1个
				a_section_fit_index[j][i] = index_f[i * every_section + j];
			}
		}

		for (int i = 0; i < section; i++)   //每组取fitness的最大值、最小值和中值
		{
			k = 0;
			for (int j = 0; j < every_section; j++)
			{
				temp_section_fit[j] = a_section_fit[j][i];   //a_section_fit是二维，要对每个区间进行排序
				temp[j] = a_section_fit[j][i];
				temp_section_fit_index[j] = a_section_fit_index[j][i]; //一个区间的f索引
			}

			ArrLength = every_section;  //一个区间排序
										//quicksort(temp_section_fit, ArrLength, 0, ArrLength - 1);  //快速排序一个区间上的fitness：排名
			heapSort(temp_section_fit, ArrLength);
			for (int w = 0; w < ArrLength; w++)
			{
				for (z = 0; z < ArrLength; z++)
				{
					if (temp_section_fit[w] == temp[z])   //排序之后的和原来的相比，可以知道下标
					{
						one_section_index_f[k] = z;   //对应原来的下标
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
		}//i 10个区间结束

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

		//通过NN计算每个区间的概率
		nn_compute(section_fit_index_sort, W_inhidden, W_hiddenout, every_rate);

		//累加概率
		sum_rate[0] = every_rate[0];
		for (int l = 1; l < section; l++)
		{
			sum_rate[l] = sum_rate[l - 1] + every_rate[l];
		}
		//轮盘赌选择落到哪一个区间上
		select_action(sum_rate, a_section_index);

		for (int m = 0; m < every_section; m++)
		{
			one_d_select_index_x[m] = a_index_section[m][a_section_index[0]]; //取的那一个区间里的所有，方便以后再分
			select_index_f[m] = a_section_fit_index[m][a_section_index[0]];
		}
		free(a_index_section);
		free(a_section_fit_index);
		a_index_section = NULL;
		a_section_fit_index = NULL;
	}
	/*	else
	{
	randsel = rand() % condition;     //随机整数
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

	//	int all_recurrent = 20;    //整体循环20次：20*100=2000个新点
	int ArrLength;
	ArrLength = initial_popsize;
	int ArrLength_f;

	ArrLength_f = initial_popsize;
	//	int row;
	int every_section = 100;

	int condition;
	double x[final_popsize][positionDim];
	double temp_x[final_popsize][positionDim];

	double *one_d_x;               //一个维度上的x记录下来
	int index_x[final_popsize][positionDim];  //二维上分别排完序之后的索引（下标）
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

	int *one_d_select_index_x; //要返回的x的索引序号
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
			x[i][j] = (INIT_MAX - INIT_MIN)*rand() / (double)RAND_MAX + INIT_MIN;  //初始化x
			one_x[j] = x[i][j];
		}
		fitnessf(one_x, positionDim, FUNNUM, &one_f);    //把x的每一个粒子(多维)都放进去计算适应值
		f[i] = one_f;
	}


	for (int a = 0; a < all_recurrent; a++)     //整体循环20次
	{
		ArrLength = initial_popsize + a * num_recurrent;   //随着迭代增加而增加的粒子数，POP_SIZE
		one_d_x = (double(*))malloc(ArrLength * sizeof(double));   //反复利用到的，动态大小
		temp_test = (double(*))malloc(ArrLength * sizeof(double));
		one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));
		for (int d = 0; d < positionDim; d++)  // 对所有粒子的不同维度进行从小到大排序，并记录原序号
		{
			int k;
			k = 0;
			for (int i = 0; i < ArrLength; i++)
			{
				one_d_x[i] = x[i][d];  //一个维度上的x记录下来
				temp_test[i] = x[i][d];     //为了排序之后找到和之前对应的索引，
			}
			//quicksort(one_d_x, ArrLength, 0, ArrLength - 1);    //快速排序：一个维度上的x排序
			heapSort(one_d_x, ArrLength);
			for (int i = 0; i < ArrLength; i++)
			{
				//sort_X[i][d] = one_d_x[i];  //记录排序之后的x位置：其实没用
				for (int z = 0; z < ArrLength; z++)
				{
					if (one_d_x[i] == temp_test[z])
					{
						one_d_index_x[k++] = z;   //取原索引下标所以不加1，如果说的是第几位而不是下标就加1
						break;
					}
				}
			}

			for (int i = 0; i < ArrLength; i++)
			{

				index_x[i][d] = one_d_index_x[i];   //排序之后的原序号
			}
		}//d
		free(one_d_x);
		free(temp_test);
		free(one_d_index_x);

		one_d_x = NULL;
		temp_test = NULL;
		one_d_index_x = NULL;
		//计算x的适应值
		ArrLength_f = initial_popsize + a * num_recurrent;   //随着迭代增加而增加的粒子数
		one_d_f = (double(*))malloc(ArrLength_f * sizeof(double));
		temp_f = (double(*))malloc(ArrLength_f * sizeof(double));
		one_d_index_f = (int(*))malloc(ArrLength_f * sizeof(int));
/*
		for (int l = 0; l < ArrLength_f; l++)  //x的一行一行放进去计算适应值
		{
			for (int d = 0; d < positionDim; d++)
			{
				one_x[d] = x[l][d];
			}

			fitnessf(one_x, positionDim, FUNNUM, &one_f);    //把x的每一个粒子(多维)都放进去计算适应值
			f[l] = one_f;
		}
*/
		//适应值排序 记录排名序号
		for (int i = 0; i < ArrLength_f; i++)
		{
			one_d_f[i] = f[i];
			temp_f[i] = f[i];
		}

		//quicksort(one_d_f, ArrLength_f, 0, ArrLength_f - 1);    //快速排序
		heapSort(one_d_f, ArrLength_f);
		int k;
		k = 0;
		index_f = (int(*))malloc(ArrLength_f * sizeof(int));

		for (int i = 0; i < ArrLength_f; i++)
		{
			for (int z = 0; z < ArrLength_f; z++) {
				if (one_d_f[i] == temp_f[z]) {
					one_d_index_f[k++] = z;   //取原序号下标索引
					break;
				}
			}
			index_f[i] = one_d_index_f[i];   //排序之后的原序号下标
			re_index_f[i] = index_f[i];
		}
		free(one_d_f);
		free(temp_f);
		free(one_d_index_f);

		one_d_f = NULL;
		temp_f = NULL;
		one_d_index_f = NULL;
		memcpy(temp_x, x, sizeof(x));  //暂存一下

		 //condition = ArrLength;

		for (int s = 0; s < num_recurrent; s++)
		{
/*注释
			condition = ArrLength;
			every_section = (int)floor(condition / section);  //分组：100个 10个...
			one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));
			one_d_select_index_x = (int(*))malloc(every_section * sizeof(int)); //要返回的一个维度上的x的索引序号
			select_index_f = (int(*))malloc(every_section * sizeof(int));    //要返回的f的索引序号
*/
			for (int d = 0; d < positionDim; d++)
			{
/*注释
				condition = ArrLength;
				every_section = (int)floor(condition / section);  //分组：100个 10个...
*/
				//ADD one line
				one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));	
				for (int i = 0; i < ArrLength; i++)
				{
					one_d_index_x[i] = index_x[i][d];  //取出一维来进行操作 放进下一个函数
				}

				// !!!!!!!!!!!!!!!!!!!stupid!!!  下面这种写法永远是1 ！
				//			condition = sizeof(one_d_index_x) / sizeof(int);//行数：粒子数不能少于30 input_num
				condition = ArrLength;

				while (condition > input_num)
				{
					//  不对！！！！！！
					//	row = (int)sizeof(one_d_index_x) / sizeof(int); // 只传了一维进来，所以只是一维数组

					//ADD 3 LINES
						every_section = (int)floor(condition / section);  //分组：100个 10个...
						one_d_select_index_x = (int(*))malloc(every_section * sizeof(int)); //要返回的一个维度上的x的索引序号
						select_index_f = (int(*))malloc(every_section * sizeof(int));    //要返回的f的索引序号
					select_f_index_draw_rate(temp_x, f, one_d_index_x, d, index_f, W_inhidden, W_hiddenout, condition, every_section, one_d_select_index_x, select_index_f);
					//ADD 2 LINES
						free(one_d_index_x); //因为要循环利用，大小会改变所以得释放
						free(index_f);
					//	one_d_index_x = NULL;
					//	index_f = NULL;

					//ADD 2 LINES
						one_d_index_x = (int(*))malloc(every_section * sizeof(int));
						index_f = (int(*))malloc(every_section * sizeof(int));
					for (int i = 0; i < every_section; i++)
					{
						one_d_index_x[i] = one_d_select_index_x[i]; //继续进行时利用选出之后的索引进行再次选择
						index_f[i] = select_index_f[i];
					}
					//ADD 2 LINES
						free(one_d_select_index_x);//要由100即将变为10了
						free(select_index_f);
					//	one_d_select_index_x = NULL;
					//    select_index_f = NULL;
					//			printf("%d",select_index_f[0]);

					//  !!!!!!!!!!!!!!!!!!!stupid!!!  下面这种写法永远是1 ！
					//		condition = sizeof(one_d_index_x) / sizeof(int);//行数：粒子数不能少于30 input_num
					condition = every_section;

					/*注释
					every_section = (int)floor(condition / section);
					*/

				} //while 一般是2次，1000到100再到10
				  //if (every_section <= 3 &&condition<=30)  //1000-3000个x分
				  //{
				int r1; //小的
				int r2; //大的
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
				x[ArrLength + s][d] = sel_new_x;  //增加产生新的一个点，
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
			/*注释
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
			fitnessf(one_x_add_position, positionDim, FUNNUM, &add_one_f);    //计算新添加的x的适应值																	 
			f[ArrLength_f + s] = add_one_f;
		}

	}//all次*num_recurrent = 2000个新点

	nn_best_fit = f[0];     //找最小适应值
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
	int mark2[nn_num] = { 0 };//标记个体有没有被选中
	int index[nn_num] = { 0 };//
	double min;
	int maxindex;

	for (int i = 0; i < m; i++)// m = nn_num
	{
		for (int j = 0; j < 4; j++)
		{
			int r2 = rand() % m + 1;   //1-m之间哪个个体（整数）
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

} //selection_tournment函数

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
			if (first % 2 == 0)//交叉
			{
				point = rand() % total_weight + 1;  //随机选择交叉点
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
}//变异

void main()
{
	int FUNNUM;
	double nn_fitness_one_f[nn_num];
	double temp_compare[nn_num];
	double min_fit[MAX_GENERATION];
	double GA_nn_fitness[nn_num];
	//	double selpop[nn_num][total_weight];
	double pc = 0.5;    //交叉概率
	double pm = 0.1;    //变异概率
	double the_min_fit_inallG = nn_num;
	int record_sort_index[nn_num];   //1个函数下的nn_num个NN 名次
	double record_nn_fitness_sort_index[nn_num][benchmark_number + 1];  //10个函数下的nn_num个NN 名次
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

	initial_NN();   //有了nn_num个神经网络了
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
			fscanf(fnn[nn], "%lf", &neural_network[nn][i]);    //把最好的神经网络的权值取出来
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

			for (int nn = 0; nn < nn_num; nn++)      //需要并行：nn_num个神经网络一起跑
			{
				double re_convert1[input_num * hidden_num] = { 0 };
				double re_convert2[hidden_num*output_num] = { 0 };
				double W_inhidden[input_num][hidden_num];
				double W_hiddenout[hidden_num][output_num];

				for (int i = 0; i < input_num * hidden_num; i++)        //拆分神经网络：为了更好的计算
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

				nn_fitness[nn][FUNNUM] = dif_nn(W_inhidden, W_hiddenout, FUNNUM + 1);  //1个神经网络在所有函数下有不同的fitness（根据位置x来计算的）

			} //应该是并行部分：nn_num个神经网络一起跑

		}//FUNNUM


		for (int fun_num = 0; fun_num < benchmark_number; fun_num++)
		{
			for (int nn = 0; nn < nn_num; nn++)
			{
				nn_fitness_one_f[nn] = nn_fitness[nn][fun_num];     //同一个函数下的所有神经网络的最小适应值拿出来
				temp_compare[nn] = nn_fitness[nn][fun_num];
			}
			int k = 0;
			//quicksort(nn_fitness_one_f, nn_num, 0, nn_num - 1);  //快速排序所有NN在一个函数上的fitness：排名
			heapSort(nn_fitness_one_f, nn_num);
			for (int w = 0; w < nn_num; w++)
			{
				for (int z = 0; z < nn_num; z++)
				{
					if (nn_fitness_one_f[w] == temp_compare[z])   //排序之后的和原来的相比，可以知道下标
					{
						record_sort_index[k++] = z + 1;   //对应排名名次
						break;
					}
				}
			}
			for (int i = 0; i < nn_num; i++)
			{
				record_nn_fitness_sort_index[i][fun_num] = record_sort_index[i]; //nn_num * fun_num个名次
			}
		}//10个函数

		for (int l = 0; l < nn_num; l++)   //平均排名
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


		min_fit[g] = GA_nn_fitness[0];     //找最小排名及对应的网络
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

		fprintf(fp, "g=%d最优：%lf\n", g + 1, min_fit[g]);
		fprintf(fp, "历史最优在%d代取到：%lf\n", record_g, the_min_fit_inallG);

		selection_tournment(neural_network, GA_nn_fitness); //改变selpop
		crossover(neural_network, pc);
		mutation(neural_network, pm);
		
		fprintf(f_temp_result, "g=%d最优：%lf\n", g + 1, min_fit[g]);
		fprintf(f_temp_result, "历史最优在%d代取到：%lf\n", record_g, the_min_fit_inallG);
		fclose(f_temp_NN);
		fclose(f_temp_result);

		for (int nn = 0; nn < nn_num; nn++) //保存每一代的各个神经网络
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
	}//GA代
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
}//main结束

