#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#define nn_num 100
#define benchmark_number 10
#define input_num 30
#define hidden_num 40
#define output_num 10
#define positionDim 2
#define section 10
#define total_weight 1600

#define MAX_GENERATION  300 
#define initial_popsize 1000
#define num_recurrent 100   
#define all_recurrent 30    
#define final_popsize initial_popsize+num_recurrent*all_recurrent

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
int fitcount;

double neural_network[nn_num][total_weight];
double nn_fitness[nn_num][benchmark_number];

void initial_NN()   
{
	int nn, i;
	printf("initial_NN \n");
	for (nn = 0; nn < nn_num; nn++)
	{
		for (i = 0; i < total_weight; i++)
		{
			neural_network[nn][i] = (double)rand() / RAND_MAX * 2 - 1.0;   
		}	
	}

	//printf("end end end initial_NN() is end end end end \n");
}
void swap(double *a, double *b) 
{
	double temp;
	temp = *a;
	*a = *b;
	*b = temp;
}
void adjust(double* arr, int idx1, int idx2)
{
	int idx;
	if (NULL == arr || idx1 >= idx2 || idx1 < 0 || idx2 < 0)
		return;
	double tmp = arr[idx1]; 
	for (idx = idx1 * 2 + 1; idx <= idx2; idx = idx * 2 + 1)  
	{
		if (idx + 1 <= idx2 && arr[idx] < arr[idx + 1])
			++idx;
		if (arr[idx] > tmp) 
		{
			arr[idx1] = arr[idx];  
			idx1 = idx;  
		}
		else
			break;
	}
	arr[idx1] = tmp;
}



double heapSort(double* arr, int length)
{
	int idx;
	if (NULL == arr || length <= 0)
		return -1;
	
	for (idx = length / 2 - 1; idx >= 0; --idx)  
	{
		adjust(arr, idx, length - 1);  
	}
	
	for (idx = length - 1; idx > 0; --idx)
	{
		swap(&arr[0], &arr[idx]); 
		adjust(arr, 0, idx - 1); 
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
	int j;
	

	r = rand() / (RAND_MAX + 1.0);      //产生0-1之间的随机小数
	if (r < sum_rate[0])
	{
		a_section_index[0] = 0;
		a_section_index[1] = 1;
	}
	else
	{
		for (j = 0; j < section - 1; j++)
		{
			if (r >= sum_rate[j] && r < sum_rate[j + 1])
			{
				a_section_index[0] = j + 1;
				a_section_index[1] = j + 2;
			}
		}
	}
}



void select_f_index_draw_rate(double temp_x[][positionDim], double f[], int *one_d_index_x, int d, int *index_f, double W_inhidden[][hidden_num], double W_hiddenout[][output_num], int condition, int every_section, int *one_d_select_index_x, int *select_index_f)
{
	int i,j,l,k,zz,w,m;
	int randsel;
	int *index_x = (int(*))malloc(condition * sizeof(int));
	int not_select_index_x;
	int index_1, index_2, index_end;   //取索引值序号的第一个
	int ArrLength;

	//int(*a_index_section)[section];  
	//a_index_section = (int(*)[section])malloc(every_section*section * sizeof(int*));

//int (*a_index_section)[section]=(int(*)[section])malloc(sizeof(int)*every_section*section);

int **a_index_section;
a_index_section = (int**)malloc(sizeof(int*)*every_section);
for(i=0;i<every_section;i++)
{
	a_index_section[i] = (int*)malloc(sizeof(int)*section);
}

//	double(*a_section_fit)[section];
//	a_section_fit = (double((*)[section]))malloc(every_section*section * sizeof(double*));
double **a_section_fit;
a_section_fit = (double**)malloc(sizeof(double*)*every_section);
for(i=0;i<every_section;i++)
{
	a_section_fit[i] = (double*)malloc(sizeof(double)*section);
}

//	int(*a_section_fit_index)[section];
//	a_section_fit_index = (int(*)[section])malloc(every_section*section * sizeof(int*));
int **a_section_fit_index;
a_section_fit_index = (int**)malloc(sizeof(int*)*every_section);
for(i=0;i<every_section;i++)
{
	a_section_fit_index[i] = (int*)malloc(sizeof(int)*section);
}

	int section_fit_index_sort[section][3];
	int max_index[section];
	int min_index[section];
	int mid_index[section];

	int index_f_record;
	double temp_f[final_popsize]={0};
	double boundary[section + 1];  //11个边界值
	double every_rate[section] = { 0 }; //10个通过NN计算得出的概率
	double sum_rate[section];
	int a_section_index[2];   
	double *temp_section_fit;
	int *temp_section_fit_index;
	double *temp;
	int *one_section_index_f;
	
	int index1, index2, index3;

	temp_section_fit = (double(*))malloc(sizeof(double)*every_section);
	temp = (double(*))malloc(sizeof(double)*every_section);
	temp_section_fit_index = (int(*))malloc(sizeof(int)*every_section);
	one_section_index_f = (int(*))malloc(sizeof(int)*every_section);

	memcpy(index_x, one_d_index_x, sizeof(one_d_index_x)*condition/2);

//printf("memcpy index_x\n");

		for (i = 0; i < section; i++)
		{
			for (j = 0; j < every_section; j++)
			{
				a_index_section[j][i] = index_x[i * every_section + j]; 

//printf("-------index_x=%d",index_x[i * every_section + j]);
//printf("\n");
			//printf("-a_index_section=%d\n",a_index_section[j][i]);
			}

		}

//printf("\na_index_section hang=%d,lie=%d\n",sizeof(a_index_section)/sizeof(a_index_section[0]),sizeof(a_index_section[0])/sizeof(a_index_section[0][0]));

//		free(index_x);

		//index_x = NULL;
		for (j = 0; j < condition; j++) 
		{
			index_f_record = index_f[j];
			temp_f[j] = f[index_f_record];
			//printf("-temp_f[j]=%lf\n",temp_f[j]);
		}
//printf("--test temp_f--");
		for (i = 0; i < section; i++)
		{
			for (j = 0; j < every_section; j++)
			{
				a_section_fit[j][i] = temp_f[i * every_section + j]; 
				a_section_fit_index[j][i] = index_f[i * every_section + j];
			}
		}
//printf("a_section_fit[0][0]=%lf\n",a_section_fit[0][0]);
		for (i = 0; i < section; i++)   //每组取fitness的最大值、最小值和中值
		{
			k = 0;
			for (j = 0; j < every_section; j++)
			{
				temp_section_fit[j] = a_section_fit[j][i];   //a_section_fit是二维，要对每个区间进行排序
				temp[j] = a_section_fit[j][i];
				temp_section_fit_index[j] = a_section_fit_index[j][i]; //一个区间的f索引
			/*	if(isnan(temp_section_fit[j])==1)
				{
				temp_section_fit[j]=10000000;
				}
				if(isnan(temp[j])==1)
				{
				temp[j]=10000000;
				}*/

			}

			ArrLength = every_section;  
			//quicksort(temp_section_fit, ArrLength, 0, ArrLength - 1);  //快速排序一个区间上的fitness：排名
			heapSort(temp_section_fit, ArrLength);
			for (w = 0; w < ArrLength; w++)
			{
				for (zz = 0; zz < ArrLength; zz++)
				{
					if (temp_section_fit[w] == temp[zz])   //排序之后的和原来的相比，可以知道下标
					{
						one_section_index_f[k] = zz;   //对应原来的下标
						k++;
						break;
					}
				}
			}
//printf("---k=%d---",k);
			index1 = one_section_index_f[k - 1];
			index2 = one_section_index_f[0];
			index3 = (int)floor((one_section_index_f[(int)floor(ArrLength / 2)] + one_section_index_f[(int)floor(ArrLength / 2) + 1]) / 2);

			max_index[i] = temp_section_fit_index[index1];
			min_index[i] = temp_section_fit_index[index2];
			mid_index[i] = temp_section_fit_index[index3];

			section_fit_index_sort[i][0] = max_index[i];
			section_fit_index_sort[i][1] = min_index[i];
			section_fit_index_sort[i][2] = mid_index[i];
		}//i section end

//printf("before free temp_section_fit\n");

		free(temp_section_fit);
		free(temp);
		free(temp_section_fit_index);
		free(one_section_index_f);


		//free(a_section_fit);
//printf("before free a_section_fit success!!!!!!!\n");
for(i=0;i<every_section;i++)
{
	free(a_section_fit[i]);
}

free(a_section_fit);
//printf("free a_section_fit success!!!!!!!\n");

/*
		temp_section_fit = NULL;
		temp = NULL;
		temp_section_fit_index = NULL;
		one_section_index_f = NULL;
		a_section_fit = NULL;
*/
		//a_section_fit.shrink_to_fit();

		boundary[0] = temp_x[a_index_section[0][0]][d];
		for (i = 0; i < section-1; i++)
		{
			index_1 = a_index_section[every_section - 1][i];
			index_2 = a_index_section[0][i + 1];
			boundary[i + 1] = (temp_x[index_1][d] + temp_x[index_2][d]) / 2;
		}

		index_end = a_index_section[every_section - 1][section - 1];
		boundary[section] = temp_x[index_end][d];

//	printf("nn_compute\n");
		nn_compute(section_fit_index_sort, W_inhidden, W_hiddenout, every_rate);


		sum_rate[0] = every_rate[0];
		for (l = 1; l < section; l++)
		{
			sum_rate[l] = sum_rate[l - 1] + every_rate[l];
		}
//	printf("select_action\n");
		select_action(sum_rate, a_section_index);

		for (m = 0; m < every_section; m++)
		{
			one_d_select_index_x[m] = a_index_section[m][a_section_index[0]]; //取的那一个区间里的所有，方便以后再分
			select_index_f[m] = a_section_fit_index[m][a_section_index[0]];

		}


//free(a_index_section);
//printf("before free a_index_section\n");
for(i=0;i<every_section;i++)
{
	free(a_index_section[i]);
}
free(a_index_section);
//printf("free a_index_section success!!!!!!!\n");
for(i=0;i<every_section;i++)
{
	free(a_section_fit_index[i]);
}
free(a_section_fit_index);
//printf("free a_section_fit_index success!!!!!!!\n");
//free(a_section_fit_index);

/*
		a_index_section = NULL;
		a_section_fit_index = NULL;
*/
//	}
free(index_x);
//printf("free index_x success!\n");
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
//printf("dif_nn beginning! \n");
	double INIT_MIN;
	double INIT_MAX;
	
	int ArrLength;
	ArrLength = initial_popsize;
	int ArrLength_f;

	ArrLength_f = initial_popsize;
	int every_section = 100;

	int condition;
	double x[final_popsize][positionDim];
	double temp_x[final_popsize][positionDim];

	double *one_d_x;               
	int index_x[final_popsize][positionDim]; 
											  
	double *temp_test;
	int *one_d_index_x;

	double f[final_popsize];
	double *one_d_f;
	int *index_f;
	int re_index_f[final_popsize];
	double *temp_f;
	int *one_d_index_f;

	int *one_d_select_index_x; 
	int *select_index_f;
	double sel_new_x;
	double add_position[num_recurrent][positionDim];
	int new_x_flag = 0;;

	double nn_best_fit;
	double add_one_f;
	double one_x[positionDim];
	double one_x_add_position[positionDim];
	double one_f;

	int a,i,j,m,d,s,zz,l,flag;
	flag=0;

	switch (FUNNUM)
	{
		case 1:INIT_MIN = -100; INIT_MAX = 100; break;
		case 2:INIT_MIN = -100; INIT_MAX = 100; break;
		case 3:INIT_MIN = -100; INIT_MAX = 100; break;
		case 4:INIT_MIN = -100; INIT_MAX = 100; break;
		case 5:INIT_MIN = -100; INIT_MAX = 100; break;
		case 6:INIT_MIN = -100; INIT_MAX = 100; break;
		case 7:INIT_MIN = -100; INIT_MAX = 100;   break;
		case 8:INIT_MIN = -100; INIT_MAX = 100;  break;
		case 9:INIT_MIN = -100; INIT_MAX = 100;    break;
		case 10:INIT_MIN = -100; INIT_MAX = 100; break;
	}
for (i = 0; i < final_popsize; i++)
{
	f[i]=1;
}

	
	for (i = 0; i < initial_popsize; i++)
	{
		for (j = 0; j < positionDim; j++)
		{
			//x[i][j] = (INIT_MAX - INIT_MIN)*     rand() / (double)RAND_MAX      + INIT_MIN; 


			x[i][j]=(        (    (double)rand() / RAND_MAX +0.00001   )   *2-1        )*100;
			one_x[j] = x[i][j];
		}
		fitnessf(one_x,positionDim,FUNNUM,&one_f);
		f[i] = one_f;
	}
//printf("f[1]=%lf\n",f[1]);
	for (a = 0; a < all_recurrent; a++)    
	{
		ArrLength = initial_popsize + a * num_recurrent;  
		one_d_x = (double(*))malloc(ArrLength * sizeof(double));  
		temp_test = (double(*))malloc(ArrLength * sizeof(double));
		one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));

		for (d = 0; d < positionDim; d++)  
		{
			int k;
			k = 0;
			for (i = 0; i < ArrLength; i++)
			{
				one_d_x[i] = x[i][d];  
				temp_test[i] = x[i][d];     
			}
		
			heapSort(one_d_x, ArrLength);
			for (i = 0; i < ArrLength; i++)
			{
				
				for (zz = 0; zz < ArrLength; zz++)
				{
					if (one_d_x[i] == temp_test[zz])
					{
						one_d_index_x[k++] = zz;   
						break;
					}
				}
			}

			for (i = 0; i < ArrLength; i++)
			{

				index_x[i][d] = one_d_index_x[i];   
			}
		}//d
		free(one_d_x);
		free(temp_test);
		free(one_d_index_x);



/*note emmmmmmm		
		one_d_x = NULL;
		temp_test = NULL;
		one_d_index_x = NULL;
*/
		
		ArrLength_f = initial_popsize + a * num_recurrent;  
		one_d_f = (double(*))malloc(ArrLength_f * sizeof(double));
		temp_f = (double(*))malloc(ArrLength_f * sizeof(double));
		one_d_index_f = (int(*))malloc(ArrLength_f * sizeof(int));



	/*	for (l = 0; l < ArrLength_f; l++)  //x的一行一行放进去计算适应值
		{
			for (d = 0; d < positionDim; d++)
			{
				one_x[d] = x[l][d];
			}

			fitnessf(one_x, positionDim, FUNNUM, &one_f);    //把x的每一个粒子(多维)都放进去计算适应值
			f[l] = one_f;
		}
	*/
		
		for (i = 0; i < ArrLength_f; i++)
		{
			one_d_f[i] = f[i];
			temp_f[i] = f[i];
		}

		//quicksort(one_d_f, ArrLength_f, 0, ArrLength_f - 1);    //快速排序
		heapSort(one_d_f, ArrLength_f);
		int k;
		k = 0;
		index_f = (int(*))malloc(ArrLength_f * sizeof(int));

		for (i = 0; i < ArrLength_f; i++)
		{
			for (zz = 0; zz < ArrLength_f; zz++) {
				if (one_d_f[i] == temp_f[zz]) {
					one_d_index_f[k++] = zz;   //取原序号下标索引
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

		for (s = 0; s < num_recurrent; s++)
		{
/* note
			condition = ArrLength;
			every_section = (int)floor(condition / section);  //分组：100个 10个...
			one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));
			one_d_select_index_x = (int(*))malloc(every_section * sizeof(int)); //要返回的一个维度上的x的索引序号
			select_index_f = (int(*))malloc(every_section * sizeof(int));    //要返回的f的索引序号
*/



			
			for (d = 0; d < positionDim; d++)
			{
/*note
				condition = ArrLength;
				every_section = (int)floor(condition / section);  
*/
			//ADD 1 LINE
			one_d_index_x = (int(*))malloc(ArrLength * sizeof(int));
				for (i = 0; i < ArrLength; i++)
				{
					one_d_index_x[i] = index_x[i][d];  
				}


				condition = ArrLength;

				//printf("\n before while\n");
				while (condition > input_num)
				{
				//ADD 3 LINES
				every_section = (int)floor(condition / section);
				one_d_select_index_x = (int(*))malloc(every_section * sizeof(int));
				select_index_f = (int(*))malloc(every_section * sizeof(int));

flag++;
//printf("flag= %d\n",flag);
				select_f_index_draw_rate(temp_x, f, one_d_index_x, d, index_f, W_inhidden, W_hiddenout, condition, every_section, one_d_select_index_x, select_index_f);

//printf("\n select_f_index_draw_rate over \n");

				//ADD 4 LINES
				free(one_d_index_x);
//printf("--free one_d_index_x-----");
				free(index_f);
				one_d_index_x = (int(*))malloc(every_section * sizeof(int));
				index_f = (int(*))malloc(every_section * sizeof(int));

					for (i = 0; i < every_section; i++)
					{
						one_d_index_x[i] = one_d_select_index_x[i]; 
						index_f[i] = select_index_f[i];
					}
				//ADD 2 LINES
				free(one_d_select_index_x);
				free(select_index_f);
				condition = every_section;
				//note
				//	every_section = (int)floor(condition / section);

				} //while over
//printf("while over\n");	
					int r1; 
					int r2; 
					int temp;
					r1= rand() % condition ;
					r2= rand() % condition;
					if (r1 == r2)
					{
						r2= rand() % condition;
					}
					if (r1 > r2)
					{
						temp = r1;
						r1 = r2;
						r2 = temp;
					}
				
					sel_new_x = (temp_x[one_d_index_x[r2]][d] - (temp_x[one_d_index_x[r1]][d])* rand() / (double)RAND_MAX + temp_x[one_d_index_x[r1]][d]);
					
				
				add_position[s][d] = sel_new_x;
				x[ArrLength + s][d] = sel_new_x; 
		
				//ADD 3 LINES
				free(one_d_index_x);
				free(index_f);
				index_f = (int(*))malloc(ArrLength_f * sizeof(int));
				for (j = 0; j < ArrLength_f; j++)
				{
					index_f[j] = re_index_f[j];
				}
			}//d
/*note
			free(one_d_index_x);
			free(one_d_select_index_x);
			free(select_index_f);
			one_d_index_x = NULL;
			one_d_select_index_x = NULL;
			select_index_f = NULL;
*/
		}//s

		free(index_f);
//printf("---free index_f---");
		for (s = 0; s < num_recurrent; s++)
		{
			for (d = 0; d < positionDim; d++)
			{
				one_x_add_position[d] = add_position[s][d];
			}
			fitnessf(one_x_add_position, positionDim, FUNNUM, &add_one_f);   							 
			f[ArrLength_f + s] = add_one_f;
		}

	}//all次*num_recurrent = 2000个新点

	nn_best_fit = f[0];     //找最小适应值
	for (m = 0; m < ArrLength; m++)
	{
		if (f[m] < nn_best_fit)
		{
			nn_best_fit = f[m];
		}
	}

//	printf("\ndif_nn over \n");
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
	int i,j,l,n,k;

	for (i = 0; i < m; i++)// m = nn_num
	{
		for (j = 0; j < 4; j++)
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
		for (k = 1; k < 4; k++)
		{
			if (GA_nn_fitness[select[k]] < min)
			{
				min = GA_nn_fitness[select[k]];
				maxindex = k;
			}
		}
		index[i] = maxindex;
		for (n = 0; n < nn_num; n++)
		{
			mark2[n] = 0;
		}

		for (n = 0; n < 4; n++)
		{
			select[n] = 0;
		}


		for (l = 0; l < total_weight; l++)
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
	int i;

	for (two = 0; two < nn_num; two++)
	{
		r = rand() / (RAND_MAX + 1.0);
		if (r < pc)
		{
			++first;
			if (first % 2 == 0)//交叉
			{
				point = rand() % total_weight + 1;  //随机选择交叉点
				for (i = 0; i < point; i++)
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
	int i,d;

	for (i = 0; i < nn_num; i++)
	{
		for (d = 0; d < total_weight; d++)
		{
			rand_r = rand() / (RAND_MAX + 1.0);
			if (rand_r < pm)
			{
				neural_network[i][d] = (double)rand() / RAND_MAX * 2.0 - 1.0;
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
	
	int fun_num;
	int i,j,zz,w,g,nn,l;
	char fileName[256];
	int cf_num=10;	
	
	FILE* fp = fopen("GA100results.txt", "a");
	FILE* fw = fopen("GA100bestNN.txt", "a");

	if (fp == NULL || fw == NULL)
	{
		printf("failed to open file\n");
		system("pause");
	}
	srand((unsigned)time(NULL));
	printf("this now running");

	initial_NN();  

	for (g = 0; g < MAX_GENERATION; g++)
	{
		FILE* f_temp_NN = fopen("temp_bestNN.txt", "a");
		FILE* f_temp_result = fopen("temp_result.txt", "a");
		FILE* a_generation_NN[nn_num + 1];
		//record the undated NN in this generation's NN
		for (nn = 0; nn < nn_num; nn++) 
		{
			sprintf(fileName, "NN/g%d/NN_%d.txt", g, nn);
			a_generation_NN[nn] = fopen(fileName, "a");
			if (a_generation_NN[nn] == NULL)
			{
				printf("\n Error: Cannot open input file for writing \n");
			}
			for (i = 0; i < total_weight; i++)
			{
				fprintf(a_generation_NN[nn], "%lf\n", neural_network[nn][i]);
			}
		}

		for (FUNNUM = 0; FUNNUM < benchmark_number; FUNNUM++)
		{
			if (ini_flag == 1)
			{
				if ((n_flag != positionDim) || (func_flag != FUNNUM+1))
				{
					ini_flag = 0;
				}
			}

			if (ini_flag == 0)
			{
				FILE *fpt;
				char FileName[256];
				free(M);
				free(OShift);
				free(y);
				free(z);
				free(x_bound);
				y = (double *)malloc(sizeof(double)  *  positionDim);
				z = (double *)malloc(sizeof(double)  *  positionDim);
				x_bound = (double *)malloc(sizeof(double)  *  positionDim);
				for (i = 0; i<positionDim; i++)
					x_bound[i] = 100.0;

				if (!(positionDim == 2 || positionDim == 10 || positionDim == 20 || positionDim == 30 || positionDim == 50 || positionDim == 100))
				{
					printf("\nError: Test functions are only defined for D=2,10,20,30,50,100.\n");
				}
				if (positionDim == 2 && ((FUNNUM+1 >= 17 && FUNNUM+1 <= 22) || (FUNNUM+1 >= 29 && FUNNUM+1 <= 30)))
				{
					printf("\nError: hf01,hf02,hf03,hf04,hf05,hf06,cf07&cf08 are NOT defined for D=2.\n");
				}

				/* Load Matrix M*/

				sprintf(FileName, "input_data/M_%d_D%d.txt", FUNNUM+1, positionDim);
				fpt = fopen(FileName, "r");
				if (fpt == NULL)
				{
					printf("\n Error: Cannot open input file for reading \n");
				}

				if (FUNNUM+1<20)
				{
					M = (double*)malloc(positionDim*positionDim * sizeof(double));
					if (M == NULL)
						printf("\nError: there is insufficient memory available!\n");
					for (i = 0; i<positionDim*positionDim; i++)
					{
						fscanf(fpt, "%lf", &M[i]);
					}
				}
				else
				{
					M = (double*)malloc(cf_num*positionDim*positionDim * sizeof(double));
					if (M == NULL)
						printf("\nError: there is insufficient memory available!\n");
					for (i = 0; i<cf_num*positionDim*positionDim; i++)
					{
						fscanf(fpt, "%lf", &M[i]);
					}
				}
				fclose(fpt);

				/* Load shift_data */
				sprintf(FileName, "input_data/shift_data_%d.txt", FUNNUM+1);
				fpt = fopen(FileName, "r");

				if (fpt == NULL)
				{
					printf("\n Error: Cannot open input file for reading \n");
				}

				if (FUNNUM+1<20)
				{
					OShift = (double *)malloc(positionDim * sizeof(double));
					if (OShift == NULL)
						printf("\nError: there is insufficient memory available!\n");
					for (i = 0; i<positionDim; i++)
					{
						fscanf(fpt, "%lf", &OShift[i]);
		//printf("OShift[0]---------=%lf\n",OShift[0]);

					}
				}
				else
				{
					OShift = (double *)malloc(positionDim*cf_num * sizeof(double));
					if (OShift == NULL)
						printf("\nError: there is insufficient memory available!\n");
					for (i = 0; i<cf_num - 1; i++)
					{
						for (j = 0; j<positionDim; j++)
						{
							fscanf(fpt, "%lf", &OShift[i*positionDim + j]);
						}
						fscanf(fpt, "%*[^\n]%*c");
					}
					for (j = 0; j<positionDim; j++)
					{
						fscanf(fpt, "%lf", &OShift[(cf_num - 1)*positionDim + j]);
					}

				}
				fclose(fpt);


				/* Load Shuffle_data */

				if (FUNNUM+1 >= 11 && FUNNUM+1 <= 20)
				{
					sprintf(FileName, "input_data/shuffle_data_%d_D%d.txt", FUNNUM+1, positionDim);
					fpt = fopen(FileName, "r");
					if (fpt == NULL)
					{
						printf("\n Error: Cannot open input file for reading \n");
					}
					SS = (int *)malloc(positionDim * sizeof(int));
					if (SS == NULL)
						printf("\nError: there is insufficient memory available!\n");
					for (i = 0; i<positionDim; i++)
					{
						fscanf(fpt, "%d", &SS[i]);
					}
					fclose(fpt);
				}
				else if (FUNNUM+1 == 29 || FUNNUM+1 == 30)
				{
					sprintf(FileName, "input_data/shuffle_data_%d_D%d.txt", FUNNUM+1, positionDim);
					fpt = fopen(FileName, "r");
					if (fpt == NULL)
					{
						printf("\n Error: Cannot open input file for reading \n");
					}
					SS = (int *)malloc(positionDim*cf_num * sizeof(int));
					if (SS == NULL)
						printf("\nError: there is insufficient memory available!\n");
					for (i = 0; i<positionDim*cf_num; i++)
					{
						fscanf(fpt, "%d", &SS[i]);
					}
					fclose(fpt);
				}


				n_flag = positionDim;
				func_flag = FUNNUM+1;
				ini_flag = 1;
			}

	#pragma omp parallel for num_threads(17) private(nn,i,j) 
			for (nn = 0; nn < nn_num; nn++)      //需要并行：nn_num个神经网络一起跑
			{
				double re_convert1[input_num * hidden_num] = { 0 };
				double re_convert2[hidden_num*output_num] = { 0 };
				double W_inhidden[input_num][hidden_num];
				double W_hiddenout[hidden_num][output_num];

				for (i = 0; i < input_num * hidden_num; i++)        //拆分神经网络：为了更好的计算
				{
					re_convert1[i] = neural_network[nn][i];
				}
				//printf("re_convert1[1100]%lf \n", re_convert1[1100]);

				int k = 0;
				for (i = input_num * hidden_num; i < total_weight; i++)
				{
					re_convert2[k] = neural_network[nn][i];
					k++;
				}

				for (i = 0; i < input_num; i++) 
				{
					for (j = 0; j < hidden_num; j++) 
					{
						W_inhidden[i][j] = re_convert1[i * hidden_num + j];
					}
				}

				for (i = 0; i < hidden_num; i++)
				{
					for (j = 0; j < output_num; j++)
					{
						W_hiddenout[i][j] = re_convert2[i * output_num + j];
					}
				}
				
				nn_fitness[nn][FUNNUM] = dif_nn(W_inhidden, W_hiddenout, FUNNUM + 1); 
 				//printf("nn_fitness=%lf\n",nn_fitness[0][0]);
printf("nn_fitness[%d][%d]=%lf\n",nn,FUNNUM,nn_fitness[nn][FUNNUM]);
//printf("nn=%dth,funnum=%dth,is done",nn,FUNNUM);
			} //nn_num NN 
printf("nn=%dth,funnum=%dth,is done",nn,FUNNUM);
		}//FUNNUM


		for (fun_num = 0; fun_num < benchmark_number; fun_num++)
		{
			for (nn = 0; nn < nn_num; nn++)
			{
				nn_fitness_one_f[nn] = nn_fitness[nn][fun_num];     //同一个函数下的所有神经网络的最小适应值拿出来
				temp_compare[nn] = nn_fitness[nn][fun_num];
			}
			int k = 0;
			//quicksort(nn_fitness_one_f, nn_num, 0, nn_num - 1);  //快速排序所有NN在一个函数上的fitness：排名
			heapSort(nn_fitness_one_f, nn_num);
			for (w = 0; w < nn_num; w++)
			{
				for (zz = 0; zz < nn_num; zz++)
				{
					if (nn_fitness_one_f[w] == temp_compare[zz])   //排序之后的和原来的相比，可以知道下标
					{
						record_sort_index[k++] = zz + 1;   //对应排名名次
						break;
					}
				}
			}
			for (i = 0; i < nn_num; i++)
			{
				record_nn_fitness_sort_index[i][fun_num] = record_sort_index[i]; //nn_num * fun_num个名次
			}
		}//10个函数

		for (l = 0; l < nn_num; l++)   //平均排名
		{
			double sum = 0;
			for (i = 0; i < benchmark_number; i++)
			{
				sum += record_nn_fitness_sort_index[l][i];
			}

			record_nn_fitness_sort_index[l][benchmark_number] = sum / benchmark_number;
			GA_nn_fitness[l] = sum / benchmark_number;
			printf("the %dth NN in all f's avg ranking is :%lf \n", l + 1, GA_nn_fitness[l]);
		}

	
		min_fit[g] = GA_nn_fitness[0];     //找最小排名及对应的网络
		for (nn = 1; nn < nn_num; nn++)
		{
			if (GA_nn_fitness[nn] < min_fit[g])
			{
				min_fit[g] = GA_nn_fitness[nn];
				for (i = 0; i < total_weight; i++)
				{
					GA_best_NN[i] = neural_network[nn][i];
				}			
			}
			else
			{
				for (i = 0; i < total_weight; i++)
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
		for (i = 0; i < total_weight; i++)
		{
			fprintf(a_generation_NN[nn_num], "%lf\n", GA_best_NN[i]);
		}

		printf("the generation %d min ranking is:%lf \n", g + 1, min_fit[g]);


		if (min_fit[g] < the_min_fit_inallG)
		{
			the_min_fit_inallG = min_fit[g];
			for (i = 0; i < total_weight; i++)
			{
				gbest_NN[i] = GA_best_NN[i];
				fprintf(f_temp_NN, "%lf\n", gbest_NN[i]);
			}
			record_g = g;
		}
		printf("the history min ranking is:%lf \n", the_min_fit_inallG);

		fprintf(fp, "g=%d,the best:%lf\n", g + 1, min_fit[g]);
		fprintf(fp, "history best %d:%lf\n", record_g, the_min_fit_inallG);

		selection_tournment(neural_network, GA_nn_fitness); 
		crossover(neural_network, pc);
		mutation(neural_network, pm);


		fprintf(f_temp_result, "g=%d,the best:%lf\n", g + 1, min_fit[g]);
		fprintf(f_temp_result, "history best %d:%lf\n", record_g, the_min_fit_inallG);
		fclose(f_temp_NN);
		fclose(f_temp_result);



		for (i = 0; i < nn_num+1; i++)
		{
			fclose(a_generation_NN[i]);
		}
	}//GA代
	for (i = 0; i < total_weight; i++)
	{
		fprintf(fw, "%lf\n", gbest_NN[i]);
	}
	free(y);
	free(z);
	free(M);
	free(OShift);
	free(x_bound);
	fclose(fp);
	fclose(fw);
	
}//main结束
