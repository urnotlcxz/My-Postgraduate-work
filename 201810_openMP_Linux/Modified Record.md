> * the 7th row:  #include <omp.h> 加不加好像都可以？
> * the 894-1043th row:  把测试函数要读的数据提前拿出来，防止并行时同时读文件的错误
> * the 1045th row:    #pragma omp parallel for num_threads(17) private(nn,i,j)


## init_x_100.c
> * 初始撒点从1000变为了100个点
> * 而且所有函数的、所有的种群用的这100个点是完全一模一样的
> * 在外面初始化的