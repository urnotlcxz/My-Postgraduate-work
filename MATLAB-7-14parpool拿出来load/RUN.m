%  parpool open local 2
clear all;
global input_num hidden_num output_num nn_num benchmark_number section
global initial_flag fitcount positionDim

input_num = 30;
hidden_num = 40;
output_num = 10;
positionDim = 2;
nn_num = 5;           %20个不同的神经网络
benchmark_number = 10; %10个测试函数
initial_flag=0;
fitcount = 0;
section = 10;
MAX_GENERATION=10;
pc=0.7;
pm=0.1;
for nn=1:nn_num   %初始化神经网络
    neural_network(nn,:) = initial_NN();
end

load fbias_data;    %f_bias

for g=1:MAX_GENERATION
    for FUNNUM=1:benchmark_number       %测试的函数标号
        switch FUNNUM
            case 1
                INIT_MIN = -100; INIT_MAX = 100;                    %测试函数的范围
                load sphere_func_data   %  o
                M=0;
                bias=f_bias(FUNNUM);
            case 2
                INIT_MIN = -100; INIT_MAX = 100;                    %测试函数的范围
                load schwefel_102_data
                M=0;
                bias=f_bias(FUNNUM);
            case 3
                INIT_MIN = -100; INIT_MAX = 100;                    %测试函数的范围
                load high_cond_elliptic_rot_data
                if positionDim==2,load elliptic_M_D2,    % M
                elseif positionDim==10,load elliptic_M_D10,
                elseif positionDim==30,load elliptic_M_D30,
                elseif positionDim==50,load elliptic_M_D50,
                else
                    A=normrnd(0,1,D,D);
                    [M,r]=cGram_Schmidt(A);
                end
                bias=f_bias(FUNNUM);
            case 4
                INIT_MIN = -100; INIT_MAX = 100;                    %测试函数的范围
            case 5
                INIT_MIN = -0.5; INIT_MAX = 0.5;                    %测试函数的范围
                load weierstrass_data
                if positionDim==2,load weierstrass_M_D2,
                elseif positionDim==10,load weierstrass_M_D10,
                elseif positionDim==30,load weierstrass_M_D30,
                elseif positionDim==50,load weierstrass_M_D50,
                else
                    M=rot_matrix(D,c);
                end
                
                bias=f_bias(11);
            case 6
                INIT_MIN = -100; INIT_MAX = 100;                    %测试函数的范围
                load rosenbrock_func_data
                M=0;
                bias=f_bias(FUNNUM);
            case 7
                INIT_MIN = 0; INIT_MAX = 600;                    %测试函数的范围
                load griewank_func_data
                if positionDim==2,load griewank_M_D2,
                elseif positionDim==10,load griewank_M_D10,
                elseif positionDim==30,load griewank_M_D30,
                elseif positionDim==50,load griewank_M_D50,
                else
                    M=rot_matrix(D,c);
                    M=M.*(1+0.3.*normrnd(0,1,D,D));
                end
                bias=f_bias(FUNNUM);
            case 8
                INIT_MIN = -32; INIT_MAX = 32;                    %测试函数的范围
                load ackley_func_data
                if positionDim==2,load ackley_M_D2,
                elseif positionDim==10,load ackley_M_D10,
                elseif positionDim==30,load ackley_M_D30,
                elseif positionDim==50,load ackley_M_D50,
                else
                    M=rot_matrix(D,c);
                end
                bias=f_bias(FUNNUM);
            case 9
                INIT_MIN = -5; INIT_MAX = 5;                    %测试函数的范围
                load rastrigin_func_data
                M=0;
                bias=f_bias(FUNNUM);
            case 10
                INIT_MIN = -100; INIT_MAX = 100;                    %测试函数的范围
                load E_ScafferF6_func_data
                if positionDim==2,load E_ScafferF6_M_D2,
                elseif positionDim==10,load E_ScafferF6_M_D10,
                elseif positionDim==30,load E_ScafferF6_M_D30,
                elseif positionDim==50,load E_ScafferF6_M_D50,
                else
                    M=rot_matrix(D,c);
                end
                bias=f_bias(14);
        end
        parfor nn=1:nn_num    %不能用并行池？
            re_convert1 = neural_network(nn,1:input_num*hidden_num);
            re_convert2 = neural_network(nn,(input_num*hidden_num)+1:end);
            W_inhidden = reshape(re_convert1,hidden_num,input_num)';
            W_hiddenout = reshape(re_convert2,output_num,hidden_num)';
            nn_fitness(nn,FUNNUM) = dif_nn(W_inhidden,W_hiddenout,FUNNUM,INIT_MIN,INIT_MAX,o,M,bias);  %1个神经网络在所有函数下有不同的fitness（根据位置x来计算的）
            fprintf('66666666\n');
        end
    end
    for fun_num=1:benchmark_number         %10个测试函数实验
        [sort_fit,record_sort_index(:,fun_num)]= sort(nn_fitness(:,fun_num));%对20个网络在一个函数下的fitness进行排序
        for nn=1:nn_num
            pre_th = record_sort_index(nn,fun_num);  %排名第1-20个的神经网络的原下标
            record_nn_fitness_sort_index(pre_th,fun_num) = nn;  %原神经网络在第fun_num个函数下的排名
        end
    end
    for i=1:nn_num
        record_nn_fitness_sort_index(i,benchmark_number+1) = mean(record_nn_fitness_sort_index(i,1:benchmark_number)); %第benchmark_number+1列是平均排名值~nn适应值
        fprintf('第%d代GA，第%d个神经网络, 在%d个函数的平均排名为： %e \n',g,i,benchmark_number,record_nn_fitness_sort_index(i,benchmark_number+1));
    end
    min_fit(g)=min(record_nn_fitness_sort_index(:,benchmark_number+1));
    fprintf('第%d代GA，在%d个函数的最小排名为： %e \n',g,benchmark_number,min_fit(g));
    
    selnn = selection_tournment(neural_network,record_nn_fitness_sort_index(:,benchmark_number+1));
    cronn= crossover(selnn,pc);                      %交叉
    newnn = mutation(cronn,pm);                      %变异
    neural_network = newnn;  
end
% parpool close;