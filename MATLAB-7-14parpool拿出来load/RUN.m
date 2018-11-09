%  parpool open local 2
clear all;
global input_num hidden_num output_num nn_num benchmark_number section
global initial_flag fitcount positionDim

input_num = 30;
hidden_num = 40;
output_num = 10;
positionDim = 2;
nn_num = 5;           %20����ͬ��������
benchmark_number = 10; %10�����Ժ���
initial_flag=0;
fitcount = 0;
section = 10;
MAX_GENERATION=10;
pc=0.7;
pm=0.1;
for nn=1:nn_num   %��ʼ��������
    neural_network(nn,:) = initial_NN();
end

load fbias_data;    %f_bias

for g=1:MAX_GENERATION
    for FUNNUM=1:benchmark_number       %���Եĺ������
        switch FUNNUM
            case 1
                INIT_MIN = -100; INIT_MAX = 100;                    %���Ժ����ķ�Χ
                load sphere_func_data   %  o
                M=0;
                bias=f_bias(FUNNUM);
            case 2
                INIT_MIN = -100; INIT_MAX = 100;                    %���Ժ����ķ�Χ
                load schwefel_102_data
                M=0;
                bias=f_bias(FUNNUM);
            case 3
                INIT_MIN = -100; INIT_MAX = 100;                    %���Ժ����ķ�Χ
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
                INIT_MIN = -100; INIT_MAX = 100;                    %���Ժ����ķ�Χ
            case 5
                INIT_MIN = -0.5; INIT_MAX = 0.5;                    %���Ժ����ķ�Χ
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
                INIT_MIN = -100; INIT_MAX = 100;                    %���Ժ����ķ�Χ
                load rosenbrock_func_data
                M=0;
                bias=f_bias(FUNNUM);
            case 7
                INIT_MIN = 0; INIT_MAX = 600;                    %���Ժ����ķ�Χ
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
                INIT_MIN = -32; INIT_MAX = 32;                    %���Ժ����ķ�Χ
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
                INIT_MIN = -5; INIT_MAX = 5;                    %���Ժ����ķ�Χ
                load rastrigin_func_data
                M=0;
                bias=f_bias(FUNNUM);
            case 10
                INIT_MIN = -100; INIT_MAX = 100;                    %���Ժ����ķ�Χ
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
        parfor nn=1:nn_num    %�����ò��гأ�
            re_convert1 = neural_network(nn,1:input_num*hidden_num);
            re_convert2 = neural_network(nn,(input_num*hidden_num)+1:end);
            W_inhidden = reshape(re_convert1,hidden_num,input_num)';
            W_hiddenout = reshape(re_convert2,output_num,hidden_num)';
            nn_fitness(nn,FUNNUM) = dif_nn(W_inhidden,W_hiddenout,FUNNUM,INIT_MIN,INIT_MAX,o,M,bias);  %1�������������к������в�ͬ��fitness������λ��x������ģ�
            fprintf('66666666\n');
        end
    end
    for fun_num=1:benchmark_number         %10�����Ժ���ʵ��
        [sort_fit,record_sort_index(:,fun_num)]= sort(nn_fitness(:,fun_num));%��20��������һ�������µ�fitness��������
        for nn=1:nn_num
            pre_th = record_sort_index(nn,fun_num);  %������1-20�����������ԭ�±�
            record_nn_fitness_sort_index(pre_th,fun_num) = nn;  %ԭ�������ڵ�fun_num�������µ�����
        end
    end
    for i=1:nn_num
        record_nn_fitness_sort_index(i,benchmark_number+1) = mean(record_nn_fitness_sort_index(i,1:benchmark_number)); %��benchmark_number+1����ƽ������ֵ~nn��Ӧֵ
        fprintf('��%d��GA����%d��������, ��%d��������ƽ������Ϊ�� %e \n',g,i,benchmark_number,record_nn_fitness_sort_index(i,benchmark_number+1));
    end
    min_fit(g)=min(record_nn_fitness_sort_index(:,benchmark_number+1));
    fprintf('��%d��GA����%d����������С����Ϊ�� %e \n',g,benchmark_number,min_fit(g));
    
    selnn = selection_tournment(neural_network,record_nn_fitness_sort_index(:,benchmark_number+1));
    cronn= crossover(selnn,pc);                      %����
    newnn = mutation(cronn,pm);                      %����
    neural_network = newnn;  
end
% parpool close;