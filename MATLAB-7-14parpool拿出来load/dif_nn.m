function  [nn_fitness] = dif_nn(W_inhidden,W_hiddenout,FUNNUM,INIT_MIN,INIT_MAX,o,M,bias)
global fitcount
positionDim=2;
best_value=0;
POP_SIZE = 100;  %一开始x有1000个，以后慢慢叠加
for i=1:POP_SIZE                         %随机产生初始位置
    for j=1:positionDim
        x(i,j)=rand*(INIT_MAX-INIT_MIN)+INIT_MIN;  %C语言生成0-1的数是：
    end
end

num_recurrent = 100;
for all=1:30   %循环20次这个过程
    for d=1:positionDim   % 对所有粒子的不同维度进行从小到大排序，并记录原序号
        [sort_X(:,d),index_x(d,:)] = sort(x(:,d));
    end
    [POP_SIZE,~] = size(sort_X);  %~没有用到所以可以写成~
%     f = zeros(POP_SIZE,1);      %同样因为转换的时候说必须得定义，虽然并不知道这样对不对
%     sort_f = zeros(POP_SIZE,positionDim);
    for j=1:POP_SIZE
        f(j)=fitnessf(x(j,:),FUNNUM,o,M,bias);  %初始位置的适应值    fitnessf先不用改
        fitcount = fitcount+1;
    end
   % 对所有粒子的不同维度进行从小到大排序，并记录原序号
        [sort_f,index_f] = sort(f);  %对适应值排序之后，取排名序号

%     index_x =index_x';   
%     index_f =index_f';
    temp_x = x;    %放入原位置x 利用序号index去找
    re_index_x = index_x;
    re_index_f = index_f;
    [condition,~]= size(index_x);
    cum_pos=[];
    
    for s=1:num_recurrent  %选100次 出100个点
        for d = 1:positionDim
            while condition >= 30
                [select_index_x(d,:),select_index_f] = select_f_index_draw_rate(temp_x,f,index_x(d,:),d,index_f,W_inhidden,W_hiddenout);%temp初始，f是位置排好序之后的对应着的适应值，index_f是适应值排序的序号
                clear index_x;
%                 select_index_x = select_index_x';
                index_x(d,:) = select_index_x(d,:);
                index_f = select_index_f;
                
                clear select_index_x select_index_f;
                [condition,~]= size(index_x);
            end        
            final_d = temp_x(index_x(1,d),d)+(temp_x(index_x(end,d),d)-temp_x(index_x(1,d),d))*rand;
            add_position(s,d) = final_d;
            index_x = re_index_x;
            index_f = re_index_f;
            [condition,~]= size(index_x);
        end
    end  
  
    cum_pos=[cum_pos;add_position];
    x = [x;add_position];
    clear sort_X index_x index_f
    add_position_fit = zeros(1,num_recurrent);
    for s=1:num_recurrent
        add_position_fit(s) = fitnessf(add_position(s,:),FUNNUM,o,M,bias);
        fitcount=fitcount+1;
    end
    f = [f add_position_fit];
end     %100*20=2000个新点

 nn_best_fit = min(f);
 nn_fitness = nn_best_fit-best_value;
