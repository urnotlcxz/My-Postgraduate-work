function  [nn_fitness] = dif_nn(W_inhidden,W_hiddenout,FUNNUM,INIT_MIN,INIT_MAX,o,M,bias)
global fitcount
positionDim=2;
best_value=0;
POP_SIZE = 100;  %һ��ʼx��1000�����Ժ���������
for i=1:POP_SIZE                         %���������ʼλ��
    for j=1:positionDim
        x(i,j)=rand*(INIT_MAX-INIT_MIN)+INIT_MIN;  %C��������0-1�����ǣ�
    end
end

num_recurrent = 100;
for all=1:30   %ѭ��20���������
    for d=1:positionDim   % ���������ӵĲ�ͬά�Ƚ��д�С�������򣬲���¼ԭ���
        [sort_X(:,d),index_x(d,:)] = sort(x(:,d));
    end
    [POP_SIZE,~] = size(sort_X);  %~û���õ����Կ���д��~
%     f = zeros(POP_SIZE,1);      %ͬ����Ϊת����ʱ��˵����ö��壬��Ȼ����֪�������Բ���
%     sort_f = zeros(POP_SIZE,positionDim);
    for j=1:POP_SIZE
        f(j)=fitnessf(x(j,:),FUNNUM,o,M,bias);  %��ʼλ�õ���Ӧֵ    fitnessf�Ȳ��ø�
        fitcount = fitcount+1;
    end
   % ���������ӵĲ�ͬά�Ƚ��д�С�������򣬲���¼ԭ���
        [sort_f,index_f] = sort(f);  %����Ӧֵ����֮��ȡ�������

%     index_x =index_x';   
%     index_f =index_f';
    temp_x = x;    %����ԭλ��x �������indexȥ��
    re_index_x = index_x;
    re_index_f = index_f;
    [condition,~]= size(index_x);
    cum_pos=[];
    
    for s=1:num_recurrent  %ѡ100�� ��100����
        for d = 1:positionDim
            while condition >= 30
                [select_index_x(d,:),select_index_f] = select_f_index_draw_rate(temp_x,f,index_x(d,:),d,index_f,W_inhidden,W_hiddenout);%temp��ʼ��f��λ���ź���֮��Ķ�Ӧ�ŵ���Ӧֵ��index_f����Ӧֵ��������
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
end     %100*20=2000���µ�

 nn_best_fit = min(f);
 nn_fitness = nn_best_fit-best_value;
