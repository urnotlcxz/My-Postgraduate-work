function selpop=selection_tournment(x,fit) %锦标赛选择
[m,D]=size(x);  %m是行数 
selpop = x;
select=zeros(4,1);%从锦标赛取出来临时存放的数组,4个一组选，,初始化
mark2=zeros(m,1); %标记数组,初始化
index=zeros(m,1);%最终提取出来的下标,初始化
for i=1:m  
    for j=1:4   %比如4个形成1组，每组取最好的一个
        r2=floor(rand*m+1); %1~m 个体
%         if(mark2(r2)==1)
        while(mark2(r2)==1)
            r2=floor(rand*m+1);  
        end
        mark2(r2)=1;
        select(j)=r2;
    end
    min=fit(select(1));
    maxindex=select(1);
    for k=2:4
        if(fit(select(k))<min)
            min=fit(select(k));
            maxindex=k;
        end
    end
    index(i)=maxindex;
    mark2=zeros(m,1);%没选一次都清空
    select=zeros(4,1);
    selpop(i,:) = x(index(i),:);
end
end



