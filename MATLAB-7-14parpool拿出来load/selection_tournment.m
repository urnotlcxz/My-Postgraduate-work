function selpop=selection_tournment(x,fit) %������ѡ��
[m,D]=size(x);  %m������ 
selpop = x;
select=zeros(4,1);%�ӽ�����ȡ������ʱ��ŵ�����,4��һ��ѡ��,��ʼ��
mark2=zeros(m,1); %�������,��ʼ��
index=zeros(m,1);%������ȡ�������±�,��ʼ��
for i=1:m  
    for j=1:4   %����4���γ�1�飬ÿ��ȡ��õ�һ��
        r2=floor(rand*m+1); %1~m ����
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
    mark2=zeros(m,1);%ûѡһ�ζ����
    select=zeros(4,1);
    selpop(i,:) = x(index(i),:);
end
end



