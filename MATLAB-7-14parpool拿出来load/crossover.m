% 交叉
function cropop=crossover(x,pc)          %pc=0.5
[m,D]=size(x);
cropop = x;
flag=zeros(1,m);
for i=1:m
    r0=floor(rand*m+1);%randi
    r1=floor(rand*m+1);
    if(flag(r0)==1)
        r0=floor(rand*m+1);
    end
    if(flag(r1)==1||r1==r0)
        r1=floor(rand*m+1);
    end
    while (flag(r0)==0 ||flag(r1)==0)
        if(rand<pc)
            for j=1:D
                if(rand<pc)
                    cropop(r0,j)=x(r1,j);
                    cropop(r1,j)=x(r0,j);
                else
                    cropop(r0,j)=x(r0,j);
                    cropop(r1,j)=x(r1,j);
                end
                flag(r0)=1;
                flag(r1)=1;
            end
        end
    end
end
end
% for i=1:m-1   %将相邻的两个个体进行交叉
%         if(rand<pc)
%                 cpoint=round(rand*D);
%                 newpop(i,:)=[x(i,1:cpoint),x(i+1,cpoint+1:D)];
%                 newpop(i+1,:)=[x(i+1,1:cpoint),x(i,cpoint+1:D)];
%         else
%                 newpop(i,:)=x(i,:);
%                 newpop(i+1,:)=x(i+1,:);
%         end
% end
