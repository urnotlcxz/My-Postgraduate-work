% 变异
function newpop=mutation(x,pm)
[m,D]=size(x);
newpop = ones(size(x));

for i=1:m
    for d=1:D
        if rand<pm
            newpop(i,d) = 2 * rand - 1;
        else
            newpop(i,d)=x(i,d);
        end
    end 
end


% for i=1:m
%     if(rand<pm)
%         mpoint=round(rand*D);     %产生的变异点在1-62之间
%         if mpoint<=0
%             mpoint=1;          %变异位置
%         end
%         newpop(i,:)=x(i,:);
%         if(mpoint<proDim+1)
%               newpop(i,mpoint) = rand*(INIT_MAX-INIT_MIN)+INIT_MIN;
%         else
%              newpop(i,mpoint) = rand*2-1;
%         end
%     else
%         newpop(i,:)=x(i,:);
%     end
% end
