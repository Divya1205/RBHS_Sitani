function [S1,A]=robustPCA_v3(iterations,X,lambda1,lambda2,ran)
%% ||X - (A+S)||F + ||S||l1 + ||A||*

[m,n]=size(X);
r=ran;
LM=rand(m,r);
RM=rand(n,r);
A=LM*RM';
 %lambda1=lambda2;
%%




         S1 = wthresh((X-A),'s',lambda2);
        
        for loop=1:iterations
%for i=1:200
            [U,S_11,V] = svd(X-S1);
            Sigma = max(0,S_11-lambda1);
            A = U*Sigma*V';
  % S1 = sign(X-L1)*max(0,abs(X-L1) - lambda2);
            S1 = wthresh((X-A),'s',lambda2);
        end
        hotspot_loc=[];
        label_pred=[];
%        for col=1:size(S1,2) 
%            
%        end


       
end
