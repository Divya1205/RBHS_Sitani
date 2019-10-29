clc
clear
close all


%iter=iter;



%lambda=10^-1,%lambda_svm=10^(-0.5);


%tol=1e-6;
%% dataset 2
 %%%%%%%% ASA-1-5
%%%%%%%% BLOCK 6-25
%%%%%%%%% Physiochemical 26-31
%%%%%  PSSM 32-51
%%%%% Solvent exposure 52-58
A1=xlsread('/Users/sitani/Desktop/workCodesPub/New_Folder/hotspotdatasets/train_HB34.xls');
train_whole=A1(:,1:58)';
label_whole=A1(:,59);
A2=xlsread('/Users/sitani/Desktop/workCodesPub/New_Folder/hotspotdatasets/test_BID18.xlsx');
label_test2=A2(:,59);
test2=A2(:,1:58)';

%% 
trls=label_whole'+1;
ttls=label_test2'+1;
trls(:,(trls==2))=1;
ttls(:,(ttls==2))=1;
label_whole=trls';
label_test2=ttls';
clear ttls
clear trls
%%
for i = 1:size(train_whole,2)
        train_whole(:,i) = (train_whole(:,i)-mean(train_whole(:,i)))/std(train_whole(:,i));
%        train2(:,i) = (train2(:,i)-min(train2(:,i)))./(max(train2(:,i))-min(train2(:,i)));
    
end
 %%
 lambda1=1e0;
lambda2=0.3;
ran=1;

[Strain_whole,Atrain_whole]=robustPCA_v3(30,train_whole,lambda1,lambda2,ran);
%%
for i = 1:size(Atrain_whole,2)
    
 Atrain_whole(:,i) = (Atrain_whole(:,i)-mean(Atrain_whole(:,i)))/std(Atrain_whole(:,i));
%       
end

%%
k=randperm(size(train_whole,2))';
num=floor(0.8*size(train_whole,2));

train2=train_whole(:,k(1:num,:));
Atrain2=Atrain_whole(:,k(1:num,:));
label_train2=label_whole(k(1:num,:),:);

valid2=train_whole(:,k(num+1:end,:));
Avalid2=Atrain_whole(:,k(num+1:end,:));
label_valid2=label_whole(k(num+1:end,:),:);

%% grouping data according to labels..first negative and then positive data
% % train2=[train2(:,label_train2==0),train2(:,label_train2==1)];
% label_train2=[label_train2(label_train2==0,:);label_train2(label_train2==1,:)];

%% normalizing training sets

 for i = 1:size(train2,2)
        train2(:,i) = (train2(:,i)-mean(train2(:,i)))/std(train2(:,i));
%        train2(:,i) = (train2(:,i)-min(train2(:,i)))./(max(train2(:,i))-min(train2(:,i)));
    
 end
%%

for i = 1:size(valid2,2)
        valid2(:,i) = (valid2(:,i)-mean(valid2(:,i)))/std(valid2(:,i));
       %valid2(:,i) = (valid2(:,i)-min(valid2(:,i)))./(max(valid2(:,i))-min(valid2(:,i)));
     % valid2(:,i)=valid2(:,i)./norm(valid2(:,i),2);
end
%%
for i = 1:size(test2,2)
        test2(:,i) = (test2(:,i)-mean(test2(:,i)))/std(test2(:,i));
       %valid2(:,i) = (valid2(:,i)-min(valid2(:,i)))./(max(valid2(:,i))-min(valid2(:,i)));
     % valid2(:,i)=valid2(:,i)./norm(valid2(:,i),2);
end


%[S1,A]=robustPCA_v3(200,train2,1e-15,1e-2,5);
%%
lambda1=1e0;
lambda2=0.3;
ran=1;
[Stest,Atest2]=robustPCA_v3(30,test2,lambda1,lambda2,ran);
%% normalizing test  matrices

for i = 1:size(Atest2,2)
    %Avalid2(:,i) = (Avalid2(:,i)-min(Avalid2(:,i)))./(max(Avalid2(:,i))-min(Avalid2(:,i)));
        Atest2(:,i) = (Atest2(:,i)-mean(Atest2(:,i)))/std(Atest2(:,i));
%        Svalid(:,i) = (Svalid(:,i)-min(Svalid(:,i)))./(max(Svalid(:,i))-min(Svalid(:,i)));
     % valid2(:,i)=valid2(:,i)./norm(valid2(:,i),2);
end
%%

%%
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/label_train2.csv',label_train2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/Avalid2.csv',Avalid2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/label_valid2.csv',label_valid2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/Atrain2.csv',Atrain2);

csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/label_train2.csv',label_train2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/Avalid2.csv',Avalid2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/label_valid2.csv',label_valid2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/Atrain2.csv',Atrain2);

csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/label_whole.csv',label_whole);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/Atrain_whole.csv',Atrain_whole);

csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/label_test2.csv',label_test2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/Atest2.csv',Atest2);

csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/train.csv',train2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/train_whole.csv',train_whole);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/test.csv',test2);
csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/valid.csv',valid2);



%% 
fprintf("Done and saved the results")
%% working with sparse matrix
% S=abs(S1);
% % m=max(S);
% % t=mean(m);
% % threshold=0.55;
% % pos=find(m>threshold);
% 
% S=sum(S,1);
% [value,pos]=sort(S,'descend');
%%
% label_pred=zeros(size(label_train,1),1);
% label_pred(pos(1,1:170),1)=1;
%%
% [accuracy,F1_score,sensitivity,specificity,precision]=Calcmetric(label_pred,label_train);
%%
% label_train=not(label_train);
% label_valid=not(label_valid);
 t =templateSVM('Standardize',1,'KernelFunction','poly','PolynomialOrder',2);%2
% t =templateSVM('Standardize',1,'KernelFunction','linear');
%t =templateSVM('Standardize',1,'KernelFunction','rbf');

Mdl = fitcecoc(Atrain2',label_train2,'Learners',t);

[label_pred,score] = predict(Mdl,Atest2');
 correct = length(find(label_pred==label_test2));
 %percent_svm=correct/length(label) * 100;
 accuracy=correct/length(label_test2)
 %%
 [accuracy,F1_score,sensitivity,specificity,precision]=Calcmetric(label_pred,label_test2)
