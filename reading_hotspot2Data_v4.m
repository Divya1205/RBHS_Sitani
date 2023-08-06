clc
clear
close all

%%%%%%%%version withoiut any normalization

%iter=iter;



%lambda=10^-1,%lambda_svm=10^(-0.5);


%tol=1e-6;
%% dataset 2
 %%%%%%%% ASA-1-5
%%%%%%%% BLOCK 6-25
%%%%%%%%% Physiochemical 26-31
%%%%%  PSSM 32-51
%%%%% Solvent exposure 52-58
%58,59,98
A1=xlsread('train_HB34.xls');
train_whole=A1(:,1:58)';
label_whole=A1(:,59);
A2=xlsread('test_BID18.xlsx');

label_test2=A2(:,59);
test2=A2(:,1:58)';
%% removing residues with high values
%test2=[test2(:,1:57),test2(:,60:97),test2(:,99:end)];
%%

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

%%
train_whole_one=train_whole(:,(label_whole==1));
label_whole_one=label_whole((label_whole==1),1);
train_whole_zero=train_whole(:,(label_whole==0));
label_whole_zero=label_whole((label_whole==0),1);
%%
% for i = 1:size(train_whole,2)
%         train_whole(:,i) = train_whole(:,i)./norm(train_whole(:,i));
% %        train2(:,i) = (train2(:,i)-min(train2(:,i)))./(max(train2(:,i))-min(train2(:,i)));
%      %train_whole(:,i) = (train_whole(:,i)-min(train_whole(:,i)))./(max(train_whole(:,i))-min(train_whole(:,i)));
% end
 %%
 lambda1=1e0;
lambda2=0.3;

%rank=1 for GBM
%ran=28; % for KNN
%ran=1; % for SVM
ran=1;
%[Strain_whole,Atrain_whole]=robustPCA_v3(200,train_whole,1e-15,1e-2,5);
[Strain_whole,Atrain_whole_temp]=robustPCA_v3(30,train_whole,lambda1,lambda2,ran);% for running classifier codes
%[Strain_whole,Atrain_whole_temp]=robustPCA_v3(20,train_whole,0.9,0.2,1); %for priting plots

%%
Atrain_whole=Atrain_whole_temp;
%%
% for i = 1:size(Atrain_whole,2)
% % Atrain_whole(:,i) = (Atrain_whole(:,i)-min(Atrain_whole(:,i)))./(max(Atrain_whole(:,i))-min(Atrain_whole(:,i)));
%  Atrain_whole(:,i) = Atrain_whole(:,i)./norm(train_whole(:,i));
% %       
% end

%%
% k=randperm(size(train_whole,2))';
% num=floor(0.8*size(train_whole,2));
% 
% train2=train_whole(:,k(1:num,:));
% Atrain2=Atrain_whole(:,k(1:num,:));
% label_train2=label_whole(k(1:num,:),:);
% 
% valid2=train_whole(:,k(num+1:end,:));
% Avalid2=Atrain_whole(:,k(num+1:end,:));
% label_valid2=label_whole(k(num+1:end,:),:);

%% grouping data according to labels..first negative and then positive data
% % train2=[train2(:,label_train2==0),train2(:,label_train2==1)];
% label_train2=[label_train2(label_train2==0,:);label_train2(label_train2==1,:)];

% %% normalizing training sets
% 
%  for i = 1:size(train2,2)
%          train2(:,i) = (train2(:,i)-mean(train2(:,i)))/(std(train2(:,i))*sqrt(57));
% %       train2(:,i) = (train2(:,i)-min(train2(:,i)))./(max(train2(:,i))-min(train2(:,i)));
%     
%  end
%%

% for i = 1:size(valid2,2)
%         valid2(:,i) = (valid2(:,i)-mean(valid2(:,i)))/std(valid2(:,i));
%        %valid2(:,i) = (valid2(:,i)-min(valid2(:,i)))./(max(valid2(:,i))-min(valid2(:,i)));
%      % valid2(:,i)=valid2(:,i)./norm(valid2(:,i),2);
% end
%%
% for i = 1:size(test2,2)
%         test2(:,i) = (test2(:,i)-mean(test2(:,i)))/(std(test2(:,i))*sqrt(57));
% %        test2(:,i) = (test2(:,i)-min(test2(:,i)))./(max(test2(:,i))-min(test2(:,i)));
%      % valid2(:,i)=valid2(:,i)./norm(valid2(:,i),2);
% end


%[S1,A]=robustPCA_v3(200,train2,1e-15,1e-2,5);
%% Using RPCA for feature selection
% S=abs(Strain_whole');
%  m=max(S);
% t=mean(m);
% 
% threshold=0;
% pos=find(m>threshold);
% 
% % 
% % S=sum(S,1);
% % [value,pos]=sort(S,'descend');
% %unimp_features=find(sum(Strain_whole,2)~=0);
% unimp_features=find(pos~=0);

%%
%lambda1=1e0;
%lambda2=0.3;
%ran=1;%for GBM

%ran=28;%for KNN
%ran=1 ;%for SVM
%ran=1;
%[Stest,Atest_whole_temp]=robustPCA_v3(200,test2,1e-15,1e-2,5);
[Stest,Atest_whole_temp]=robustPCA_v3(30,test2,lambda1,lambda2,ran);%;original
%[Stest,Atest_whole_temp]=robustPCA_v3(30,test2,0.9,0.2,1);
%% normalizing test  matrices
Atest_whole=Atest_whole_temp;
%%
% for i = 1:size(Atest_whole,2)
%    % Atest_whole(:,i) = (Atest_whole(:,i)-min(Atest_whole(:,i)))./(max(Atest_whole(:,i))-min(Atest_whole(:,i)));
%         Atest_whole(:,i) =Atest_whole(:,i)./norm(Atest_whole(:,i));
% %        Svalid(:,i) = (Svalid(:,i)-min(Svalid(:,i)))./(max(Svalid(:,i))-min(Svalid(:,i)));
%      % valid2(:,i)=valid2(:,i)./norm(valid2(:,i),2);
% end
%%
% loc=1:58;
% 
% impfeatures_pos=setdiff(loc,pos);
%  Atest2=Atest_whole(impfeatures_pos,:);
% 
Atest2=Atest_whole;
% Atrain_whole=Atrain_whole(impfeatures_pos,:);
%%
%csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/label_train2.csv',label_train2);
% csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/Avalid2.csv',Avalid2);
% csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/label_valid2.csv',label_valid2);
%csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/Classifiers_RPCA_Dataset2/Atrain2.csv',Atrain2);

%csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/label_train2.csv',label_train2);
% csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/Avalid2.csv',Avalid2);
% csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/label_valid2.csv',label_valid2);
%csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/Atrain2.csv',Atrain2);

csvwrite('/Users/sitani/Documents/CS_final_codes/workonhotspots/FinalCodesForPublication/label_whole.csv',label_whole);
csvwrite('/Users/sitani/Documents/CS_final_codes/workonhotspots/FinalCodesForPublication/Atrain_whole.csv',Atrain_whole);

csvwrite('/Users/sitani/Documents/CS_final_codes/workonhotspots/FinalCodesForPublication/label_test2.csv',label_test2);
csvwrite('/Users/sitani/Documents/CS_final_codes/workonhotspots/FinalCodesForPublication/Atest2.csv',Atest2);

%csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/train.csv',train2);
csvwrite('/Users/sitani/Documents/CS_final_codes/workonhotspots/FinalCodesForPublication/train_whole.csv',train_whole);
csvwrite('/Users/sitani/Documents/CS_final_codes/workonhotspots/FinalCodesForPublication/test.csv',test2);
% csvwrite('/Users/sitani/Desktop/CS_final_codes/workonhotspots/FinalCodesForPublication/valid.csv',valid2);



%% 
fprintf("Done and saved the results")
%%
temp=Atrain_whole;
%%
% for i = 1:size(temp,2)
% temp(:,i) =( (temp(:,i)-min(temp(:,i)))./(max(temp(:,i))-min(temp(:,i))));
% % Atrain_whole(:,i) = (Atrain_whole(:,i)-mean(Atrain_whole(:,i)))/std(Atrain_whole(:,i));
% % 
% %temp(:,i) =(temp(:,i)-mean(temp(:,i)))/(std(temp(:,i))*sqrt(57));
% end
%%
figure(1)

imagesc(temp)

colormap('jet')
ylabel('Features')
xlabel('Samples')
title('Low Rank Matrix')
ax=gca;
ax.FontSize = 14;
ax.ColorScale='linear';
ax.TitleFontWeight = 'bold';
ax.FontWeight='bold';

%%
temp1=Strain_whole;
% for i = 1:size(temp1,2)
% %temp1(:,i) = (temp1(:,i)-mean(temp1(:,i)))/(std(temp1(:,i))*sqrt(57));
% temp1(:,i) =(temp1(:,i)-min(temp1(:,i)))./(max(temp1(:,i))-min(temp1(:,i)));
% % Atrain_whole(:,i) = (Atrain_whole(:,i)-mean(Atrain_whole(:,i)))/std(Atrain_whole(:,i));
% %       
% end
figure(2)
imagesc(temp1)
colormap('jet')

ylabel('Features')
xlabel('Samples')
title('Sparse Matrix')
ax=gca;
ax.FontSize = 14;
ax.ColorScale='linear';
ax.TitleFontWeight = 'bold';
ax.FontWeight='bold';


%%
temp2=train_whole;
% for i = 1:size(temp2,2)
% temp2(:,i) = (temp2(:,i)-min(temp2(:,i)))./(max(temp2(:,i))-min(temp2(:,i)));
% %temp2(:,i) = (temp2(:,i)-mean(temp2(:,i)))/(std(temp2(:,i))*sqrt(57));
% % Atrain_whole(:,i) = (Atrain_whole(:,i)-mean(Atrain_whole(:,i)))/std(Atrain_whole(:,i));
% %       
% end

figure(3)
imagesc(temp2)
colormap('jet')

ylabel('Features')
xlabel('Samples')
title('Data Matrix')
ax=gca;
ax.FontSize = 14;
ax.ColorScale='linear';
ax.TitleFontWeight = 'bold';
ax.FontWeight='bold';


%%