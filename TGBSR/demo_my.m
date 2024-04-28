% Code for the following paper:
% 鏂囨澃锛岄鐝傦紝寮犳锛屽緪鍕囷紝鍩轰簬浣庣З寮犻噺鍥惧涔犵殑涓嶅畬鏁村瑙嗚鑱氱被锛岃嚜鍔ㄥ寲瀛︽姤,2021.
% written by Jie Wen
%  Any problems, please contact jiewen_pr@126.com
clear memory
clear all
% clc
warning off
addpath("tool")
addpath('./twist');
addpath('./data');
Dataname = 'yaleA_3view';
% percentDel = 0.3; 琛ㄧず鍒犻櫎澶氬皯鏁版嵁 0.1銆?0.3銆?0.5銆?
% lambda1 = 0.1;
% lambda2 = 0.01;
percentDel = 0.1;
f = 1;
load(Dataname);
Datanfold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Datanfold);
ind_folds = folds{f}; 
clear new_folds KnnGraph
truthF = Y+1;
clear Y
numClust = length(unique(truthF));
fp_r_result= [];
result = [];
lambda = [];
num_k = [];
Final = [];


          
% pre_lambda1 = [1e-4,1e-3,1e-2,1e-1,1e1,1e2,1e3,1e4];
% pre_lambda2 = [1e-4,1e-3,1e-2,1e-1,1e1,1e2,1e3,1e4];
% % pre_k_1 = [1,3,5,7,9,11,13,15];%,6,7,8,9,10,11,12,13,14,15,16
% 
pre_lambda1 = [0.01];
pre_lambda2 = [1e-4];
pre_k_1 = [7];

% pre_lambda1 = [0.02];
% pre_lambda2 = [1e-4];
% pre_k_1 = [1];
for idx_1 = 1 : numel(pre_lambda1)
    for idx_2 = 1 : numel(pre_lambda2)
            for idk_1  = 1 : numel(pre_k_1)
                    lambda1 = pre_lambda1(idx_1);
                    lambda2 = pre_lambda2(idx_2);
                    k_1 = pre_k_1(idk_1);
                    temp1 = [lambda1,lambda2];
                    lambda = [lambda;temp1];
                    temp2 = [k_1];
                    num_k = [num_k;temp2];
                    for iv = 1:length(X)
                        X1 = X{iv}';
                        X1 = NormalizeFea(X1,0);  % 0-column 1-row
                        ind_0 = find(ind_folds(:,iv) == 0);
                        ind_1 = find(ind_folds(:,iv) == 1);
                        % ------------- 鏋勯?犵己澶辫瑙掔殑绱㈠紩鐭╅樀 ----------- %
                        linshi_W = eye(size(X{iv},1));
                        linshi_W(:,ind_0) = [];
                        W{iv} = linshi_W*linshi_W';
                        % ---------- 鍒濆KNN鍥炬瀯寤? ----------- %
                        X1(:,ind_0) = [];
                         %璁剧疆Z鐨勫弬鏁?
                         options1 = [];
                         options1.NeighborMode = 'KNN';
                         options1.k = k_1;
                         options1.WeightMode = 'Binary';      % Binary  HeatKernel  Cosine
%                          options.t = 5;
                         Z1 = full(constructW(X1',options1));
                         Z1 = Z1- diag(diag(Z1));
                         linshi_W = diag(ind_folds(:,iv));
                         linshi_W(:,ind_0) = [];
                         Z_ini{iv} = linshi_W*max(Z1,Z1')*linshi_W';
                         clear Z1 linshi_W    
                    end
                    clear X1 ind_0 ind_1
                    max_iter = 120;
                    miu = 1e-2;
                    rho = 1.2;
%                     [Z,U,obj,obj2] = Lasar(Z_ini,W,numClust,lambda1,lambda2,miu,rho,max_iter);
                    [Z,U,obj,obj2,convegence_value] = LasarMy(Z_ini,W,numClust,lambda1,lambda2,miu,rho,max_iter);
                    convegence_value = convegence_value';
                    Fng_U = NormalizeFea(U,1);

                    clear  U
                    pre_labels_U = kmeans(real(Fng_U),numClust,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
                   
                    result_cluster_U = ClusteringMeasure(truthF, pre_labels_U)*100;
                    [result_cluster_U lambda1 lambda2 options1.k];
                    result = [result;result_cluster_U];
                    fp_r = compute_f(truthF, pre_labels_U);
                    fp_r_result = [fp_r_result;fp_r];
                    Final = [lambda,num_k,result,fp_r_result];
            end
    end
     
end
result
fp_r*100
% folderName = 'convegence_image';
% % 检查文件夹是否存在，如果不存在则创建
% if ~exist(folderName, 'dir')
%     mkdir(folderName);
% end
% 
% x=1:length(convegence_value);
% set(gca,'FontName','Times New Roman','FontSize',10);
% set(gca,'XTick',[0:10:100])
% plot(x,convegence_value,"o-")
% 
% 
% xlabel('iteration');
% ylabel('stop criteria');
% title('MSRCV1');
% saveas(gcf, fullfile(folderName, 'MSRCV10.5_convergence.jpg'));

                             
                            
                                

                                
                                
                                
                                


