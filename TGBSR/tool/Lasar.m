function [S,U,obj,obj2] = Lasar(Z_ini,W,num_cluster,lambda1,lambda2,miu,rho,max_iter)

S = Z_ini;
A = S;
for iv = 1:length(S)
    C{iv} = zeros(size(S{iv}));
end
Nsamp = size(S{1},1);
nv = length(S);
% ------------------- U -------------------- %
sum_S = 0;
for iv = 1:nv
    sum_S = sum_S + S{iv};
end
sum_S = (sum_S+sum_S')*0.5;
LSv = diag(sum(sum_S))-sum_S;
LSv = (LSv+LSv')*0.5;
try
    opts.tol = 1e-4; 
    [U,~] = eigs(LSv,num_cluster,'sa',opts);   % U: n*num_cluster
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
        opts.tol = 1e-4; 
        [U,~] = eigs(LSv, eye(size(LSv)),num_cluster,'sa',opts);
    else
        rethrow(ME);
    end
end  
for iter = 1:max_iter
    S_pre = S;
    A_pre = A;
    U_pre = U;
    if iter > 1
        % ------------------- U -------------------- %
        sum_S = 0;
        for iv = 1:nv
            sum_S = sum_S + S{iv};
        end
        sum_S = (sum_S+sum_S')*0.5;
        LSv = diag(sum(sum_S))-sum_S;
        LSv = (LSv+LSv')*0.5;
        try
            opts.tol = 1e-4; 
            [U,~] = eigs(LSv,num_cluster,'sa',opts);   % U: n*num_cluster
        catch ME
            if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
                opts.tol = 1e-4; 
                [U,~] = eigs(LSv, eye(size(LSv)),num_cluster,'sa',opts);
            else
                rethrow(ME);
            end
        end  
    end
    % ------------------ S -------------------- %
    H = EuDist2(U,U,0);
    for iv = 1:nv
        linshi_M = A{iv}-C{iv}/miu;
        linshi_S = (Z_ini{iv}.*W{iv}-lambda2*0.25*H+0.5*miu*linshi_M)./(0.5*miu+W{iv});
        Z1 = zeros(size(linshi_S));
        for is = 1:Nsamp
           ind_c = 1:Nsamp;
           ind_c(is) = [];
           Z1(is,ind_c) = EProjSimplex_new(linshi_S(is,ind_c));
        end
        S{iv} = Z1;
    end
    clear Z1 linshi_S linshi_M
    % ---------------- A ----------------- %
    S_tensor = cat(3, S{:,:});
    C_tensor = cat(3, C{:,:});
    Sv = S_tensor(:);
    Cv = C_tensor(:);
    [Av, objV] = wshrinkObj(Sv + 1/miu*Cv,lambda1/miu,[Nsamp,Nsamp,nv],0,1);
    A_tensor = reshape(Av, [Nsamp,Nsamp,nv]);
    for iv = 1:nv
        A{iv} = A_tensor(:,:,iv);
        % -------- C{iv} ------- %
        C{iv} = C{iv}+miu*(S{iv}-A{iv});
    end
    clear A_tensor S_tensor C_tensor Av Sv Cv
    miu = min(miu*rho, 1e10);
    % ------------ obj ------------ %
    diff_S = 0;
    diff_A = 0;
%     diff_U = max(abs(U(:)-U_pre(:)));
    linshi_obj  = 0;
    %% check convergence
    for iv = 1:nv
        leq{iv} = S{iv}-A{iv};
        diff_S = max(diff_S,max(abs(S{iv}(:)-S_pre{iv}(:))));
        diff_A = max(diff_A,max(abs(A{iv}(:)-A_pre{iv}(:)))); 
    end
    leqm = cat(3, leq{:,:});
    leqm2 = max(abs(leqm(:)));
    clear leqm leq

    err = max([leqm2,diff_S,diff_A]);
%     err = leqm2;

    fprintf('iter = %d, miu = %.3f, err = %.8f, difS = %.8f, diffSA = %.8f\n'...
            , iter,miu,err,diff_S,leqm2);
    obj(iter) = err;  
    obj2(iter) = leqm2;    
%     Rec_sample = reshape(E{2}(:,1),50,40);
%     imshow(Rec_sample,[]);title('E')
    if iter>5 & leqm2 < 1e-6
        iter
        break;
    end    
    
end