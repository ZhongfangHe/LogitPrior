% Consider a linear regression: yt = xt' * b + N(0,s)
% Consider the prior bj ~ N(0, tau0 * tauj),
% tau0 ~ IG(a,b)
% tauj = 1/(1+exp(-c-d*zj)), zj~N(0,1)
% d is the key parameter to determine sparsity vs shrinkage
%
% d->0, shrink
% d->+inf, spike-and-slab


clear;
dbstop if warning;
dbstop if error;
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Feb\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Apr\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Jul\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Sep\Functions'));
% rng(123);% existing
rng(123456); %final
% rng(123456789); %verify
% rng(123211);
% rng(98765432);
% rng(37211);
% rng(222);


%% Select data (produce y,x,ind_SV)
ds = {'Simulation','Equity','Inflation'}; dsj = 1; %select dataset to be estimated
disp(['Data = ',ds{dsj}]);
if dsj == 1 %simulate data
    n = 500; %select number of observations
    K = 10; %select number of regressors
    Kzeros = round(0.1*K); %select portion of zero coefficients
    disp(['Kzeros = ', num2str(Kzeros), ', K = ', num2str(K)]);
    x = randn(n,K);
    if and(Kzeros>0, Kzeros<K)
        btrue = [0.1*ones(K-Kzeros,1); zeros(Kzeros,1)];
    elseif Kzeros == 0
        btrue = 0.1*ones(K,1);
    else
        btrue = zeros(K,1);
    end
    R2 = 0.5; %targeted R2 of regressors
    var_eps = var(x*btrue)*(1-R2)/R2; 
    eps = sqrt(var_eps)*randn(n,1);
    y = x*btrue+eps;
    ind_SV = 0;
elseif dsj == 2 %equity 
    read_file = 'Data_Equity.xlsx';
    read_sheet = 'Data';
    data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:N297');
    [ng,nr] = size(data);
    equity = data(:,1);
    reg = data(:,2:nr); 
    
    y = equity(2:ng);
    x = [ones(ng-1,1) equity(1:(ng-1)) reg(1:(ng-1),:)]; %full
    tmp = log(x(:,5)); %svar
    x(:,5) = tmp;
    ind_SV = 1;
elseif dsj == 3 %inflation
    read_file = 'Data_Inflation_LS.xlsx';
    read_sheet = 'Data2'; %change of inflation rate
    data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B3:V222');    
    [ng,nr] = size(data);
    inflation = data(:,1);
    reg = data(:,2:nr);    
    y = inflation(2:ng); %change
    uset = 1:(nr-1);
    x = [ones(ng-1,1) inflation(1:(ng-1)) reg(1:(ng-1),uset)]; %full
    ind_SV = 1;
else
    error('Wrong data selected!');
end






%% Estimate 
burnin = 1000*5;
ndraws = 5000*2;
% ind_SV = 0;
% draws = Reg_SpikeSlab_z_SA(y, x, burnin, ndraws, ind_SV);
% draws = Reg_qprior_z(y, x, burnin, ndraws, ind_SV);
% draws1 = Reg_SpikeSlab(y, x, burnin, ndraws, ind_SV);
% draws2 = Reg_qprior10(y, x, burnin, ndraws, ind_SV);
draws = Reg_logit3(y, x, burnin, ndraws, ind_SV);








