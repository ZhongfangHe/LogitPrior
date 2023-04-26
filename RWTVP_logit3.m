% Consider a TVP regression: yt = xt' * bt + N(0,st), bt = btm1 + N(0,v2)
% Consider the prior vj ~ N(0, tau0 * tauj),
% tau0 ~ IG(a,b)
% tauj = 1/(1+exp(-c-d*zj)), zj~N(0,1)
% d is the key parameter to determine sparsity vs shrinkage
%
% d->0, shrink
% d->+inf, spike-and-slab
%
% restrict c = 0
%
% Similar prior is placed for initial value b0




function draws = RWTVP_logit3(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws after burnin
%   ind_SV: an indicator if SV for measurement noise variance
%   ind_sparse: an indicator if sparsifying is performed (not applicable here; always set 0)
%   ind_forecast: an indicator if Kalman filter is run for subsequent forecasts
% Outputs:
%   draws: a structure of the final draws.


[n,K] = size(x);
minNum = 1e-10;
maxNum = 1e10;


%% Priors: scaling factor for state noise, v ~ N(0, tau0/(1+exp(-d*z)))
s0d = 100; d = sqrt(s0d)*randn;
z = randn(K,1);
tau = 1./(1+exp(-d*z)); %local para
a0 = 2; b0 = 2; tau0 = 1/gamrnd(a0,1/b0); %global para
w = tau0*tau; 
v = sqrt(w).*randn(K,1);



%% Priors: initial beta, beta0 ~ N(0, taul * diag(phil)), taul, phil are IBs
s0db = 100; db = sqrt(s0db)*randn;
zb = randn(K,1);
taub = 1./(1+exp(-db*zb)); %local para
a0b = 2; b0b = 2; tau0b = 1/gamrnd(a0b,1/b0b); %global para
wb = tau0b*taub; 
beta0 = sqrt(wb).*randn(K,1);


%% Initiaze state variance
state_var = cell(n,1);
for t = 1:n
    state_var{t} = eye(K);
end %covar matrices of state noise for simulation smoother (AA)


%% Set up adaptive MH
pstar = 0.25; %multivariate MH
tmp_const = -norminv(0.5*pstar);
KK = K + 2; %tau0,d,z
AMH_c = 1/(KK * pstar * (1-pstar)) + (1-1/KK)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;
logrw = 0;
logrw_start = logrw;
drawi_start = 0;

logrwb = 0;
logrwb_start = logrwb;
drawib_start = 0;




%% Priors: SV or constant measurement noise variance
if ind_SV == 1
    % long-run mean: p(mu) ~ N(mu0, Vmu), e.g. mu0 = 0; Vmu = 10;
    % persistence: p(phi) ~ N(phi0, Vphi)I(-1,1), e.g. phi0 = 0.95; invVphi = 0.04;
    % variance: p(sig2) ~ G(0.5, 2*sig2_s), sig2_s ~ IG(0.5,1/lambda), lambda ~ IG(0.5,1)
    muh0 = 0; invVmuh = 1/10; % mean: p(mu) ~ N(mu0, Vmu)
    phih0 = 0.95; invVphih = 1/0.04; % AR(1): p(phi) ~ N(phi0, Vphi)I(-1,1)
    priorSV = [muh0 invVmuh phih0 invVphih 0 0]'; %collect prior hyperparameters
    muh = muh0 + sqrt(1/invVmuh) * randn;
    phih = phih0 + sqrt(1/invVphih) * trandn((-1-phih0)*sqrt(invVphih),(1-phih0)*sqrt(invVphih));

    lambdah = 1/gamrnd(0.5,1);
    sigh2_s = 1/gamrnd(0.5,lambdah);
    sigh2 = gamrnd(0.5,2*sigh2_s);
    sigh = sqrt(sigh2);

    hSV = log(var(y))*ones(n,1); %initialize by log OLS residual variance.
    vary = exp(hSV);
else %Jeffery's prior p(sig2) \prop 1/sig2
    sig2 = var(y); %initialize
    vary = sig2 * ones(n,1);
end 


%% MCMC
draws.count_para = 0;
draws.logrw_para = zeros(ndraws,1); %adaptive MH for v

draws.count_parab = 0;
draws.logrw_parab = zeros(ndraws,1); %adaptive MH for beta0

draws.z = zeros(ndraws,K); 
draws.d = zeros(ndraws,1); %logit para for v

draws.zb = zeros(ndraws,K); 
draws.db = zeros(ndraws,1); %logit para for beta0

draws.tau0 = zeros(ndraws,1);
draws.tau = zeros(ndraws,K);
draws.w = zeros(ndraws,K);
draws.v = zeros(ndraws,K); %reg coef v

draws.tau0b = zeros(ndraws,1);
draws.taub = zeros(ndraws,K);
draws.wb = zeros(ndraws,K);
draws.beta0 = zeros(ndraws,K); %reg coef beta0

draws.beta = cell(K,1);
for j = 1:K
    draws.beta{j} = zeros(ndraws,n);
end %beta


if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

if ind_sparse == 1
    draws.v_sparse = zeros(ndraws,K);
    draws.beta0_sparse = zeros(ndraws,K);
    draws.beta_sparse = cell(K,1);
    for j = 1:K
        draws.beta_sparse{j} = zeros(ndraws,n);
    end
end

if ind_forecast == 1
    draws.bn_mean = zeros(ndraws,K);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K,K);
    end
    if ind_sparse == 1
        draws.bn_smean = zeros(ndraws,K);
        draws.bn_scov = cell(ndraws,1);
        for j = 1:ndraws
            draws.bn_scov{j} = zeros(K,K);
        end
    end    
end

tic;
ntotal = burnin + ndraws;
beta0_star = zeros(K,1);
para_mean = zeros(KK,1);
para_cov = zeros(KK,KK);
parab_mean = zeros(KK,1);
parab_cov = zeros(KK,KK);
for drawi = 1:ntotal  
    % Draw beta_star (AA) 
    yy = y - x*beta0;
    xx = x .* repmat(v',n,1);
    beta_star = Simulation_Smoother_DK(yy, xx, vary, state_var(2:n),...
        beta0_star, state_var{1});
    
    
    % Draw hyper-para: taul_d, phil_d
    taul_d = 1/exprnd(1/scale2 + 1/taul);
    phil_d = 1./exprnd(1 + 1./phil);
    
    
    % Draw hyper-para: taul, phil, d
    b02 = beta0.^2;
    tmp = 1/taul_d + 0.5*sum(b02./phil);
    taul = 1/gamrnd(0.5*(1+K),1/tmp);
    phil = 1./exprnd(1./phil_d + 0.5*b02/taul);
    psil = taul * phil;
    
    d0 = 0.5*q + 0.5*(1-q)*dfix;
    for j = 1:K
        if delta(j) == 1
            tmp = d0 + 0.5*v(j)*v(j);
            d(j) = 1/gamrnd(d0+0.5,1/tmp);
        else
            d(j) = 1/gamrnd(d0,1/d0);
        end
    end
    
    
    % Draw hyper_para: delta
    xstar = x.*beta_star;
    omegay = y./vary;
    vary_half = sqrt(vary);
    for j = 1:K
        delta(j) = 0;
        logprior0 = betaln(sum(delta)+1, K-sum(delta)+1);
        idx = find(delta == 1);
        if isempty(idx)
            xj0 = x;
            varj0 = psil; 
        else
            xj0 = [x xstar(:,idx')];
            varj0 = [psil; d(idx)];
        end
        nj0 = length(varj0);
        tmp = xj0./repmat(vary_half,1,nj0);
        tmp2 = tmp'*tmp;
        DDinv = diag(1./varj0) + tmp2;
        DDinv_half = chol(DDinv)';
        logdet_DDinv = 2 * sum(log(diag(DDinv_half)));
        tmp = xj0' * omegay;
        tmp2 = DDinv_half\tmp;
        tmp3 = tmp2'*tmp2;
        loglike0 = -0.5*sum(log(varj0)) - 0.5*logdet_DDinv +0.5*tmp3;
        logpost0 = logprior0 + loglike0;
        
        delta(j) = 1;
        logprior1 = betaln(sum(delta)+1, K-sum(delta)+1);
        idx = find(delta == 1);
        xj1 = [x xstar(:,idx')];
        varj1 = [psil; d(idx)];
        nj1 = length(varj1);
        tmp = xj1./repmat(vary_half,1,nj1);
        tmp2 = tmp'*tmp;
        DDinv = diag(1./varj1) + tmp2;
        DDinv_half = chol(DDinv)';
        logdet_DDinv = 2 * sum(log(diag(DDinv_half)));
        tmp = xj1' * omegay;
        tmp2 = DDinv_half\tmp;
        tmp3 = tmp2'*tmp2;
        loglike1 = -0.5*sum(log(varj1)) - 0.5*logdet_DDinv +0.5*tmp3;
        logpost1 = logprior1 + loglike1;
        
        tmp = max(logpost0, logpost1);
        p1 = exp(logpost1 - tmp);
        p0 = exp(logpost0 - tmp);
        if rand < (p1/(p0 + p1))
            delta(j) = 1;
        else
            delta(j) = 0;
        end   
    end
    
    
    % Draw q
    count_q = 0;

    q_old = q;
    qq_old = qq;
    qq_new = qq_old + exp(logrw_q) * randn;
    q_new = 1 / (1+exp(-qq_new));
    
    logprior_old = q1*log(q_old) + q2*log(1-q_old);
    logprior_new = q1*log(q_new) + q2*log(1-q_new); %p(qq)
    
    delta_sum = sum(delta);
    loglike1_old = delta_sum*log(q_old) + (K-delta_sum)*log(1-q_old);
    loglike1_new = delta_sum*log(q_new) + (K-delta_sum)*log(1-q_new); %p(delta|q)    

    d0_old = 0.5*q_old + 0.5*(1-q_old)*dfix;
    loglike2_old = K*d0_old*log(d0_old) - K*gammaln(d0_old) - (d0_old+1)*sum(log(d)) -...
        d0_old*sum(1./d);
    d0_new = 0.5*q_new + 0.5*(1-q_new)*dfix;
    loglike2_new = K*d0_new*log(d0_new) - K*gammaln(d0_new) - (d0_new+1)*sum(log(d)) -...
        d0_new*sum(1./d); %p(d|q)

    logprob = logprior_new + loglike1_new + loglike2_new ...
        - logprior_old - loglike1_old - loglike2_old;
    if log(rand) <= logprob
        q = q_new;
        qq = qq_new;
        if drawi > burnin
            count_q = 1;
        end
    end
     

    p = exp(min(0,logprob));
    logrwj = logrw_q + AMH_cq * (p - pstar_q)/drawi;   
    logrw_q = logrwj; %update proposal stdev   
   

    % Draw v, beta0 based on beta_star (AA)
    v(delta == 0) = 0;

    idx = find(delta == 1);
    if isempty(idx)
        xv = x;
        varv = psil; 
    else
        xv = [x xstar(:,idx')];
        varv = [psil; d(idx)];
    end
    nv = length(varv);
    sigy = vary_half;
    xvv = xv ./ repmat(sigy,1,nv);
    yvv = y ./ sigy;
    A_inv = diag(1./varv) + xvv' * xvv;
    if rcond(A_inv) > 1e-15
        A_inv_half = chol(A_inv);
        a_beta0v = A_inv \ (xvv' * yvv);
        beta0v = a_beta0v + A_inv_half \ randn(nv,1);
    else
        A_beta0v = robust_inv(A_inv);
        A_half = robust_chol(A_beta0v);
        a_beta0v = A_beta0v * (xvv' * yvv);
        beta0v = a_beta0v + A_half * randn(nv,1);
    end
    beta0 = beta0v(1:K);
    v(idx) = beta0v(K+1:nv);


    % ASIS for v, beta0 when delta = 1
    if ~isempty(idx)
        nidx = length(idx);
        for j = 1:nidx
            idxj = idx(j);
            
            % Compute diff_beta (SA)
            betaj = beta_star(:,idxj) * v(idxj) + beta0(idxj);
            diff_betaj = [betaj(1)-beta0(idxj); betaj(2:n)-betaj(1:n-1)];
            diff_betaj2 = diff_betaj.^2;

            % Update v2 (SA, delta = 1)
            v_sign = sign(v(idxj));
            [v2j,~] = gigrnd(0.5-0.5*n, 1/d(j), sum(diff_betaj2), 1);
            v(idxj) = v_sign * sqrt(v2j);

            % Update beta0 (SA)
            jvar_inv = 1/psil(idxj) + 1/v2j;
            jvar = 1/jvar_inv;
            jmean = jvar*betaj(1)/v2j;
            beta0(idxj) = jmean + sqrt(jvar)*randn;

            % Compute back beta_star
            beta_star(:,idxj) = (betaj - beta0(idxj)) / v(idxj);
        end
    end 
    
    
    % Residual variance
    beta = beta_star.* repmat(v',n,1) + repmat(beta0',n,1);
    yfit = sum(x .* beta,2);
    eps = y - yfit;
    if ind_SV == 1
        logz2 = log(eps.^2 + 1e-100);
        [hSV, muh, phih, sigh, sigh2_s, lambdah] = SV_update_asis(logz2, hSV, ...
            muh, phih, sigh, sigh2_s, lambdah, priorSV);    
        vary = exp(hSV); 
    else
        sig2 = 1/gamrnd(0.5*n, 2/(eps'*eps));
        vary = sig2 * ones(n,1); 
    end  
    
    
    % Sparsify beta if needed
    if ind_sparse == 1
        z = x .* beta_star;
        v_sparse = SAVS_vector(v, z); 
        
        beta0_sparse = SAVS_vector(beta0,x);
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);          
    end
    
    
    % Compute mean and covar of p(bn|y1,...,yn) for subsequent forecasts
    if ind_forecast == 1 
        v2 = v.^2;
        for t = 1:n
            state_var{t} = diag(v2);
        end %covar matrices of state noise for simulation smoother (SA)        
        P1 = state_var{1};
        a1 = beta0; %bstar1 = 0
        [bn_mean, bn_cov] = Kalman_filter_robust(y, x, ...
            vary, state_var(2:n), a1, P1);                  
        if ind_sparse == 1
            bn_smean = v_sparse.*bstarn_mean+ beta0_sparse;
            bn_scov = (v_sparse*v_sparse') .* bstarn_cov;
        end       
    end    
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K
            draws.beta{j}(i,:) = beta(:,j)';
        end   
        
        draws.q(i) = q;
        draws.logrw_q(i) = logrw_q;
        draws.count_q = draws.count_q + count_q/ndraws; %q
        
        draws.v(i,:) = v';  
        draws.delta(i,:) = delta';
        draws.d(i,:) = d'; %v        
        
        draws.beta0(i,:) = beta0';
        draws.taul(i,:) = [taul  taul_d];
        draws.phil(i,:) = [phil'  phil_d']; %beta0             
        
        draws.yfit(i,:) = yfit';                
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end        
        
        if ind_sparse == 1
            draws.v_sparse(i,:) = v_sparse';
            draws.beta0_sparse(i,:) = beta0_sparse';
            for j = 1:K
                draws.beta_sparse{j}(i,:) = beta_sparse(:,j)';
            end
        end
        
        if ind_forecast == 1
            draws.bn_mean(i,:) = bn_mean';
            draws.bn_cov{i} = bn_cov;
            if ind_sparse == 1
                draws.bn_smean(i,:) = bn_smean';
                draws.bn_scov{i} = bn_scov; 
            end            
        end        
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end


    





