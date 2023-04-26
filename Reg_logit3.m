% Consider a linear regression: yt = xt' * b + N(0,s)
% Consider the prior bj ~ N(0, tau0 * tauj),
% tau0 ~ IG(a,b)
% tauj = 1/(1+exp(-c-d*zj)), zj~N(0,1)
% d is the key parameter to determine sparsity vs shrinkage
%
% d->0, shrink
% d->+inf, spike-and-slab
%
% restrict c = 0


function draws = Reg_logit3(y, x, burnin, ndraws, ind_SV)
% Inputs:
%   y: a n-by-1 vector of target data
%   x: a n-by-K matrix of regressor data (including constant)
%   burnin: a scalar of the number of burnins
%   ndraws: a scalar of the number of effective draws after burnin
%   ind_SV: an indicator if SV for measurement noise variance
% Outputs:
%   draws: a structure of the final draws.


[n,K] = size(x);
minNum = 1e-10;
maxNum = 1e10;


%% Priors: scale d
s0d = 100; d = sqrt(s0d)*randn;


%% Priors: local comonent
z = randn(K,1);


%% Priors: global component
a = 2; b = 2; tau0 = 1/gamrnd(a,1/b);


%% Priors: reg coef
% tau = 1./(1+exp(-c-d*z)); w = tau0*tau;
% beta = sqrt(w).*randn(K,1);



%% Set up adaptive MH
% pstar = 0.44; %univariate MH
% AMH_c = 1/(pstar * (1-pstar));
% logrw_tau = 0; %tau

pstar = 0.25; %multivariate MH
tmp_const = -norminv(0.5*pstar);
KK = K + 2; %tau0,d,z
AMH_c = 1/(KK * pstar * (1-pstar)) + (1-1/KK)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;
logrw = 0;
logrw_start = logrw;
drawi_start = 0; 


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
draws.logrw_para = zeros(ndraws,1); %adaptive MH

draws.z = zeros(ndraws,K); 
draws.d = zeros(ndraws,1); %logit para

draws.tau0 = zeros(ndraws,1);
draws.tau = zeros(ndraws,K);
draws.w = zeros(ndraws,K);
draws.beta = zeros(ndraws,K); %reg coef

if ind_SV == 1
    draws.SVpara = zeros(ndraws,6); % [mu phi sig2 sig sig2_s lambda]
    draws.sig2 = zeros(ndraws,n); %residual variance
else
    draws.sig2 = zeros(ndraws,1);
end

draws.yfit = zeros(ndraws,n);

tic;
ntotal = burnin + ndraws;
para_mean = zeros(KK,1);
para_cov = zeros(KK,KK);
for drawi = 1:ntotal
    % Draw tau0,c,d,z in a single block by adaptive MH 
    count_para = 0;

    para_old = [log(tau0); d; z];
    if drawi < 100
        A = eye(KK);
    else  
        A = para_cov + 1e-6 * eye(KK) / drawi; %add a small constant
    end
    eps = mvnrnd(zeros(KK,1),A)'; %correlated normal
    para_new = para_old + exp(logrw) * eps;
    tau0_new = min(maxNum, exp(para_new(1)));
    d_new = para_new(2);
    z_new = para_new(3:KK);
    
    logprior_tau0_old = -a*log(tau0) - b/tau0;
    logprior_d_old = -0.5*d*d/s0d;
    logprior_z_old = -0.5*sum(z.*z);
    logprior_old = logprior_tau0_old + logprior_d_old ...
        + logprior_z_old;
    
    logprior_tau0_new = -a*log(tau0_new) - b/tau0_new;
    logprior_d_new = -0.5*d_new*d_new/s0d;
    logprior_z_new = -0.5*sum(z_new.*z_new);
    logprior_new = logprior_tau0_new + logprior_d_new ...
        + logprior_z_new;
    
    tau_old_raw = 1./(1+exp(-d*z));
    ww_old = tau0 * tau_old_raw;
    idx_old = find(ww_old > minNum);
    tau_old = tau_old_raw(idx_old);
    Kn = length(idx_old);
    if Kn == 0 %all coef = 0
        loglike_old = -0.5*sum(y.*y./vary);
    else %some coef ~= 0
        item1 = -0.5*Kn*log(tau0) - 0.5*sum(log(tau_old));
        sigy = sqrt(vary);
        xx = x(:,idx_old)./repmat(sigy,1,Kn);
        yy = y./sigy;
        xxtimesxx = xx'*xx;
        xxtimesyy = xx'*yy;
        BBinv = diag(1./(tau0*tau_old)) + xxtimesxx;
        BBinvbb = xxtimesyy;
        BBinv_half = chol(BBinv)';
        item2 = sum(log(diag(BBinv_half)));
        tmp = BBinv_half\BBinvbb;
        item3 = 0.5*(tmp'*tmp);
        loglike_old = item1 - item2 + item3;
    end
    
    tau_new_raw = 1./(1+exp(-d_new*z_new));
    ww_new = tau0_new * tau_new_raw;
    idx_new = find(ww_new > minNum);
    tau_new = tau_new_raw(idx_new);
    Kn = length(idx_new);
    if Kn == 0 %all coef = 0
        loglike_new = -0.5*sum(y.*y./vary);
    else %some coef ~= 0    
        item1 = -0.5*Kn*log(tau0_new) - 0.5*sum(log(tau_new));
        sigy = sqrt(vary);
        xx = x(:,idx_new)./repmat(sigy,1,Kn);
        yy = y./sigy;
        xxtimesxx = xx'*xx;
        xxtimesyy = xx'*yy;        
        BBinv = diag(1./(tau0_new*tau_new)) + xxtimesxx;
        BBinvbb = xxtimesyy;
        BBinv_half = chol(BBinv)';
        item2 = sum(log(diag(BBinv_half)));
        tmp = BBinv_half\BBinvbb;
        item3 = 0.5*(tmp'*tmp);
        loglike_new = item1 - item2 + item3;
    end

    logprob = logprior_new + loglike_new - logprior_old - loglike_old;
    if log(rand) <= logprob
        tau0 = tau0_new;
        d = d_new;
        z = z_new;
        para = para_new;
        if drawi > burnin
            count_para = 1;
        end
    else
        para = para_old;
    end     

    p = exp(min(0,logprob));
    ei = max(200, drawi/KK);
    ei_start = max(200, drawi_start/KK);
    drw = max(ei - ei_start, 20);
    logrwj = logrw + AMH_c * (p - pstar)/drw;
    if abs(logrwj - logrw_start) > 1.0986 %log(3) ~= 1.0986 
        drawi_start = drawi;
        logrw_start = logrw;
    end %restart when useful to allow for larger movement    
    logrw = logrwj; %update proposal stdev
    
    
    para_mean_old = para_mean;
    para_cov_old = para_cov;
    para_mean = (para_mean_old * (drawi-1) + para) / drawi;
    para_cov = (drawi - 1) * (para_cov_old + para_mean_old * para_mean_old') / drawi + ...
        para * para' / drawi - para_mean * para_mean'; %update the sample covariance   
         

    
    

    % Draw reg coef beta
    tau_raw = 1./(1+exp(-d*z));
    ww = tau0 * tau_raw;
    idx = find(ww > minNum);
    tau = tau_raw(idx);
    Kn = length(idx);
    if Kn == 0 %all coef = 0
        beta = zeros(K,1);
    else %some coef ~= 0 
        sigy = sqrt(vary);
        xx = x(:,idx)./repmat(sigy,1,Kn);
        yy = y./sigy;
        xxtimesxx = xx'*xx;
        xxtimesyy = xx'*yy;     
        A_inv = diag(1./(tau0*tau)) + xxtimesxx;
        A_inv_half = chol(A_inv)';
        a_beta0v = A_inv \ xxtimesyy;
        beta_nonzero = a_beta0v + A_inv_half \ randn(Kn,1);
        beta = zeros(K,1);
        beta(idx) = beta_nonzero;
%         if rcond(A_inv) > 1e-15
%             A_inv_half = chol(A_inv)';
%             a_beta0v = A_inv \ xxtimesyy;
%             beta = a_beta0v + A_inv_half \ randn(K,1);
%         else
%             A_beta0v = robust_inv(A_inv);
%             A_half = robust_chol(A_beta0v);
%             a_beta0v = A_beta0v * xxtimesyy;
%             beta = a_beta0v + A_half * randn(K,1);
%         end
    end
    
    
    

    % ASIS: compute zstar
    zstar = d * z;
    
    % ASIS: update d
    dsign = sign(d);
    d2 = gigrnd(0, 1/s0d, sum(zstar.^2), 1);
    if d2 == 0
        d2 = minNum;
    end 
    d = sqrt(d2) * dsign;
    
    % ASIS: compute back z
    z = zstar / d;    
    
    
    % Residual variance
    yfit = x*beta;
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
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin; 
        
        draws.count_para = draws.count_para + count_para/ndraws;
        draws.logrw_para(i) = logrw; %adaptive MH

        draws.z(i,:) = z'; 
        draws.d(i,:) = d; %logit para

        draws.tau0(i) = tau0;
        draws.tau(i,:) = 1./(1+exp(-d*z'));
        draws.w(i,:) = tau0./(1+exp(-d*z'));
        draws.beta(i,:) = beta'; %reg coef                    
        
        draws.yfit(i,:) = yfit';                
        
        if ind_SV == 1
            draws.sig2(i,:) = vary';
            draws.SVpara(i,:) = [muh phih sigh^2 sigh sigh2_s lambdah];
        else
            draws.sig2(i) = sig2;
        end                
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end    
end



