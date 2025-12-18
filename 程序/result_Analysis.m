%% ================= 结果选取值 =================
years = {'2021','2022','2023'};

result = struct('min',[],'max',[],'mean',[],'std',[],'q',[]);

for k = 1:3
    y = Pred_all(:,k);

    result(k).min  = min(y);
    result(k).max  = max(y);
    result(k).mean = mean(y);
    result(k).std  = std(y);
    result(k).q    = prctile(y,[2.5 97.5]);
end


%% ================= K-S 正态性检验 =================
alpha = 0.05;
years = {'2021','2022','2023'};

KS_result = struct('year',[],'h',[],'p',[],'mu',[],'sigma',[]);

for k = 1:3
    y = Pred_all(:,k);

    % 样本参数
    mu    = mean(y);
    sigma = std(y);

    % K-S 正态检验（基于样本均值和方差）
    [h,p] = kstest( (y-mu)/sigma, 'Alpha', alpha );

    % 记录结果
    KS_result(k).year  = years{k};
    KS_result(k).h     = h;        % h=0 接受正态性假设，h=1 拒绝
    KS_result(k).p     = p;        % p-value
    KS_result(k).mu    = mu;
    KS_result(k).sigma = sigma;
end

disp(KS_result)
