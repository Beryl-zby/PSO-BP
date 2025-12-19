%% ================= 导入 =================
load('PSO_BP_40runs_Result.mat');
Pred_all = Result.Pred_all;   % 40 × 3

%% ================= 有效样本统计 =================
valid_idx = ~any(isnan(Pred_all), 2);
N_valid   = sum(valid_idx);

disp(['有效预测样本数 = ', num2str(N_valid), ...
      ' / ', num2str(size(Pred_all,1))]);

%% ================= 结果统计 =================
years = {'2021','2022','2023'};
result = struct('min',[],'max',[],'mean',[],'std',[],'q',[]);

for k = 1:3
    y = Pred_all(valid_idx, k);   % ★ 只用有效样本

    result(k).min  = min(y);
    result(k).max  = max(y);
    result(k).mean = mean(y);
    result(k).std  = std(y);
    result(k).q    = prctile(y,[2.5 97.5]);
end

%% ================= K-S 正态性检验 =================
alpha = 0.05;
KS_result = struct('year',[],'h',[],'p',[],'mu',[],'sigma',[]);

for k = 1:3
    y = Pred_all(valid_idx, k);   % ★ 只用有效样本

    mu    = mean(y);
    sigma = std(y);

    [h,p] = kstest((y-mu)/sigma, 'Alpha', alpha);

    KS_result(k).year  = years{k};
    KS_result(k).h     = h;
    KS_result(k).p     = p;
    KS_result(k).mu    = mu;
    KS_result(k).sigma = sigma;
end

disp(KS_result)

%% ================= 保存 =================
Analysis.years      = years;
Analysis.result     = result;
Analysis.KS_result  = KS_result;
Analysis.alpha      = alpha;
Analysis.N_valid    = N_valid;

save('PSO_BP_Analysis.mat', 'Analysis');
