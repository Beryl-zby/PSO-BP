%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行


%% 读取数据
% data = readtable('数据集_苏北.xlsx');
data = readtable('PCA_数据集_苏北.xlsx');

% 提取自变量（B–F 列：X1–X5）
X = data{:, 2:6};
% 提取因变量（G 列：需水量）
Y = data{:, 7};

% ======= 关键：只保留前 20 行有效数据 =======
X = X(1:20, :);
Y = Y(1:20, :);

%% 数据标准化（可选，通常用于提高模型收敛性）
X = zscore(X);

%% ================== 滚动预测（Rolling Forecast） ==================
% 最小训练样本数（建议 8~10）
min_train = 8;

num_obs = size(X,1);

Y_pred_roll = NaN(num_obs,1);   % 存储滚动预测值
Y_true_roll = Y;                % 真实值

for t = min_train+1 : num_obs
    % 1. 当前训练集（1 ~ t-1）
    X_tr = X(1:t-1, :);
    Y_tr = Y(1:t-1);

    % 2. 当前预测点（第 t 年）
    X_te = X(t, :);
    Y_te = Y(t);

    % 3. 加截距项
    X_tr = [ones(size(X_tr,1),1), X_tr];
    X_te = [1, X_te];

    % 4. 回归
    beta_t = (X_tr' * X_tr) \ (X_tr' * Y_tr);

    % 5. 预测
    Y_pred_roll(t) = X_te * beta_t;
end

% 只对真正发生预测的部分进行评价
idx = (min_train+1):num_obs;
Y_obs = Y_true_roll(idx);
Y_sim = Y_pred_roll(idx);

%% ================== 训练集点预测（in-sample） ==================
% 初始训练集（1 ~ min_train）
X_train0 = X(1:min_train, :);
Y_train0 = Y(1:min_train);

% 加截距项
X_train0 = [ones(size(X_train0,1),1), X_train0];

% 回归
beta_train0 = (X_train0' * X_train0) \ (X_train0' * Y_train0);

% 训练集点预测
Y_sim_train = X_train0 * beta_train0;

%% ================== Bootstrap 置信区间（90%，分位数法） ==================
%% ================== 训练集 Bootstrap 区间（90%，in-sample） ==================
B = 1000;
alpha = 0.10;

% 固定训练集：1 ~ min_train
X_train0 = X(1:min_train, :);
Y_train0 = Y(1:min_train);

n0 = size(X_train0,1);

X_train0 = [ones(n0,1), X_train0];

Y_lower_tr = NaN(n0,1);
Y_upper_tr = NaN(n0,1);

rng(2024);

for i = 1:n0

    X_te = X_train0(i,:);

    Y_boot = zeros(B,1);

    for b = 1:B
        idx_b = randi(n0, n0, 1);
        Xb = X_train0(idx_b,:);
        Yb = Y_train0(idx_b);

        beta_b = (Xb' * Xb) \ (Xb' * Yb);

        Y_boot(b) = X_te * beta_b;
    end

    Y_lower_tr(i) = quantile(Y_boot, alpha/2);
    Y_upper_tr(i) = quantile(Y_boot, 1-alpha/2);
end

%测试集
B = 1000;          % bootstrap 重采样次数
alpha = 0.10;      % 置信水平 0.90 → alpha = 0.10

Y_lower = NaN(num_obs,1);
Y_upper = NaN(num_obs,1);

rng(2024);         % 固定随机种子，保证结果可复现

for t = min_train+1 : num_obs

    % 当前 rolling 训练集
    X_tr = X(1:t-1, :);
    Y_tr = Y(1:t-1);
    X_te = X(t, :);

    n_tr = size(X_tr,1);

    % 加截距项
    X_tr = [ones(n_tr,1), X_tr];
    X_te = [1, X_te];

    % Bootstrap 预测分布
    Y_boot = zeros(B,1);

    for b = 1:B
        % 有放回抽样
        idx_b = randi(n_tr, n_tr, 1);
        Xb = X_tr(idx_b, :);
        Yb = Y_tr(idx_b);

        % 回归
        beta_b = (Xb' * Xb) \ (Xb' * Yb);

        % 对第 t 年预测
        Y_boot(b) = X_te * beta_b;
    end

    % 分位数法构造 90% 置信区间
    Y_lower(t) = quantile(Y_boot, alpha/2);        % 5%
    Y_upper(t) = quantile(Y_boot, 1 - alpha/2);    % 95%
end

% 仅保留发生预测的区段
Y_L = Y_lower(idx);
Y_U = Y_upper(idx);



%% ================== 评价指标（Rolling） ==================
% ================== 训练集点预测评价指标 ==================
MAE_train  = mean(abs(Y_train0 - Y_sim_train));
MSE_train  = mean((Y_train0 - Y_sim_train).^2);
MAPE_train = mean(abs((Y_train0 - Y_sim_train) ./ Y_train0)) * 100;
NSE_train  = 1 - sum((Y_train0 - Y_sim_train).^2) / ...
                  sum((Y_train0 - mean(Y_train0)).^2);

fprintf('\nTraining-set Point Forecast Evaluation:\n');
fprintf('MAE_train  = %.4f\n', MAE_train);
fprintf('MSE_train  = %.4f\n', MSE_train);
fprintf('MAPE_train = %.2f %%\n', MAPE_train);
fprintf('NSE_train  = %.4f\n', NSE_train);


% ================== 测试集点预测评价指标 ==================
MAE  = mean(abs(Y_obs - Y_sim));
MSE  = mean((Y_obs - Y_sim).^2);
MAPE = mean(abs((Y_obs - Y_sim) ./ Y_obs)) * 100;
NSE  = 1 - sum((Y_obs - Y_sim).^2) / sum((Y_obs - mean(Y_obs)).^2);

fprintf('Rolling Forecast Results:\n');
fprintf('MAE  = %.4f\n', MAE);
fprintf('MSE  = %.4f\n', MSE);
fprintf('MAPE = %.2f %%\n', MAPE);
fprintf('NSE  = %.4f\n', NSE);

%% ================== 区间预测评价指标（PICP / PINAW） ==================

% ================== 训练集区间评价指标 ==================

PICP_train = mean( (Y_train0 >= Y_lower_tr) & (Y_train0 <= Y_upper_tr) );
PINAW_train = mean(Y_upper_tr - Y_lower_tr) / ...
              (max(Y_train0) - min(Y_train0));

% ================== 测试集区间评价指标 ==================
PICP = mean( (Y_obs >= Y_L) & (Y_obs <= Y_U) );
PINAW = mean(Y_U - Y_L) / (max(Y_obs) - min(Y_obs));

%% ================== 输出结果 ==================

fprintf('\nTraining-set Interval Evaluation (90%% CI):\n');
fprintf('PICP_train  = %.3f\n', PICP_train);
fprintf('PINAW_train = %.3f\n', PINAW_train);

fprintf('\nInterval Forecast Evaluation (90%% CI):\n');
fprintf('PICP  = %.3f\n', PICP);
fprintf('PINAW = %.3f\n', PINAW);

%% ================== 图 1：真实值 vs 预测值（散点） ==================
figure;
scatter(Y_obs, Y_sim, 40, 'b', 'filled');
hold on;
plot([min(Y_obs) max(Y_obs)], [min(Y_obs) max(Y_obs)], '--k');
xlabel('真实值');
ylabel('预测值');
title('滚动预测：预测值 vs. 真实值');
grid on;

%% ================== 图 2：滚动预测时间序列对比 ==================
figure;
plot(idx, Y_obs, 'r-*', 'LineWidth', 1.2); hold on;
plot(idx, Y_sim, 'b-o', 'LineWidth', 1.2);
legend('真实值', '滚动预测值', 'Location', 'best');
xlabel('年份序号');
ylabel('需水量');
title(['滚动预测结果对比；MAE = ' num2str(MAE,'%.2f') ...
       ', NSE = ' num2str(NSE,'%.2f')]);
grid on;


%% ================== 图 3：Rolling + Bootstrap（90% CI） ==================
figure;
hold on;

% 置信区间带
fill([idx fliplr(idx)], ...
     [Y_L' fliplr(Y_U')], ...
     [0.85 0.85 1], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.6);

% 滚动点预测
plot(idx, Y_sim, 'b-o', 'LineWidth', 1.2);

% 真实值
plot(idx, Y_obs, 'r-*', 'LineWidth', 1.2);

legend('90% Bootstrap CI','滚动点预测','真实值','Location','best');
xlabel('年份序号');
ylabel('需水量');
title('Rolling Bootstrap 需水预测区间（90% 置信水平）');
grid on;



