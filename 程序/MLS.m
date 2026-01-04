%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行


%% 读取数据
data = readtable('数据集_苏北.xlsx');

% 提取自变量（B–F 列：X1–X5）
X = data{:, 2:13};
% 提取因变量（G 列：需水量）
Y = data{:, 14};

% ======= 关键：只保留前 20 行有效数据 =======
X = X(1:20, :);
Y = Y(1:20, :);

%% 数据标准化（可选，通常用于提高模型收敛性）
X = zscore(X);

%% 划分数据集
train_ratio = 0.8;  % 训练集比例
train_size = floor(train_ratio * size(X, 1));
X_train = X(1:train_size, :);
Y_train = Y(1:train_size, :);
X_test = X(train_size+1:end, :);
Y_test = Y(train_size+1:end, :);

%% 添加常数项
X_train = [ones(size(X_train, 1), 1), X_train];  % 使得模型中包含截距项

%% 使用最小二乘法计算回归系数
beta = (X_train' * X_train) \ (X_train' * Y_train);

%% 对测试集进行预测
X_test = [ones(size(X_test, 1), 1), X_test];  % 添加常数项
Y_psed = X_test * beta;  % 预测值

%% ================== Bootstrap 参数 ==================
B = 1000;          % 重采样次数
alpha = 0.10;      % 90% 置信区间
rng(2024);         % 固定随机种子

%% ================== 训练集 Bootstrap 区间（90%） ==================
n_tr = size(X_train,1);

Y_boot_tr = zeros(n_tr, B);

for b = 1:B
    % 有放回抽样
    idx_b = randi(n_tr, n_tr, 1);
    Xb = X_train(idx_b,:);
    Yb = Y_train(idx_b);

    beta_b = (Xb' * Xb) \ (Xb' * Yb);

    Y_boot_tr(:,b) = X_train * beta_b;
end

Y_L_tr = quantile(Y_boot_tr, alpha/2, 2);
Y_U_tr = quantile(Y_boot_tr, 1-alpha/2, 2);

%% ================== 测试集 Bootstrap 区间（90%） ==================
n_te = size(X_test,1);

Y_boot_te = zeros(n_te, B);

for b = 1:B
    idx_b = randi(n_tr, n_tr, 1);
    Xb = X_train(idx_b,:);
    Yb = Y_train(idx_b);

    beta_b = (Xb' * Xb) \ (Xb' * Yb);

    Y_boot_te(:,b) = X_test * beta_b;
end

Y_L_te = quantile(Y_boot_te, alpha/2, 2);
Y_U_te = quantile(Y_boot_te, 1-alpha/2, 2);


%% ================== 点预测评价指标 ==================

% —— 训练集 —— %
Y_sim_train = X_train * beta;

MAE_train  = mean(abs(Y_train - Y_sim_train));
MSE_train  = mean((Y_train - Y_sim_train).^2);
MAPE_train = mean(abs((Y_train - Y_sim_train) ./ Y_train)) * 100;
NSE_train  = 1 - sum((Y_train - Y_sim_train).^2) / ...
                  sum((Y_train - mean(Y_train)).^2);

% —— 测试集 —— %
MAE_test  = mean(abs(Y_test - Y_psed));
MSE_test  = mean((Y_test - Y_psed).^2);
MAPE_test = mean(abs((Y_test - Y_psed) ./ Y_test)) * 100;
NSE_test  = 1 - sum((Y_test - Y_psed).^2) / ...
                 sum((Y_test - mean(Y_test)).^2);

fprintf('\nPoint Forecast Evaluation:\n');
fprintf('Train: MAE=%.3f, MSE=%.3f, MAPE=%.2f%%, NSE=%.3f\n', ...
        MAE_train, MSE_train, MAPE_train, NSE_train);
fprintf('Test : MAE=%.3f, MSE=%.3f, MAPE=%.2f%%, NSE=%.3f\n', ...
        MAE_test, MSE_test, MAPE_test, NSE_test);


%% ================== 区间预测评价指标（PICP / PINAW） ==================

% —— 训练集 —— %
PICP_train = mean( (Y_train >= Y_L_tr) & (Y_train <= Y_U_tr) );
PINAW_train = mean(Y_U_tr - Y_L_tr) / ...
              (max(Y_train) - min(Y_train));

% —— 测试集 —— %
PICP_test = mean( (Y_test >= Y_L_te) & (Y_test <= Y_U_te) );
PINAW_test = mean(Y_U_te - Y_L_te) / ...
             (max(Y_test) - min(Y_test));

fprintf('\nInterval Forecast Evaluation (90%% CI):\n');
fprintf('Train: PICP=%.3f, PINAW=%.3f\n', PICP_train, PINAW_train);
fprintf('Test : PICP=%.3f, PINAW=%.3f\n', PICP_test, PINAW_test);


%% ================== 绘制真实值–预测值散点图 ==================
sz = 25;      % 点大小
c  = 'b';     % 颜色

%% —— 1. 训练集散点图 ——
figure;
scatter(Y_train, Y_train, sz, c, 'filled');
hold on;
plot(xlim, xlim, '--k');   % y = x 参考线
xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min(Y_train) max(Y_train)]);
ylim([min(Y_train) max(Y_train)]);
title('训练集预测值 vs. 训练集真实值');
grid on;

%% —— 2. 测试集散点图 ——
figure;
scatter(Y_test, Y_psed, sz, c, 'filled');
hold on;
plot(xlim, xlim, '--k');   % y = x 参考线
xlabel('测试集真实值');
ylabel('测试集预测值');
xlim([min(Y_test) max(Y_test)]);
ylim([min(Y_psed) max(Y_psed)]);
title('测试集预测值 vs. 测试集真实值');
grid on;

RMSE_train = sqrt(mean((Y_train - Y_train).^2));
RMSE_test  = sqrt(mean((Y_test  - Y_psed).^2));

M = length(Y_train);
N = length(Y_test);
%% ================== 训练集样本序列对比图 ==================
figure;
plot(1:M, Y_train, 'r-*', ...
     1:M, Y_train, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值', 'Location', 'best');
xlabel('预测样本');
ylabel('预测结果');
title(['训练集预测结果对比；RMSE = ' num2str(RMSE_train, '%.3f')]);
xlim([1 M]);
grid on;


%% ================== 测试集样本序列对比图 ==================
figure;
plot(1:N, Y_test, 'r-*', ...
     1:N, Y_psed, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值', 'Location', 'best');
xlabel('预测样本');
ylabel('预测结果');
title(['测试集预测结果对比；RMSE = ' num2str(RMSE_test, '%.3f')]);
xlim([1 N]);
grid on;
