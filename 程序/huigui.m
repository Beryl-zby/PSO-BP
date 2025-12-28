%% ================== 多元回归（修正版，可运行） ==================
clear; clc; close all;

%% 读取数据
res = xlsread('数据集_苏北.xlsx');

X = res(:,1:5);   % 解释变量
Y = res(:,6);     % 年需水量

%% ---------- Step 1：删除 NaN 行 ----------
idx_valid = all(~isnan([X Y]), 2);
X = X(idx_valid,:);
Y = Y(idx_valid);

%% ---------- Step 2：删除零方差列 ----------
varX = var(X);
idx_keep = varX > 1e-6;   % 阈值
X = X(:, idx_keep);

fprintf('保留的解释变量个数 = %d\n', size(X,2));

%% ---------- Step 3：稳健回归 ----------
mdl = fitlm(X, Y, 'RobustOpts','on');

disp(mdl);

%% ---------- Step 4：指标 ----------
R2_adj = mdl.Rsquared.Adjusted;
fprintf('Adjusted R2 = %.3f\n', R2_adj);

%% ---------- Step 5：残差诊断 ----------
figure;
plotResiduals(mdl,'fitted');
title('Residuals vs Fitted');

figure;
plotDiagnostics(mdl,'cookd');
title('Cook Distance');
