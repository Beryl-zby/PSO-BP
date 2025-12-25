clear; clc; close all

%% 用 readmatrix 读纯数值
opts = detectImportOptions('PCA数据.xlsx');
opts.DataRange = 'A2';   % 从第2行开始
res = readmatrix('PCA数据.xlsx', opts);

% 检查是否读成功
disp('是否存在 NaN（1 表示有）:')
disp(any(isnan(res),'all'))

X = res(:,1:7);   % 7 个输入变量

% 删除包含 NaN 的行（非常重要）
validRow = all(~isnan(X), 2);
X = X(validRow, :);


varNames = {'人口','GDP','播种面积','灌溉强度',...
            '工业用水强度','滞后需水','需水差分'};

%% ===== 1. 检查方差 =====
sigma = std(X);

disp('各变量标准差 = ');
disp(table(varNames', sigma', 'VariableNames', {'变量','Std'}))

%% ===== 2. 剔除低方差变量 =====
threshold = 1e-6;   % 阈值（可以写进论文）
validIdx = sigma > threshold;

X_clean = X(:, validIdx);
varNames_clean = varNames(validIdx);

disp('保留下来的变量：')
disp(varNames_clean)

%% ===== 3. 标准化 =====
Xz = zscore(X_clean);

%% ===== 4. PCA =====
R = corrcoef(Xz);
[V,D] = eig(R);

[lambda, idx] = sort(diag(D), 'descend');
V = V(:, idx);

contribution = lambda / sum(lambda);
cum_contribution = cumsum(contribution);

disp('特征值 ='); disp(lambda')
disp('贡献率 ='); disp(contribution')
disp('累计贡献率 ='); disp(cum_contribution')

%% ===== 5. 主成分载荷 =====
Loadings = V(:,1:3);
disp('前三个主成分载荷：')
disp(array2table(Loadings,...
    'RowNames',varNames_clean,...
    'VariableNames',{'PC1','PC2','PC3'}))

%% ===== 6. 计算主成分得分（最关键一步）=====
F = Xz * V(:,1:3);    % 20 × 3

disp('主成分得分（前 5 个样本）：')
disp(F(1:5,:))
