%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行



%%  导入数据
% 行：2004–2023（按时间顺序）
% 列：1–5 输入变量，6 输出变量
res = xlsread('数据集_苏北.xlsx');

%% ================= 3. 样本划分（按时间，不打乱） =================
% 训练集：2005–2020（16 年）
% 验证集：2021–2023（3 年）
temp = randperm(19);

P_train = res(1:12, 2:4)';   % 2005–2020
T_train = res(1:12, 15)';
M = size(P_train, 2);

P_test  = res(13:15, 2:4)'; % 2021–2023
T_test  = res(13:15, 15)';
N = size(P_test, 2);
%%  数据归一化
%   对输入、输出都做归一化，并保存了 ps_input、ps_output 用于反归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  节点个数  BP网络结构
%   9（输入） → 7（隐藏） → 1（输出）
inputnum  = size(p_train, 1);  % 输入层节点数
hiddennum = 2;                 % 隐藏层节点数
outputnum = size(t_train,1);   % 输出层节点数

%%  建立网络
net = newff(p_train, t_train, hiddennum);

%%  设置训练参数
net.trainParam.epochs     = 1000;      % 训练次数
net.trainParam.goal       = 1e-6;      % 目标误差
net.trainParam.lr         = 0.01;      % 学习率
net.trainParam.showWindow = 0;         % 关闭窗口

%%  参数初始化
c1      = 4.494;       % 学习因子
c2      = 4.494;       % 学习因子
maxgen  =   50;        % 种群更新次数  
sizepop =    5;        % 种群规模
Vmax    =  1.0;        % 最大速度
Vmin    = -1.0;        % 最小速度
popmax  =  1.0;        % 最大边界
popmin  = -1.0;        % 最小边界

%%  节点总数  粒子编码内容（每一个粒子 = 一个完整 BP 网络初始参数方案）
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

for i = 1 : sizepop
    pop(i, :) = rands(1, numsum);  % 初始化种群
    V(i, :) = rands(1, numsum);    % 初始化速度
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train); 
end

%%  个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % 全局最佳
gbest = pop;                   % 个体最佳
fitnessgbest = fitness;        % 个体最佳适应度值
BestFit = fitnesszbest;        % 全局最佳适应度值

%%  迭代寻优
for i = 1 : maxgen
    for j = 1 : sizepop
        
        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % 自适应变异
        pos = unidrnd(numsum);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);

    end
    
    for j = 1 : sizepop

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

%%  提取最优初始权值和阈值
w1 = zbest(1 : inputnum * hiddennum);
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum);
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

%%  最优值赋值
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%%  打开训练窗口 
net.trainParam.showWindow = 1;        % 打开窗口

%%  网络训练
net = train(net, p_train, t_train);

%%  仿真预测
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );


%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%% =========================================================
%%  Bias correction（基于训练集均值偏差）
%% =========================================================
bias = mean(T_train(:) - T_sim1(:));   % 训练期系统性偏差
T_sim2_bc = T_sim2 + bias;             % 偏差修正后的测试集预测


%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2, 2)' ./ M);
error2 = sqrt(sum((T_sim2 - T_test) .^2, 2)' ./ N);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  误差曲线迭代图
figure;
plot(1 : length(BestFit), BestFit, 'LineWidth', 1.5);
xlabel('粒子群迭代次数');
ylabel('适应度值');
xlim([1, length(BestFit)])
string = {'模型迭代误差变化'};
title(string)
grid on

%%  相关指标计算
% ================= MAE =================
mae1 = sum(abs(T_sim1 - T_train), 2)' ./ M;
mae2 = sum(abs(T_sim2 - T_test ), 2)' ./ N;

disp(['训练集数据的 MAE 为：', num2str(mae1)])
disp(['测试集数据的 MAE 为：', num2str(mae2)])

% ================= MSE =================
mse1 = sum((T_sim1 - T_train).^2, 2)' ./ M;
mse2 = sum((T_sim2 - T_test ).^2, 2)' ./ N;

disp(['训练集数据的 MSE 为：', num2str(mse1)])
disp(['测试集数据的 MSE 为：', num2str(mse2)])

% ================= MAPE =================
% 注意：若 T_train 或 T_test 中存在 0，需要提前处理
mape1 = sum(abs((T_sim1 - T_train) ./ T_train), 2)' ./ M * 100;
mape2 = sum(abs((T_sim2 - T_test ) ./ T_test ), 2)' ./ N * 100;

disp(['训练集数据的 MAPE 为：', num2str(mape1), ' %'])
disp(['测试集数据的 MAPE 为：', num2str(mape2), ' %'])

% ================= NSE（Nash-Sutcliffe Efficiency）=================
nse1 = 1 - sum((T_sim1 - T_train).^2) / sum((T_train - mean(T_train)).^2);
nse2 = 1 - sum((T_sim2 - T_test ).^2) / sum((T_test  - mean(T_test )).^2);

disp(['训练集数据的 NSE 为：', num2str(nse1)])
disp(['测试集数据的 NSE 为：', num2str(nse2)])

% %%  绘制散点图
% sz = 25;
% c = 'b';
% 
% figure
% scatter(T_train, T_sim1, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('训练集真实值');
% ylabel('训练集预测值');
% xlim([min(T_train) max(T_train)])
% ylim([min(T_sim1) max(T_sim1)])
% title('训练集预测值 vs. 训练集真实值')
% 
% figure
% scatter(T_test, T_sim2, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('测试集真实值');
% ylabel('测试集预测值');
% xlim([min(T_test) max(T_test)])
% ylim([min(T_sim2) max(T_sim2)])
% title('测试集预测值 vs. 测试集真实值')

%% =========================================================
%%  Bootstrap 不确定性分析（论文一致版）
%%  原理：Residual bootstrap + 分位数区间
%%  用途：构造预测区间 + 单点（2021）落区间验证
%% =========================================================

%% ---------- 0. 参数 ----------
B = 1000;          % bootstrap 次数
alpha = 0.10;      % 90% 预测区间
q_low  = alpha/2;  % 0.05
q_high = 1-alpha/2;% 0.95

%% ---------- 1. 训练期残差 ----------
residuals = T_train(:) - T_sim1(:);
n_res = length(residuals);

%% ---------- 2. 对“未来预测期”构造预测区间 ----------
% 这里的 future 指：2021–2023（或你将来任意预测期）
N_future = length(T_sim2_bc);
T_boot_future = zeros(B, N_future);

for b = 1:B
    res_b = residuals(randi(n_res, N_future, 1));
    T_boot_future(b,:) = T_sim2_bc(:)' + res_b';
end

T_lower = quantile(T_boot_future, q_low);
T_upper = quantile(T_boot_future, q_high);

%% ---------- 3. 单点验证（论文做法） ----------
% 只验证“2021 年是否落入预测区间”
y_2021 = T_test(1);   % 2021 年真实值

is_in_interval_2021 = ...
    (y_2021 >= T_lower(1)) && (y_2021 <= T_upper(1));

disp('==============================================')
disp('Bootstrap 预测区间验证（论文一致逻辑）')
disp(['2021 年是否落入 90% 预测区间： ', num2str(is_in_interval_2021)])

%% ---------- 4. 区间宽度（仅作描述性分析） ----------
interval_width = mean(T_upper - T_lower);
disp(['平均预测区间宽度（90%）：', num2str(interval_width)])

%% ---------- 5. 可视化 ----------
figure
x = 1:N_future;

fill([x fliplr(x)], ...
     [T_lower fliplr(T_upper)], ...
     [0.85 0.85 0.85], 'EdgeColor','none'); hold on

plot(x, T_sim2_bc, 'b-o','LineWidth',1.5)
plot(x, T_test, 'r-*','LineWidth',1.5)

legend('90% 预测区间','预测值','真实值','Location','best')
xlabel('年份（预测期）')
ylabel('用水量')
title('Bootstrap 预测区间（论文一致方法）')
grid on
