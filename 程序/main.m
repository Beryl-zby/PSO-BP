%% 清空环境
warning off
close all
clear
clc

N_run = 1;

%% 导入数据
res = xlsread('数据集_修改.xlsx');

%% 样本划分（按时间）
N_train = 16;
N_test  = 3;

% 输入：1-7 + YearIndex(第9列)；输出：第8列
P_train_raw = res(1:N_train, [1:7, 9])';
T_train_raw = res(1:N_train, 8)';

P_test_raw  = res(N_train+1:N_train+N_test, [1:7, 9])';
T_test      = res(N_train+1:N_train+N_test, 8)';

idx_train = res(1:N_train, 9);
idx_test  = res(N_train+1:N_train+N_test, 9);

M = size(P_train_raw,2);
N = size(P_test_raw,2);

%% 归一化（仅用训练集拟合归一化参数）
[p_train, ps_input] = mapminmax(P_train_raw, 0, 1);
p_test = mapminmax('apply', P_test_raw, ps_input);

[t_train, ps_output] = mapminmax(T_train_raw, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%% ===== 训练集内部切验证集（不泄露未来）=====
K_val = 3;                        % 用训练集最后3年做验证
idx_fit = 1:(M-K_val);
idx_val = (M-K_val+1):M;

p_fit = p_train(:, idx_fit);
t_fit = t_train(:, idx_fit);

p_val = p_train(:, idx_val);
t_val = t_train(:, idx_val);

idx_val_year = idx_train(idx_val);   % 验证集对应YearIndex
% ============================================

%% 预测矩阵
Pred_all = zeros(N_run, N);

h = waitbar(0,'PSO-BP 预测中，请稍候...');

for run = 1:N_run

    %% BP网络结构（小样本更稳）
    inputnum  = size(p_train,1);
    hiddennum = 1;          % 可试 1/2，但 1 通常更稳
    outputnum = 1;

    net = newff(p_train, t_train, hiddennum, {'tansig','purelin'}, 'trainbr');
    net.divideFcn = 'dividetrain';
    net.trainParam.showWindow = 0;

    %% PSO 参数
    c1 = 2; c2 = 2;
    maxgen  = 50;
    sizepop = 10;

    Vmax = 1.0; Vmin = -1.0;
    popmax = 1.0; popmin = -1.0;

    numsum = inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum;

    pop = zeros(sizepop, numsum);
    V   = zeros(sizepop, numsum);
    fitness = zeros(sizepop,1);

    %% 初始化并算fitness（只看验证集）
    for i = 1:sizepop
        pop(i,:) = rands(1, numsum);
        V(i,:)   = rands(1, numsum);
        fitness(i) = fun(pop(i,:), hiddennum, net, p_fit, t_fit, p_val, t_val);
    end

    [fitnesszbest, bestindex] = min(fitness);
    zbest = pop(bestindex,:);
    gbest = pop;
    fitnessgbest = fitness;

    BestFit = fitnesszbest;

    %% PSO迭代
    for gen = 1:maxgen
        for j = 1:sizepop

            % 速度更新
            V(j,:) = V(j,:) ...
                + c1*rand*(gbest(j,:) - pop(j,:)) ...
                + c2*rand*(zbest - pop(j,:));

            V(j, V(j,:)>Vmax) = Vmax;
            V(j, V(j,:)<Vmin) = Vmin;

            % 位置更新
            pop(j,:) = pop(j,:) + 0.2*V(j,:);

            pop(j, pop(j,:)>popmax) = popmax;
            pop(j, pop(j,:)<popmin) = popmin;

            % 变异
            if rand > 0.85
                pos = unidrnd(numsum);
                pop(j,pos) = rands(1,1);
            end

            % 适应度（只看验证集）
            fitness(j) = fun(pop(j,:), hiddennum, net, p_fit, t_fit, p_val, t_val);
        end

        % 更新最优
        for j = 1:sizepop
            if fitness(j) < fitnessgbest(j)
                gbest(j,:) = pop(j,:);
                fitnessgbest(j) = fitness(j);
            end
            if fitness(j) < fitnesszbest
                zbest = pop(j,:);
                fitnesszbest = fitness(j);
            end
        end

        BestFit = [BestFit; fitnesszbest]; %#ok<AGROW>
    end

    %% 用PSO最优粒子赋初值
    w1 = zbest(1 : inputnum*hiddennum);
    B1 = zbest(inputnum*hiddennum + 1 : inputnum*hiddennum + hiddennum);
    w2 = zbest(inputnum*hiddennum + hiddennum + 1 : inputnum*hiddennum + hiddennum + hiddennum*outputnum);
    B2 = zbest(inputnum*hiddennum + hiddennum + hiddennum*outputnum + 1 : end);

    net.Iw{1,1} = reshape(w1, hiddennum, inputnum);
    net.Lw{2,1} = reshape(w2, outputnum, hiddennum);
    net.b{1}    = B1(:);
    net.b{2}    = B2(:);

    %% 最终训练：用全训练集（16年）多训
    net.trainParam.showWindow = 1;
    net.trainParam.epochs = 250;
    net.trainParam.goal   = 1e-6;

    net = train(net, p_train, t_train);

    %% BP预测（反归一化）
    t_sim_train = sim(net, p_train);
    t_sim_val   = sim(net, p_val);
    t_sim_test  = sim(net, p_test);

    T_sim_train = mapminmax('reverse', t_sim_train, ps_output);  % 训练(全16年)预测
    T_sim_val   = mapminmax('reverse', t_sim_val,   ps_output);  % 验证(末尾3年)预测
    T_sim_test  = mapminmax('reverse', t_sim_test,  ps_output);  % 测试(未来3年)预测

    y_train_true = T_train_raw(:);
    y_val_true   = y_train_true(idx_val);   % 验证真实值（反归一化）
    y_test_true  = T_test(:);

    %% ================= 校正器：交互项 + idx^2 + 用验证集挑lambda（不看测试集） =================
    lambdas = [0.01, 0.05, 0.1, 0.2, 0.5];

    best_val_mae = inf;
    best_lambda  = lambdas(1);
    best_T_test_corr = T_sim_test(:);

    % 校正器训练特征（用全训练集）
    X_cal = [ T_sim_train(:), idx_train(:), T_sim_train(:).*idx_train(:), idx_train(:).^2 ];
    mu = mean(X_cal,1);
    sg = std(X_cal,0,1);  sg(sg==0)=1;
    Xc = (X_cal - mu) ./ sg;
    Xc1 = [ones(M,1), Xc];

    % 验证/测试特征
    X_val = [ T_sim_val(:), idx_val_year(:), T_sim_val(:).*idx_val_year(:), idx_val_year(:).^2 ];
    Xv = (X_val - mu) ./ sg;
    Xv1 = [ones(size(Xv,1),1), Xv];

    X_te  = [ T_sim_test(:), idx_test(:), T_sim_test(:).*idx_test(:), idx_test(:).^2 ];
    Xt = (X_te - mu) ./ sg;
    Xt1 = [ones(size(Xt,1),1), Xt];

    for k = 1:numel(lambdas)
        lambda = lambdas(k);

        I = eye(size(Xc1,2)); I(1,1)=0;
        beta = (Xc1'*Xc1 + lambda*I) \ (Xc1'*y_train_true);

        % 先看验证集效果（不看测试集！）
        T_val2 = Xv1 * beta;

        % bias 用“验证集残差”的中位数（专门贴近尾部段）
        bias = median(y_val_true(:) - T_val2(:));
        T_val2 = T_val2 + bias;

        val_mae = mean(abs(T_val2(:) - y_val_true(:)));

        if val_mae < best_val_mae
            best_val_mae = val_mae;
            best_lambda  = lambda;

            % 用同样beta+bias校正测试集
            T_test2 = Xt1 * beta + bias;
            best_T_test_corr = T_test2(:);
        end
    end

    T_sim2 = best_T_test_corr(:);  % 最终测试预测（校正后）
    disp(['校正器选中的 lambda = ', num2str(best_lambda), '，验证MAE = ', num2str(best_val_mae)])
    % ============================================================================================

    %% 异常值过滤（可选）
    lower_bound = 200;
    upper_bound = 1000;

    if any(T_sim2 < lower_bound) || any(T_sim2 > upper_bound) || any(isnan(T_sim2)) || any(isinf(T_sim2))
        Pred_all(run,:) = NaN;
    else
        Pred_all(run,:) = T_sim2(:)';
    end

    waitbar(run/N_run, h, sprintf('PSO-BP 预测进度：%d / %d', run, N_run));
end

close(h);
disp('PSO-BP 预测完成');

%% 保存结果
Result.Pred_all = Pred_all;
Result.T_test   = T_test;
Result.N_run    = N_run;
save('PSO_BP_Result.mat','Result');

%% 评价指标（训练用BP原输出，测试用校正后输出）
% 训练集用校正前的BP预测（你也可以改成校正后训练，但一般报告用原模型）
T_sim1 = T_sim_train(:);
T_train_true = y_train_true(:);

error1 = sqrt(mean((T_sim1(:) - T_train_true(:)).^2));
error2 = sqrt(mean((T_sim2(:) - y_test_true(:)).^2));

R1 = 1 - norm(T_train_true(:) - T_sim1(:))^2 / norm(T_train_true(:) - mean(T_train_true(:)))^2;
R2 = 1 - norm(y_test_true(:)  - T_sim2(:))^2 / norm(y_test_true(:)  - mean(y_test_true(:)))^2;

mae1 = mean(abs(T_sim1(:) - T_train_true(:)));
mae2 = mean(abs(T_sim2(:) - y_test_true(:)));

mbe1 = mean(T_sim1(:) - T_train_true(:));
mbe2 = mean(T_sim2(:) - y_test_true(:));

disp(['训练集 R2 = ', num2str(R1)])
disp(['测试集 R2 = ', num2str(R2)])
disp(['训练集 MAE = ', num2str(mae1)])
disp(['测试集 MAE = ', num2str(mae2)])
disp(['训练集 MBE = ', num2str(mbe1)])
disp(['测试集 MBE = ', num2str(mbe2)])

%% 图
figure
plot(1:M, T_train_true, 'r-*', 1:M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值','预测值')
title(['训练集：RMSE=', num2str(error1)])
grid on

figure
plot(1:N, y_test_true, 'r-*', 1:N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值','预测值(校正后)')
title(['测试集：RMSE=', num2str(error2)])
grid on

figure
plot(BestFit, 'LineWidth', 1.5)
xlabel('PSO迭代次数')
ylabel('验证集RMSE(适应度)')
title('PSO迭代曲线（仅用训练内验证）')
grid on
