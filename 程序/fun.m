function error = fun(pop, hiddennum, net, p_train, t_train, p_val, t_val)
% =========================================================
% PSO-BP 适应度函数（基于验证集误差）
% 训练：p_train, t_train
% 评估：p_val, t_val
% =========================================================

%% ========== 节点个数 ==========
inputnum  = size(p_train, 1);   % 输入层节点数
outputnum = size(t_train, 1);   % 输出层节点数

%% ========== 提取权值和阈值 ==========
w1 = pop(1 : inputnum * hiddennum);
B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);

w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
          inputnum * hiddennum + hiddennum + hiddennum * outputnum);

B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
          inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

%% ========== 网络赋值 ==========
net.Iw{1,1} = reshape(w1, hiddennum, inputnum);
net.Lw{2,1} = reshape(w2, outputnum, hiddennum);
net.b{1}    = reshape(B1, hiddennum, 1);
net.b{2}    = B2';

%% ========== PSO 阶段：限制训练强度（防止过拟合） ==========
net.trainParam.showWindow = 0;
net.trainParam.epochs     = 50;      % ★ PSO评估阶段不要训太久
net.trainParam.goal       = 1e-4;

%% ========== 网络训练（仅训练集） ==========
net = train(net, p_train, t_train);

%% ========== 验证集仿真 ==========
t_sim_val = sim(net, p_val);

%% ========== 适应度值（验证集 RMSE） ==========
% 用 RMSE，比你原来那个嵌套 sqrt 更清晰、数值更稳定
error = sqrt(mean((t_sim_val - t_val).^2));

% 若数值异常，直接惩罚
if isnan(error) || isinf(error)
    error = 1e6;
end

end
