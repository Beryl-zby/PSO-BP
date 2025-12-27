function error = fun(pop, hiddennum, net, p_fit, t_fit, p_val, t_val)

%% 节点数
inputnum  = size(p_fit,1);
outputnum = size(t_fit,1);   % 一般为1

%% 解码权值与阈值
w1 = pop(1 : inputnum*hiddennum);
B1 = pop(inputnum*hiddennum + 1 : inputnum*hiddennum + hiddennum);

w2 = pop(inputnum*hiddennum + hiddennum + 1 : ...
         inputnum*hiddennum + hiddennum + hiddennum*outputnum);

B2 = pop(inputnum*hiddennum + hiddennum + hiddennum*outputnum + 1 : ...
         inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum);

%% 赋值
net.Iw{1,1} = reshape(w1, hiddennum, inputnum);
net.Lw{2,1} = reshape(w2, outputnum, hiddennum);
net.b{1}    = B1(:);
net.b{2}    = B2(:);

%% PSO评估：少训（加速）
net.trainParam.showWindow = 0;
net.trainParam.epochs     = 50;
net.trainParam.goal       = 1e-4;

%% 用训练子集训练
net = train(net, p_fit, t_fit);

%% 在验证子集上评估
t_sim_val = sim(net, p_val);

%% 验证集RMSE作为fitness
error = sqrt(mean((t_sim_val(:) - t_val(:)).^2));

if isnan(error) || isinf(error)
    error = 1e6;
end

end
