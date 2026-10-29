function [spec,cfg]=water_suite_setup(family)
% Candidate sets are explicit and fixed before the three-way comparison.
base=fileparts(fileparts(mfilename('fullpath')));
cfg=struct('seed',20261024,'members',12,'pso_population',24,...
    'pso_generations',40,'max_iterations',300,'bayes_epochs',300,'industrial_drop',.1);
switch lower(family)
    case 'bp'
        cfg.pso_population=0;cfg.pso_generations=0;
        spec=struct('id','bp','name','正则BP集成','folder','07_BP_Ensemble',...
            'model','model_BP_Ensemble','description','3-1-1；权重偏置共同L2；随机小初值+解析Jacobian阻尼高斯牛顿；12成员等权集成；无PSO');
        candidates=cell(1,4);values=[.01,.03,.1,1];
        for k=1:4,candidates{k}=struct('label',sprintf('lambda=%.2g',values(k)),'penalty',values(k));end
    case 'pso'
        spec=struct('id','pso','name','正则PSO-BP集成','folder','01_PSO_BP_Ensemble',...
            'model','model_PSO_BP_Ensemble','description','3-1-1；权重与偏置共同L2正则；PSO+解析Jacobian训练；12成员等权集成');
        candidates=cell(1,4);values=[.01,.03,.1,1];
        for k=1:4,candidates{k}=struct('label',sprintf('lambda=%.2g',values(k)),'penalty',values(k));end
    case 'bayes'
        spec=struct('id','bayes','name','贝叶斯正则神经网络','folder','02_Bayesian_BP',...
            'model','model_Bayesian_BP','description','trainbr自适应平衡误差与权重/偏置惩罚；候选1/2/4隐藏节点；12成员等权集成；非完整贝叶斯神经网络');
        candidates=cell(1,3);values=[1,2,4];
        for k=1:3,candidates{k}=struct('label',sprintf('hidden=%d',values(k)),'hidden',values(k));end
    case 'gpr'
        spec=struct('id','gpr','name','高斯过程回归','folder','03_Gaussian_Process',...
            'model','model_Gaussian_Process','description','精确GPR；训练期边际似然拟合；平方指数/Matérn3/2核及常数/线性均值内验证选择');
        candidates=cell(1,4);k=0;
        for kernel={'squaredexponential','matern32'}
            for basis={'constant','linear'}
                k=k+1;candidates{k}=struct('label',[kernel{1},'/',basis{1}],'kernel',kernel{1},'basis',basis{1});
            end
        end
    otherwise
        error('Unknown family %s',family);
end
spec.candidates=candidates;spec.path=fullfile(base,spec.folder);addpath(spec.path);
end
