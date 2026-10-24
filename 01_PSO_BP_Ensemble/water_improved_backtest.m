function [R,models] = water_improved_backtest(X,Y,years,cfg)
% Expanding-history comparison. Every hyperparameter fit sees its prefix only.
% The 2020--2023 results are development backtests, not a new holdout.
Y=Y(:);years=years(:);
assert(size(X,1)==numel(Y)&&size(X,2)==3&&isequal(years,(2010:2023)'));
assert(all(isfinite(X),'all')&&all(isfinite(Y))&&all(Y>0));
methods={'小样本正则PSO-BP','岭回归（原网格）','岭回归（加密网格）','普通线性回归','最近实际值'};
grids={cfg.nn_penalties,10.^(-5:2),logspace(-5,2,29),0,0};
M=numel(methods);v=find(years>2019);n0=v(1)-1;
fixed=zeros(numel(v),M);rolling=fixed;selected=zeros(numel(v)+1,M);
records=cell(0,9);models=cell(1,M);training=cell(0,6);
cache=containers.Map('KeyType','char','ValueType','any');
for m=1:M
    for fold=1:numel(v)+1
        if fold<=numel(v),n=v(fold)-1;else,n=numel(Y);end
        grid=grids{m};scores=zeros(numel(grid),1);
        if m~=5
            for g=1:numel(grid)
                foldErrors=zeros(3,1);
                for j=1:3
                    at=n-3+j;f=getFit(at-1,m,grid(g));
                    p=anchor(f,m,X(at,:),X(at-1,:),Y(at-1));
                    foldErrors(j)=100*abs(p-Y(at))/Y(at);
                    records(end+1,:)={methods{m},years(n),grid(g),years(at),years(at-1),at-1,p,Y(at),foldErrors(j)}; %#ok<AGROW>
                end
                scores(g)=mean(foldErrors);
            end
        end
        [~,best]=min(scores);lambda=grid(best);selected(fold,m)=lambda;
        f=getFit(n,m,lambda);
        if fold<=numel(v)
            at=v(fold);rolling(fold,m)=anchor(f,m,X(at,:),X(n,:),Y(n));
            if fold==1,fixed(:,m)=anchor(f,m,X(v,:),X(n0,:),Y(n0));end
        else
            models{m}=f;
        end
        if m==1
            training(end+1,:)={years(n),lambda,f.training_mape,f.training_rmse,...
                max([f.optimization.gradient_inf]),nnz(strcmp({f.optimization.stop},'max_iterations'))}; %#ok<AGROW>
        end
    end
end
fm=zeros(M,4);rm=fm;
for m=1:M,fm(m,:)=metrics(Y(v),fixed(:,m));rm(m,:)=metrics(Y(v),rolling(:,m));end
R=struct('methods',{methods},'historical_years',years,'historical_actual',Y,...
    'features',X,'validation_years',years(v),'actual',Y(v),...
    'fixed_prediction',fixed,'rolling_prediction',rolling,...
    'fixed_metrics',fm,'rolling_metrics',rm,'score',mean([fm(:,1),rm(:,1)],2),...
    'lambda_by_training_end',selected,'training_end_years',[years(v)-1;years(end)],...
    'tuning_rows',{records},'training_rows',{training},'settings',cfg,...
    'validation_scope','开发回测；2020—2023已参与此前模型探索和本轮方案开发，不是新增独立测试集',...
    'selection_rule','每个训练前缀的末3年逐年内验证选择参数；两种外层MAPE均值仅作描述性比较',...
    'development_disclosure','结构、较强正则网格与数值求解器在已知数据上开发；代码层面各次拟合/选参只读当前训练前缀，不能据此称完全无开发集选择偏差');
R.improves_original_both=all([fm(1,1)<fm(2,1),rm(1,1)<rm(2,1)]);
R.improves_dense_combined=R.score(1)<R.score(3);
R.improves_all_linear_both=all(fm(1,1)<fm(2:4,1))&&all(rm(1,1)<rm(2:4,1));
R.improves_all_linear_metrics=all(R.fixed_metrics(1,:)<R.fixed_metrics(2:4,:),'all')&&all(R.rolling_metrics(1,:)<R.rolling_metrics(2:4,:),'all');
R.relative_mape_reduction_original=100*(1-[fm(1,1)/fm(2,1),rm(1,1)/rm(2,1)]);
R.relative_score_reduction_dense=100*(1-R.score(1)/R.score(3));

    function f=getFit(n,m,lambda)
        key=sprintf('%d/%d/%.17g',n,m,lambda);
        if isKey(cache,key),f=cache(key);return,end
        if m==1
            settings=cfg;settings.penalty=lambda;
            f=model_SmallSample_PSO_BP('train',X(1:n,:),Y(1:n),settings,cfg.seed);
        elseif m==5
            f=struct('kind','persistence','train_rows',n);
        else
            f=model_Ridge('train',X(1:n,:),Y(1:n),lambda);
            f.train_rows=n;
        end
        assert(f.train_rows==n);cache(key)=f;
    end
end

function p=anchor(f,m,X,baseX,baseY)
if m==1
    p=baseY+mean(model_SmallSample_PSO_BP('predict',f,X),2)-mean(model_SmallSample_PSO_BP('predict',f,baseX),2);
elseif m==5
    p=repmat(baseY,size(X,1),1);
else
    p=baseY+model_Ridge('predict',f,X)-model_Ridge('predict',f,baseX);
end
end

function m=metrics(y,p)
e=p(:)-y(:);m=[mean(abs(e)./abs(y(:)))*100,mean(abs(e)),sqrt(mean(e.^2)),mean(e.^2)];
end
