function B=prepare_agent_evidence(inputFile)
% Export training-prefix evidence separately from the evaluation targets.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);setup_water_paths;
if nargin<1,inputFile=[];end
D=water_suite_data(inputFile);loaded=load(fullfile(root,'results','three_methods','comparison.mat'),'S');S=loaded.S;
assert(strcmp(S.input_sha256,D.source_sha256),'Run main_ThreeMethodComparison for this workbook first.');
ids={'pso','gpr','ols'};names={'正则PSO-BP集成','高斯过程回归','普通线性回归'};indices=[1,3,6];
B=struct('schema_version',1,'input_sha256',D.source_sha256,'expert_ids',{ids},...
    'expert_names',{names},'feature_names',{D.feature_names},'folds',[]);
E=struct('validation_years',D.years(11:14),'actual',D.Y(11:14),...
    'fixed_prediction',S.fixed_prediction(:,indices),'rolling_prediction',S.rolling_prediction(:,indices),...
    'baseline_methods',{S.methods},'baseline_fixed_metrics',S.fixed_metrics,...
    'baseline_rolling_metrics',S.rolling_metrics,'future',[],'future_years',D.future_years,...
    'scenarios',{D.scenarios},'paths',{D.paths});
stages=cell(3,5);modelNames={'model_PSO_BP_Ensemble','model_Gaussian_Process','water_linear_suite_model'};
for m=1:2
    [spec,~]=water_suite_setup(ids{m});a=load(fullfile(spec.path,'results','result.mat'),'models');
    stages(m,:)=[{a.models.fixed},a.models.rolling(2:4),{a.models.final}];
end
for f=1:5
    n=9+f;packet=struct('training_end_year',D.years(n),'training_rows',n,'experts',[]);
    for m=1:3
        R=S.results{indices(m)};label=R.selected{f}.label;
        rows=cell2mat(R.tuning_rows(:,1))==D.years(n)&strcmp(R.tuning_rows(:,2),label);
        t=R.tuning_rows(rows,:);assert(size(t,1)==3);
        assert(all(cell2mat(t(:,3))<=D.years(n))&&all(cell2mat(t(:,4))<cell2mat(t(:,3))));
        if m==3
            stages{m,f}=water_linear_suite_model('train',D.X(1:n,:),D.Y(1:n),struct('penalty',0),struct());
        end
        model=stages{m,f};assert(model.training_rows==n);xx=D.X(1:n,:);
        p=feval(modelNames{m},'predict',model,xx);negative=zeros(1,3);
        for c=1:3
            down=xx;down(:,c)=.99*down(:,c);
            negative(c)=nnz(p-feval(modelNames{m},'predict',model,down)<-1e-7);
        end
        err=cell2mat(t(:,7));e=struct('id',ids{m},'configuration',label,...
            'cv_years',cell2mat(t(:,3))','cv_mape',mean(err),'cv_absolute_percent_errors',err',...
            'cv_worst_percent_error',max(err),'training_direction_negative_counts',negative,...
            'direction_test_points',n);
        packet.experts=[packet.experts,e];
    end
    B.folds=[B.folds,packet];
end
% Held-out predictions/targets and future scenario inputs NEVER enter an agent packet.
for s=1:5
    for j=1:3
        xx=D.future_X{s,j};p=zeros(7,3);downP=zeros(7,3,3);
        for m=1:3
            model=stages{m,5};anchor=D.Y(end)-feval(modelNames{m},'predict',model,D.X(end,:));
            p(:,m)=anchor+feval(modelNames{m},'predict',model,xx);
            for c=1:3
                down=xx;down(:,c)=.99*down(:,c);
                downP(:,c,m)=anchor+feval(modelNames{m},'predict',model,down);
            end
        end
        for k=1:7
            point=struct('year',D.future_years(k),'scenario',D.scenarios{s},'path',D.paths{j},...
                'expert_prediction',p(k,:),'down_expert_prediction',reshape(downP(k,:,:),3,3));
            E.future=[E.future,point];
        end
    end
end
out=fullfile(root,'05_Multi_Agent','results');if ~isfolder(out),mkdir(out);end
writeJson(fullfile(out,'training_evidence.json'),B);writeJson(fullfile(out,'evaluation_bank.json'),E);
fprintf('EXPORTED five prefix packets; evaluation bank kept separate.\n');
end

function writeJson(file,data)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);guard=onCleanup(@()fclose(fid));
fprintf(fid,'%s',jsonencode(data,'PrettyPrint',true));clear guard
end
