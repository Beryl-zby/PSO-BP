function D=water_suite_data(inputFile)
base=fileparts(fileparts(mfilename('fullpath')));
if nargin<1||isempty(inputFile),inputFile=fullfile(base,'data','需水预测模型对照.xlsx');end
h=readmatrix(inputFile,'Sheet','历史数据','UseExcel',false);h=h(isfinite(h(:,1)),:);
assert(isequal(h(:,1),(2010:2023)')&&all(isfinite(h(:,1:14)),'all'));
D.file=inputFile;D.years=h(:,1);D.Y=h(:,14);
driver=[h(:,8),h(:,10),h(:,2).*(1+h(:,3)/100)];eff=h(:,[9,13]);
D.X=features(driver,eff);D.feature_names={'农业驱动','工业驱动','人口城镇化驱动'};
settings=readcell(inputFile,'Sheet','模型设置','UseExcel',false);D.seed=20261024;drop=.1; ps=struct;
for k=2:size(settings,1)
    if isequal(settings{k,1},'seed'),D.seed=settings{k,2};end
    if isequal(settings{k,1},'industrial_drop'),drop=settings{k,2};end
    if ischar(settings{k,1})||isstring(settings{k,1}),ps.(char(settings{k,1}))=settings{k,2};end
end
r=readcell(inputFile,'Sheet','未来指标','UseExcel',false);r=r(2:end,:);
b=[ones(14,1),D.years-2010]\log(eff(:,2));decay=-b(2);
D.future_years=(2024:2030)';D.scenarios={'SSP1','SSP2','SSP3','SSP4','SSP5'};
D.paths={'效率保持','历史趋势','规划约束'};D.future_X=cell(5,3);
D.frozen_efficiency=mean(eff(end-2:end,:),1);D.frozen_base_X=features(driver(end,:),D.frozen_efficiency);D.policy_settings=ps;
for s=1:5
    a=cell2mat(r(strcmp(r(:,1),D.scenarios{s}),2:12));assert(isequal(a(:,1),D.future_years));
    df=[a(:,8),a(:,10),a(:,2).*(1+a(:,3)/100)];
    frozen=repmat(D.frozen_efficiency,7,1);
    trend=[a(:,9),eff(end,2)*exp(-decay*a(:,11).*(D.future_years-2023))];policy=trend;
    if isfield(ps,'policy_mode')&&strcmp(ps.policy_mode,'jiangsu-2012')
        u=(D.future_years-2023)/(ps.policy_target_year-2023);
        eta=ps.irrigation_base+(ps.irrigation_target-ps.irrigation_base)*u;
        agcap=eff(end,1)*ps.irrigation_base./eta;
        icap=eff(end,2)*((ps.industrial_target_2000/ps.industrial_price_factor)/eff(end,2)).^u;
        policy=[min(trend(:,1),agcap),min(trend(:,2),icap)];
    else
        cap=trend(2,2)*(1-drop).^((D.future_years-2025)/5);ix=D.future_years>2025;
        policy(ix,2)=min(trend(ix,2),cap(ix));
    end
    paths={frozen,trend,policy};
    for j=1:3,D.future_X{s,j}=features(df,paths{j});end
end
fid=fopen(inputFile,'r');guard=onCleanup(@()fclose(fid));bytes=fread(fid,Inf,'*uint8');clear guard
digest=java.security.MessageDigest.getInstance('SHA-256');digest.update(bytes);
D.source_sha256=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
end

function X=features(D,E)
X=[D(:,1).*E(:,1)*15/100000,D(:,2).*E(:,2)/10000,D(:,3)];
end
