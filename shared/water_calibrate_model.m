function model=water_calibrate_model(model,modelFunction,X,Y,mode)
% Estimate an additive correction using only the supplied training prefix.
switch mode
    case 'none'
        offset=0;window=0;
    case 'last'
        window=1;offset=Y(end)-feval(modelFunction,'predict',model,X(end,:));
    case 'median3'
        window=min(3,numel(Y));ix=size(X,1)-window+1:size(X,1);
        fitted=feval(modelFunction,'predict',model,X(ix,:));
        observed=Y(ix);offset=median(observed(:)-fitted(:));
    otherwise
        error('Unknown calibration: %s',mode);
end
model.prediction_calibration=struct('mode',mode,'offset',offset,...
    'window',window,'training_rows',size(X,1));
end
