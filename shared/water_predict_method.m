function p=water_predict_method(modelFunction,model,X,baseX,baseY)
% New BP models carry their training-only correction; archived models use the old anchor.
p=feval(modelFunction,'predict',model,X);
if isfield(model,'prediction_calibration')
    p=p+model.prediction_calibration.offset;
else
    p=baseY+p-feval(modelFunction,'predict',model,baseX);
end
end
