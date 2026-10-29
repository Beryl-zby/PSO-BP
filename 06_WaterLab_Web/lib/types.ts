export type Row={year:number;x:number[];y:number};
export type Dataset={name:string;unit:string;featureNames:string[];rows:Row[];future?:{year:number;x:number[]}[];source?:string};
export type ChartKind='trend'|'backtest'|'sensitivity';
export type Forecast={
 rows:Row[];backtest:{year:number;actual:number;fixed:number;rolling:number;ols:number;persistence:number;error:number}[];
 forecasts:{year:number;x:number[];base:number;saving:number;growth:number}[];
 sensitivity:{change:number;value:number}[][];
 metrics:Record<string,{mape:number;mae:number;rmse:number;mse:number}>;
 config:{initialization?:'bp'|'pso';members:number;population:number;generations:number;maxIterations:number;lambda:number;holdout:number;seed:number;architecture:string};
 selections:{trainingEnd:number;lambda:number;mape:number;scores:{lambda:number;mape:number}[]}[];
 diagnostics:{negativeCounts:number[];outside:number;nonpositive:number;points:number};
 driverSource:string;trace:{label:string;seconds:number}[];seconds:number;createdAt:string;
 provenance:{modelVersion?:string;method:string;source:string;prng:string;validation:string;scenarios:string};
};
