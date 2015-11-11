function [ eye, imgon ] = getEyeDat( bhv, tnum )
imgoncode = 191;

eye = bhv.AnalogData{tnum}.EyeSignal;
imgon = bhv.CodeTimes{tnum}(bhv.CodeNumbers{tnum} == imgoncode);
end

