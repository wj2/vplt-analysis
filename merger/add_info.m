%add isolation, familiarity, channel and neuron length in the datafile
%Krithika Mohan
%March 22, 2017

% clc;clear all;close all;
% load('x_20170303.mat');load('data_20170303.mat');
ch_x=[];
ch_list=fieldnames(data.NEURO.Neuron);
%from google docs
for ch=1:length(ch_list)
    ch_x=[ch_x str2num(ch_list{ch}(5:6))];
end
for i=1:length(x)
    x(i).isolation_score=[4 3 3.5 3 3.5 3.5 2.5 3.5 3 3 3 4 3 3 2.5 3];
    x(i).channel=ch_x;
    x(i).nviews_start=32;
    x(i).ndays_start=1;
    x(i).ndepth=depth';
end
save('x_20170628','x');
save('data_20170628','data');