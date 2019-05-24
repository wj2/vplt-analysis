% addpath('bhv-ephys');
dataprefix = '/Users/wjj/Dropbox/research/uc/freedman/analysis/pref_looking/data-rufus/';
nexfiles = {...
    ...% strcat(dataprefix, '10162017/rufus_10162017-1b-sort3.nex'), ...
    ...% strcat(dataprefix, '10162017/rufus_10162017-2.nex'), ...
    ... % badname strcat(dataprefix, '09272017/rufus_09272017-1.nex'), ...
    ...% strcat(dataprefix, '10062017/rufus_10062017-1.nex'), ...
    ...% strcat(dataprefix, '10242017/rufus_10242017-2.nex'), ...
    ...% strcat(dataprefix, '10252017/rufus_10252017-1.nex'), ...
    ...% strcat(dataprefix, '10262017/rufus_10262017-1.nex'), ...
    ...% strcat(dataprefix, '10262017/rufus_10262017-2.nex'), ...
    ...% strcat(dataprefix, '10272017/rufus_10272017-1.nex'), ...
    ...% strcat(dataprefix, '10302017/rufus_10302017-1.nex'), ...
    ...% strcat(dataprefix, '10312017/rufus_10312017-1.nex'), ...
    ...% strcat(dataprefix, '11012017/rufus_11012017-2.nex'), ...
    ...% strcat(dataprefix, '11012017/rufus_11012017-3.nex'), ...
    ...% strcat(dataprefix, '11022017/rufus_11022017-1.nex'), ...
    ...% strcat(dataprefix, '11032017/rufus_11032017-1.nex'), ...
    ...% strcat(dataprefix, '11092017/rufus_11092017-1.nex'), ...
    ...% offset problem strcat(dataprefix, '11132017/rufus_11132017-1a.nex'), ...
    strcat(dataprefix, '11142017/rufus_11142017-1.nex'), ...
    strcat(dataprefix, '11282017/rufus_11282017-1.nex'), ...
    strcat(dataprefix, '11292017/rufus_11292017-4.nex'), ...
    strcat(dataprefix, '11302017/rufus_11302017-1.nex'), ...
    strcat(dataprefix, '12012017/rufus_12012017-1.nex'), ...
    strcat(dataprefix, '12142017/rufus_12142017-2.nex'), ...
    strcat(dataprefix, '12182017/rufus_12182017-1.nex'), ...
    strcat(dataprefix, '12202017/rufus_12202017-1-03.nex'), ...
    strcat(dataprefix, '12212017/rufus_12212017-2.nex'), ...
    strcat(dataprefix, '12212017/rufus_12212017-3-01-fin.nex'), ...
    };

nex_offsets = [... % 0 0 ...
    ... %0
    ... % 0 0 0 0 1 0 0 0 0 0 0 0 0 ...
    ...% offset problem 1 
    1 0 0 0 0 0 0 0 0 0]; % changed first to 1

bhvfiles = {...
    ...%strcat(dataprefix, '10162017/sdmst-plt-ms_rw-rufus-TASK-10-16-2017(02).bhv'), ...
    ...%strcat(dataprefix, '10162017/sdmst-plt-ms_rw-rufus-TASK-10-16-2017(03).bhv'), ...
    ... % badname strcat(dataprefix, '09272017/sdmst-plt-ms_rw-rufus-TASKREC-09-27-2017(02).bhv'), ...
    ...%strcat(dataprefix, '10062017/sdmst-plt-ms_rw-rufus-TASK-10-06-2017(02).bhv'), ...
    ...%strcat(dataprefix, '10242017/sdmst-plt-ms_rw-rufus-TASK-10-24-2017(03).bhv'), ...
    ...%strcat(dataprefix, '10252017/sdmst-plt-ms_rw-rufus-TASK-10-25-2017(01).bhv'), ...
    ...%strcat(dataprefix, '10262017/sdmst-plt-ms_rw-rufus-TASK-10-26-2017(01).bhv'), ...
    ...%strcat(dataprefix, '10262017/sdmst-plt-ms_rw-rufus-TASK2-10-26-2017(02).bhv'), ...
    ...%strcat(dataprefix, '10272017/sdmst-plt-ms_rw-rufus-TASK-10-27-2017(01).bhv'), ...
    ...%strcat(dataprefix, '10302017/sdmst-plt-ms_rw-rufus-10-30-2017(01).bhv'), ...
    ...%strcat(dataprefix, '10312017/sdmst-plt-ms_rw-rufus-TASK-10-31-2017(01).bhv'), ...
    ...%strcat(dataprefix, '11012017/sdmst-plt-ms_rw-rufus-TASK-11-01-2017(02).bhv'), ...
    ...%strcat(dataprefix, '11012017/sdmst-plt-ms_rw-rufus-TASK-11-01-2017(04).bhv'), ...
    ...%strcat(dataprefix, '11022017/sdmst-plt-ms_rw-rufus-TASK-11-02-2017(01).bhv'), ...
    ...%strcat(dataprefix, '11032017/sdmst-plt-ms_rw-rufus-TASK-11-03-2017(01).bhv'), ...
    ...%strcat(dataprefix, '11092017/sdmst-plt-ms_rw-rufus-TASK-11-09-2017(01).bhv'), ...
    ...% offset problem strcat(dataprefix, '11132017/sdmst-plt-ms_rw-rufus-TASK-11-13-2017(03).bhv'), ...
    strcat(dataprefix, '11142017/sdmst-plt-ms_rw-rufus-11-14-2017(02).bhv'), ...
    strcat(dataprefix, '11282017/sdmst-plt-ms_rw-rufus-TASK-11-28-2017(02).bhv'), ...
    strcat(dataprefix, '11292017/sdmst-plt-ms_rw-rufus-TASK-11-29-2017(03).bhv'), ...
    strcat(dataprefix, '11302017/sdmst-plt-ms_rw-rufus-TASK-11-30-2017(02).bhv'), ...
    strcat(dataprefix, '12012017/sdmst-plt-ms_rw-rufus-TASK-12-01-2017(02).bhv'), ...
    strcat(dataprefix, '12142017/sdmst-plt-ms_rw-rufus-TASK-12-14-2017(02).bhv'), ...
    strcat(dataprefix, '12182017/sdmst-plt-ms_rw-rufus-TASK-12-18-2017(01).bhv'), ...
    strcat(dataprefix, '12202017/sdmst-plt-ms_rw-rufus-TASK-12-20-2017(01).bhv'), ...
    strcat(dataprefix, '12212017/sdmst-plt-ms_rw-rufus-TASK-12-21-2017(02).bhv'), ...
    strcat(dataprefix, '12212017/sdmst-plt-ms_rw-rufus-TASK-12-21-2017(03).bhv'), ...
    };

savetemp = strcat(dataprefix,'%s_merged_st.mat');

j = 0;
for i = 1:length(nexfiles)
    disp(i);
    nf = nexfiles{i};
    disp(nf);
    bhv = bhvfiles{i};
    no = nex_offsets(i);
    [pname, fname, ext] = fileparts(bhv);
    [nexp, nexf, nexext] = fileparts(nf);
    imglog = fullfile(pname, strcat(fname, '_imglog.txt'));
    savename = sprintf(savetemp, nexf);
    opendatafile_sdmst_plt(nf, bhv, 0, 0, savename, imglog, no);
end
