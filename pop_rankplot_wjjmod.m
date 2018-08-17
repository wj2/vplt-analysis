%Population analysis for rank preference plots for novel and familiar across neurons
%Krithika Mohan
%1 May 2015
clc;clear all;
filedir = './krithika_dim_data_resorted/';
startdir = cd;
cd(filedir);
datafiles=dir;
day=1;
normalize = false;
for filenum=4:length(datafiles)
    cd(datafiles(filenum).name)
    sprintf('Day %i %s',day,datafiles(filenum).name)
    
    %plot preference plots for N and F images
    load data; load x;
    cd ..
    
    for nunit = 1:length(x(1).spike_times)
        
        [pval] = anova_sel(data,x,nunit);
        vsa(day) = pval;
%         if pval > 0.05
%             break
%         else
            
            ntrials=length(x);
            fam=1:50; novel=76:125;ifam=51:75;early=1:10;

            frate_all=zeros(125,50);frate_all(:)=-1;
            for trial_no=1:ntrials
                if ismember(x(trial_no).task_cond_num,1:6)
                    if x(trial_no).TrialError==0 || x(trial_no).TrialError==6
                        ind_on = find(x(trial_no).code_numbers>55 & x(trial_no).code_numbers<181);
                        bin_size=1;
                        nc_ms=x(trial_no).code_times - x(trial_no).code_times(1);
                        for image_ind=1:length(ind_on)
                            img_no = x(trial_no).code_numbers(ind_on(image_ind))-55;
                            time_frame=[nc_ms(ind_on(image_ind))+80:bin_size:nc_ms(ind_on(image_ind)+1)+80];
                            spk_tm = x(trial_no).spike_times{1,nunit};
                            rep = data.BHV.UserVars(trial_no).repetition;
                            if isempty(spk_tm)
                                frate_all(img_no,rep)=NaN;
                            else
                                spks=histc(spk_tm,time_frame); 
                                mfrate = (sum(spks)*1000)/(bin_size*(time_frame(end)-time_frame(1)));
                                frate_all(img_no,rep)=mfrate;
                            end
                        end
                    end
                end   
            end

            for i=1:size(frate_all,1)
                y = frate_all(i,:);
                y(find(y==-1))=[];
                frate{i,:} = y;
            end

            for image_no=1:length(frate)
                rt_all(image_no)=nanmean(frate{image_no,1});
                rt_early(image_no) = nanmean(frate{image_no,1}(early));
                end_ind = length(frate{image_no});
                rt_late(image_no) = nanmean(frate{image_no,1}(20:25));
            end
            
            [sorted_rt_all_novel,index_rt_all_novel] = sort(rt_all(novel),'descend');
            [sorted_rt_early_novel,index_rt_early_novel] = sort(rt_early(novel),'descend');
            [sorted_rt_late_novel,index_rt_late_novel] = sort(rt_late(novel),'descend');
            
            [sorted_rt_all_fam,index_rt_all_fam] = sort(rt_all(fam),'descend');
            [sorted_rt_early_fam,index_rt_early_fam] = sort(rt_early(fam),'descend');
            [sorted_rt_late_fam,index_rt_late_fam] = sort(rt_late(fam),'descend');
            
            [sorted_rt_all_ifam,index_rt_all_ifam] = sort(rt_all(ifam),'descend');
            [sorted_rt_early_ifam,index_rt_early_ifam] = sort(rt_early(ifam),'descend');
            [sorted_rt_late_ifam,index_rt_late_ifam] = sort(rt_late(ifam),'descend');
            
            if normalize
                sorted_rt_all_novel = sorted_rt_all_novel/max(sorted_rt_all_novel);
                sorted_rt_early_novel = sorted_rt_early_novel/max(sorted_rt_early_novel);
                sorted_rt_late_novel = sorted_rt_late_novel/max(sorted_rt_late_novel);
                
                sorted_rt_all_fam = sorted_rt_all_fam/max(sorted_rt_all_fam);
                sorted_rt_early_fam = sorted_rt_early_fam/max(sorted_rt_early_fam);
                sorted_rt_late_fam = sorted_rt_late_fam/max(sorted_rt_late_fam);
                
                sorted_rt_all_ifam = sorted_rt_all_ifam/max(sorted_rt_all_ifam);
                sorted_rt_early_ifam = sorted_rt_early_ifam/max(sorted_rt_early_ifam);
                sorted_rt_late_ifam = sorted_rt_late_ifam/max(sorted_rt_late_ifam);
            end
            
            mfrate_all_novel(day,:)=sorted_rt_all_novel;
            mfrate_early_novel(day,:)=sorted_rt_early_novel;
            mfrate_late_novel(day,:)=sorted_rt_late_novel;
            
            mfrate_all_fam(day,:)=sorted_rt_all_fam;
            mfrate_early_fam(day,:)=sorted_rt_early_fam;
            mfrate_late_fam(day,:)=sorted_rt_late_fam;
            
            mfrate_all_ifam(day,:)=sorted_rt_all_ifam;
            mfrate_early_ifam(day,:)=sorted_rt_early_ifam;
            mfrate_late_ifam(day,:)=sorted_rt_late_ifam;
            
            day=day+1;
            clear img_no time_frame spk_tm spks rep mfrate frate_all y frate i image_no
            clear rt_all rt_early rt_late end_ind sorted_rt_all index_rt_all sorted_rt_early
            clear index_rt_early sorted_rt_late index_rt_late 
%         end
    end
end
%%
s_vsa = find(vsa<0.05);
length(s_vsa)
mfrate_sig_novel = mfrate_all_novel(s_vsa, :);
mfrate_sig_fam = mfrate_all_fam(s_vsa, :);
mfrate_sig_ifam = mfrate_all_ifam(s_vsa, :);

%%
jacknife = 1000;
ifam_imgs = 25;
nov_imgs = 50;
fam_imgs = 50;
conf_int = 99;
low_bar = (100 - conf_int)/2;
high_bar = conf_int + low_bar;
nov_neurs = length(mfrate_sig_novel);
fam_neurs = length(mfrate_sig_fam);
ifam_neurs = length(mfrate_sig_ifam);
novs_all = zeros(jacknife, nov_neurs, ifam_imgs);
fams_all = zeros(jacknife, fam_neurs, ifam_imgs);
ifams_all = zeros(jacknife, ifam_neurs, ifam_imgs);
norm_here = false;
for i = 1:jacknife
    for j = 1:nov_neurs
        keep = randperm(nov_imgs);
        keep = keep(1:ifam_imgs);
        newnovs = sort(mfrate_sig_novel(j, keep), 'descend');
        if norm_here
            newnovs = newnovs/max(newnovs);
        end 
        novs_all(i, j, :) = newnovs;
    end
    for j = 1:fam_neurs
        keep = randperm(fam_imgs);
        keep = keep(1:ifam_imgs);
        newfams = sort(mfrate_sig_fam(j, keep), 'descend');
        if norm_here
            newfams = newfams/max(newfams);
        end
        fams_all(i, j, :) = newfams;
    end
    for j = 1:ifam_neurs
        keep = datasample(1:ifam_imgs, ifam_imgs);
        newifams = sort(mfrate_sig_ifam(j, keep), 'descend');
        if norm_here
            newifams = newifams/max(newifams);
        end
        ifams_all(i, j, :) = newifams;
    end
end
novs_all_sj = squeeze(nanmean(novs_all, 2));
novs_all_jc = squeeze(nanmean(novs_all_sj, 1));
novs_all_var_low = novs_all_jc - prctile(novs_all_sj, low_bar, 1);
novs_all_var_high = prctile(novs_all_sj, high_bar, 1) - novs_all_jc;

fams_all_sj = squeeze(nanmean(fams_all, 2));
fams_all_jc = squeeze(nanmean(fams_all_sj, 1));
fams_all_var_low = fams_all_jc - prctile(fams_all_sj, low_bar, 1);
fams_all_var_high = prctile(fams_all_sj, high_bar, 1) - fams_all_jc;

ifams_all_sj = squeeze(nanmean(ifams_all, 2));
ifams_all_jc = squeeze(nanmean(ifams_all_sj, 1));
ifams_all_var_low = ifams_all_jc - prctile(ifams_all_sj, low_bar, 1);
ifams_all_var_high = prctile(ifams_all_sj, high_bar, 1) - ifams_all_jc;
%%
figure; set(gcf, 'Color','None');
subplot(111)

ifr_color = [218/256 112/256 214/256];
nov_color = [0 221/256 0];
fam_color = [0 0 221/256];
errorbar(1:ifam_imgs, fams_all_jc, fams_all_var_low, fams_all_var_high,...
    'Color', fam_color); hold all;
errorbar(1:ifam_imgs, ifams_all_jc, ifams_all_var_low, ifams_all_var_high,...
    'Color', ifr_color); hold all;
errorbar(1:ifam_imgs, novs_all_jc, novs_all_var_low, novs_all_var_high,...
    'Color', nov_color); hold all;

xlabel('image response rank');
ylabel('normalized firing rate');

box off
figname = 'dim_nfi_rankplot-nonorm';
print(gcf, figname, '-dsvg');

%%
% s_vsa=nbcomp{};
figure;set(gcf,'Color','w');
% subplot(131);
% plot(1:50,nanmean(mfrate_all_novel(s_vsa,:),1),'r','LineWidth',2);hold all;
% plot(1:50,nanmean(mfrate_all_fam(s_vsa,:),1),'b','LineWidth',2);hold all;
% plot(1:25,nanmean(mfrate_all_ifam(s_vsa,:),1),'g','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
% title('Rank plot - over all reps - red is novel, blue is familiar');xlabel('Rank');ylabel('Mean firing rate (Hz)');

subplot(121);
plot(1:50,nanmean(mfrate_early_novel(s_vsa,:),1),'r','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_early_fam(s_vsa,:),1),'b','LineWidth',2);hold all;
plot(1:25,nanmean(mfrate_early_ifam(s_vsa,:),1),'g','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
xlabel('Rank');
subplot(122);
plot(1:50,nanmean(mfrate_late_novel(s_vsa,:),1),'r','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_late_fam(s_vsa,:),1),'b','LineWidth',2);hold all;
plot(1:25,nanmean(mfrate_late_ifam(s_vsa,:),1),'g','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
xlabel('Rank');

%%
figure;set(gcf,'Color','w');
subplot(131);
plot(1:50,nanmean(mfrate_all_novel(s_vsa,:),1),'-','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_early_novel(s_vsa,:),1),'--','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_late_novel(s_vsa,:),1),':','LineWidth',2);hold all;
ylim([0 25]);xlim([0 60]);axis square;
title('nov - solid-all reps, dashed-early, dotted-late');xlabel('Rank');ylabel('Mean firing rate (Hz)');

subplot(132);
plot(1:50,nanmean(mfrate_all_fam(s_vsa,:),1),'-','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_early_fam(s_vsa,:),1),'--','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_late_fam(s_vsa,:),1),':','LineWidth',2);hold all;
ylim([0 25]);xlim([0 60]);axis square;
title('Fam - solid-all reps, dashed-early, dotted-late');xlabel('Rank');ylabel('Mean firing rate (Hz)');

subplot(133);
plot(1:25,nanmean(mfrate_all_ifam(s_vsa,:),1),'-','LineWidth',2);hold all
plot(1:25,nanmean(mfrate_early_ifam(s_vsa,:),1),'--','LineWidth',2);hold all
plot(1:25,nanmean(mfrate_late_ifam(s_vsa,:),1),':','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
title('IF - solid-all reps, dashed-early, dotted-late');xlabel('Rank');ylabel('Mean firing rate (Hz)');
%%
load wf;s_vsa = find(vsa<0.05);
swf = wf(:,s_vsa);
for nunit=1:length(swf)
    [~,maxi]=max(wf(:,nunit));[~,mini]=min(wf(:,nunit));
    ptw(nunit) = (abs(maxi-mini))*25;
end
ind_exc=find(ptw>397.5);
ind_inh=find(ptw<397.5);

figure;set(gcf,'Color','w');
subplot(131);
plot(1:50,nanmean(mfrate_all_novel(s_vsa(ind_inh),:),1),'r','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_all_fam(s_vsa(ind_inh),:),1),'b','LineWidth',2);hold all;
plot(1:25,nanmean(mfrate_all_ifam(s_vsa(ind_inh),:),1),'g','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
title('Rank plot - over all reps - red is novel, blue is familiar');xlabel('Rank');ylabel('Mean firing rate (Hz)');

subplot(132);
plot(1:50,nanmean(mfrate_early_novel(s_vsa(ind_inh),:),1),'r','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_early_fam(s_vsa(ind_inh),:),1),'b','LineWidth',2);hold all;
plot(1:25,nanmean(mfrate_early_ifam(s_vsa(ind_inh),:),1),'g','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
title('Rank plot - over early reps - red is novel, blue is familiar');xlabel('Rank');ylabel('Mean firing rate (Hz)');

subplot(133);
plot(1:50,nanmean(mfrate_late_novel(s_vsa(ind_inh),:),1),'r','LineWidth',2);hold all;
plot(1:50,nanmean(mfrate_late_fam(s_vsa(ind_inh),:),1),'b','LineWidth',2);hold all;
plot(1:25,nanmean(mfrate_late_ifam(s_vsa(ind_inh),:),1),'g','LineWidth',2);ylim([0 25]);xlim([0 60]);axis square;
title('Rank plot - over late reps - red is novel, blue is familiar');xlabel('Rank');ylabel('Mean firing rate (Hz)');
