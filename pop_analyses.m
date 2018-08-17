% function[mfrate]=pop_analyses()

%Population analysis for PSTH of novel and familiar across neurons
%Krithika Mohan
%24 April 2015
clc;clear all;
filedir = './krithika_dim_data_resorted/';
startdir = cd;
cd(filedir);
datafiles=dir;
day=1;scount=1;pcount=1;
for filenum=4:length(datafiles)
    cd(datafiles(filenum).name)
    %%
    %Plot average time course of N and F images
    load data; load x;
    cd ..
        
    ntrials=length(x);
    fam=1:50; nov=76:125;ifam=51:75;

    for trial_no=1:ntrials
        if ismember(x(trial_no).task_cond_num,1:6)
        if x(trial_no).TrialError ==0 %&& x(trial_no).task_cond_num>1 
            f_ind = find(ismember(x(trial_no).image_nos,fam));
            n_ind = find(ismember(x(trial_no).image_nos,nov));
            if_ind = find(ismember(x(trial_no).image_nos,ifam));
            ncount=1;fcount=1;ifcount=1;
            if ~isempty(f_ind)
                pos_trial(trial_no).f = f_ind;
            else
                pos_trial(trial_no).f=[];
            end
            if ~isempty(n_ind)
                pos_trial(trial_no).n = n_ind;
            else   
                pos_trial(trial_no).n = [];
            end
            if ~isempty(if_ind)
                pos_trial(trial_no).if = if_ind;
            else
                pos_trial(trial_no).if=[];
            end
        end
        end
    end
    %%
    for nunit = 1:length(x(1).spike_times)
        
        bin_size=5; k_mstep=25; k_pstep=20;tf=104; ff=95; sbs_v=6:100;bs_mean=(bin_size)*10;
        
        
        %%determine if this unit is visually selective through an ANOVA
        [pval] = anova_sel(data,x,nunit);
        vsa(pcount) = pval;
        pcount=pcount+1;
%         if pval > 0.05
%             break
%         else  
        sprintf('%s, %i',datafiles(filenum).name,nunit)
        scount=scount+1;
        ntrials=length(pos_trial);
        count=1;%bin_size=25;
        for trial_no=1:ntrials
            if ~isempty(pos_trial(trial_no).f)
                for sub_ind=1:length(pos_trial(trial_no).f)
                    nc_ms=x(trial_no).code_times - x(trial_no).code_times(1);
                    image_find = x(trial_no).image_nos(pos_trial(trial_no).f(sub_ind));
                    find_codetime = find(x(trial_no).code_numbers==(image_find+55));
                    if length(find_codetime)>1
                        find_codetime=find_codetime(2);
                    end
                    f_time_frame = nc_ms(find_codetime)-k_mstep:bin_size:nc_ms(find_codetime+1)+80+k_pstep;
%                     f_time_frame = nc_ms(find_codetime):bin_size:nc_ms(find_codetime+1)+80;
                    if ~isempty(f_time_frame)
%                         f_time_frame=f_time_frame(1:25);
                        f_time_frame=f_time_frame(1:tf);
                    end
                    spk_tm = x(trial_no).spike_times{1,nunit};
                    if isempty(spk_tm)
%                         frate_f(count,:)=NaN(1,25);   
                        frate_f(count,:)=NaN(1,ff);
                    else
                        fspks=histc(spk_tm,f_time_frame); 
                        if ~isempty(fspks)
    %                     if trial_no~=1
    %                         if length(fspks)> size(frate_f,2)
    %                             fspks = fspks(1:size(frate_f,2));
    %                         end
    %                     end
                                    
%                            frate_f(count,:) = (fspks*1000)/(bin_size);
                             for sbs=sbs_v
                                 ind_tf_bs = [sbs-5:sbs+4];
                                 fspks_bs(1,sbs-5)=sum(fspks(ind_tf_bs));
                             end
                             frate_f(count,:)=(fspks_bs*1000)/(bs_mean);
                        end
                    end
                    count=count+1;
                end
            end
        end
        
        
        %%        
        ntrials=length(pos_trial);
        count=1;
        for trial_no=1:ntrials
            if ~isempty(pos_trial(trial_no).n)
                for sub_ind=1:length(pos_trial(trial_no).n)
                    nc_ms=x(trial_no).code_times - x(trial_no).code_times(1);
                    image_nind = x(trial_no).image_nos(pos_trial(trial_no).n(sub_ind));
                    nind_codetime = find(x(trial_no).code_numbers==(image_nind+55));
                    if length(nind_codetime)>1
                        nind_codetime=nind_codetime(2);
                    end
%                     n_time_frame = nc_ms(nind_codetime):bin_size:nc_ms(nind_codetime+1)+80;
                    n_time_frame = nc_ms(nind_codetime)-k_mstep:bin_size:nc_ms(nind_codetime+1)+80+k_pstep;           
                    if ~isempty(n_time_frame)
%                         n_time_frame=n_time_frame(1:25);
                        n_time_frame=n_time_frame(1:tf);
                    end
                    spk_tm = x(trial_no).spike_times{1,nunit};
                    if isempty(spk_tm)
%                         frate_n(count,:)=NaN(1,25);
                        frate_n(count,:)=NaN(1,ff);

                    else
                        nspks=histc(spk_tm,n_time_frame); 
                    
                        if ~isempty(nspks)
    %                     if length(nspks)> size(frate_n,2)
    %                         nspks = nspks(1:size(frate_n,2));
    %                     end
                            for sbs=sbs_v
                                 ind_tf_bs = [sbs-5:sbs+4];
                                 nspks_bs(1,sbs-5)=sum(nspks(ind_tf_bs));
                             end
                        frate_n(count,:) = (nspks_bs*1000)/(bs_mean);

                        end
                    end

                    count=count+1;
                end
            end
        end
        %%
        ntrials=length(pos_trial);
        count=1;%bin_size=25;
        for trial_no=1:ntrials
            if ~isempty(pos_trial(trial_no).if)
                for sub_ind=1:length(pos_trial(trial_no).if)
                    nc_ms=x(trial_no).code_times - x(trial_no).code_times(1);
                    image_ifind = x(trial_no).image_nos(pos_trial(trial_no).if(sub_ind));
                    ifind_codetime = find(x(trial_no).code_numbers==(image_ifind+55));
                    if length(ifind_codetime)>1
                        ifind_codetime=ifind_codetime(2);
                    end
%                     if_time_frame = nc_ms(ifind_codetime):bin_size:nc_ms(ifind_codetime+1)+80;
                    if_time_frame = nc_ms(ifind_codetime)-k_mstep:bin_size:nc_ms(ifind_codetime+1)+80+k_pstep;
                   
                    if ~isempty(if_time_frame)
%                         if_time_frame=if_time_frame(1:25);
                        if_time_frame=if_time_frame(1:tf);
                    end
                    spk_tm = x(trial_no).spike_times{1,nunit};
                    if isempty(spk_tm)
%                         frate_if(count,:)=NaN(1,25);
                        frate_if(count,:)=NaN(1,ff);
                    else
                        ifspks=histc(spk_tm,if_time_frame); 
                    
                        if ~isempty(ifspks)
    %                     if length(ifspks)> size(frate_if,2)
    %                         ifspks = ifspks(1:size(frate_if,2));
    %                     end
                             for sbs=sbs_v
                                 ind_tf_bs = [sbs-5:sbs+4];
                                 ifspks_bs(1,sbs-5)=sum(ifspks(ind_tf_bs));
                             end
                            frate_if(count,:) = (ifspks_bs*1000)/(bs_mean);
                        end
                    end
                    count=count+1;
                end
            end
        end
                
        mfrate_ra5_50.f(day,:) = nanmean(frate_f,1);
        mfrate_ra5_50.n(day,:) = nanmean(frate_n,1);
        mfrate_ra5_50.if(day,:) = nanmean(frate_if,1);
         
        day=day+1;
%         end
    end
    clear x frate_n frate_f frate_if data ntrials fam nov ifam trial_no 
    clear f_ind n_ind if_ind ncount fcount ifcount pos_trial sub_ind nc_ms
    clear image_find find_codetime f_time_frame spk_tm fspks image_nind
    clear nind_codetime n_time_frame nspks maxffrate maxnfrate maxiffrate 
    clear fspks_bs nspks_bs ifspsk_bs sbs ind_tf_bs
%     cd ..
end
%%
% vsa=vs;
s_vsa = find(vsa<0.05);
% s_ifnrep = unit_info(s_vsa,5);
% fv=maxfrate.f(s_vsa,:);ifv=maxfrate.if(s_vsa,:);nv=maxfrate.n(s_vsa,:);
% fv=mfrate.f(s_vsa,:);ifv=mfrate.if(s_vsa,:);nv=mfrate.n(s_vsa,:);
fv=mfrate_ra5_50.f(s_vsa,:);ifv=mfrate_ra5_50.if(s_vsa,:);nv=mfrate_ra5_50.n(s_vsa,:);
% fv=maxfrate_ra.f(s_vsa,:);ifv=maxfrate_ra.if(s_vsa,:);nv=maxfrate_ra.n(s_vsa,:);

% f=fv;n=nv;ifr=ifv;
f=nanmean(fv,1);
n=nanmean(nv,1);
ifr=nanmean(ifv,1);

% lateviews = find(s_ifnrep>450);
% earlyviews = find(s_ifnrep<450);
% ifr_late = mean(ifv(lateviews,:),1);
% ifr_early=mean(ifv(earlyviews,:),1);

% figure;set(gcf,'Color',[242/255 237/255 235/255]);
figure;
set(gcf,'Color','w');
subplot(1,1,1);
hold all
% errorbar(0:10:460,f(1,1:47),f_se(1:47));
% errorbar(0:10:460,n(1,1:47),n_se(1:47));
% errorbar(0:10:460,ifr(1,1:47),ifr_se(1:47));

ifr_color = [218/256 112/256 214/256];
nov_color = [0 221/256 0];
fam_color = [0 0 221/256];
plot([0:5:465],f(1,1:94),'Color', fam_color,'LineWidth',1);hold all;
plot([0:5:465],ifr(1,1:94), 'Color', ifr_color, 'LineWidth',1);
plot([0:5:465],n(1,1:94),'Color', nov_color,'LineWidth',1);
xlabel('time (ms)');
ylabel('firing rate (spikes/s)');

% plot([0:10:460],ifr_late(1,1:47));
% plot([0:10:460],ifr_early(1,1:47));
legend('familiar (>2,000 times)','intermediately familiar (200-800 times)',...
    'novel (1-50 times)'); 
box off;
legend boxoff
figname = 'dim_nfi_plot';
print(gcf, figname, '-dsvg');
% legend('Familiar','Novel','IF','IF late - >500','IF early - <500');
% xlabel('Time (ms)');ylabel('Firing rate');
% 
% subplot(1,2,2)
% lateviews = find(s_ifnrep>450);
% earlyviews = find(s_ifnrep<450);
% ifr_late = mean(ifv(lateviews,:),1);
% ifr_early=mean(ifv(earlyviews,:),1);
%%

% % errorbar(0:10:460,f(1,1:47),f_se(1:47));
% % errorbar(0:10:460,n(1,1:47),n_se(1:47));
% % errorbar(0:10:460,ifr(1,1:47),ifr_se(1:47));
% 
% % plot([0:5:465],f(1,1:94));hold all;
% % plot([0:5:465],n(1,1:94));
subplot(1,2,2)
plot([0:5:465],ifr(1,1:94),'Color',[0 0.5 0],'LineWidth',1.5);hold on
plot([0:5:466],ifr_late(1,1:94),'Color',[0 1 0],'LineWidth',1.5);
plot([0:5:465],ifr_early(1,1:94),'Color',[154/255 205/255 50/155],'LineWidth',1.5);
% legend('Familiar','Novel','IF');      
legend('IF (200-800 views)','IF late (450-800 views)','IF early (200-450 views)');
% xlabel('Time aligned to image set (ms)');ylabel('Firing rate (Hz)');set(gcf,'Color',[242/255 237/255 235/255])
title('Peri-stimulus time histogram (N=79/99)');

%%
% load wf;s_vsa = find(vsa<0.05);
% swf = wf(:,s_vsa);
% for nunit=1:length(swf)
%     [~,maxi]=max(wf(:,nunit));[~,mini]=min(wf(:,nunit));
%     ptw(nunit) = (abs(maxi-mini))*25;
% end
% ind_exc=find(ptw>397.5);
% ind_inh=find(ptw<397.5);
% 
% % fv=mfrate.f(s_vsa,:);ifv=mfrate.if(s_vsa,:);nv=mfrate.n(s_vsa,:);
% fv=mfrate_ra.f(s_vsa,:);ifv=mfrate_ra.if(s_vsa,:);nv=mfrate_ra.n(s_vsa,:);
% 
% fve=fv(ind_exc,:);ifve=ifv(ind_exc,:);nve=nv(ind_exc,:);
% 
% fe=nanmean(fve,1);
% ne=nanmean(nve,1);
% ife=nanmean(ifve,1);
% 
% fvi=fv(ind_inh,:);ifvi=ifv(ind_inh,:);nvi=nv(ind_inh,:);
% 
% fi=nanmean(fvi,1);
% ni=nanmean(nvi,1);
% ifi=nanmean(ifvi,1);
% % 
% figure;subplot(121);set(gcf,'Color','w');
% plot([0:10:460],fe(1,1:47));hold all;
% plot([0:10:460],ne(1,1:47));
% plot([0:10:460],ife(1,1:47));
% % plot([0:20:470],ifr_late(1,1:24));
% % plot([0:20:470],ifr_early(1,1:24));
% legend('Familiar','Novel','IF');
% % legend('Familiar','Novel','IF','IF late - >500','IF early - <500');
% xlabel('Time (ms)');ylabel('Firing rate');title('PSTH of excitatory neurons (N=42)');
% 
% subplot(122);
% plot([0:10:460],fi(1,1:47));hold all;
% plot([0:10:460],ni(1,1:47));
% plot([0:10:460],ifi(1,1:47));
% % plot([0:20:470],ifr_late(1,1:24));
% % plot([0:20:470],ifr_early(1,1:24));
% legend('Familiar','Novel','IF');
% % legend('Familiar','Novel','IF','IF late - >500','IF early - <500');
% xlabel('Time (ms)');ylabel('Firing rate');title('PSTH of inhibitory neurons (N=37)');
% %%
% nexp=unit_info(s_vsa,5);
% nei=unit_info(s_vsa,6);
% ni=mean(nexp(nei==2));
% ne=mean(nexp(nei==1));
% ni_m=median(nexp(nei==2));
% ne_m=median(nexp(nei==1));
% %%
% figure;set(gcf,'Color','w');
% 
% lvsub1=find(s_ifnrep(lateviews)<500);
% lvsub2=find(s_ifnrep(lateviews)>500);
% ifv_late=mfrate_ra.if(s_vsa(lateviews),:);
% ifr_late_sub1 = mean(ifv_late(lvsub1,:),1);
% ifr_late_sub2 = mean(ifv_late(lvsub2,:),1);
% 
% plot([0:10:460],ifr_late_sub1(1,1:47),'-b');hold all;
% plot([0:10:460],ifr_late_sub2(1,1:47),'-r');
% plot([0:10:460],ifr_late(1,1:47),'*-k'); ylim([4.5 9]);axis square
% %%
% figure;set(gcf,'Color','w');
% 
% for i=1:9
%     subplot(3,3,i);
%     lvsub=randperm(length(lateviews));
%     lvsub1=lvsub(1:length(lateviews)/2);lvsub2=lvsub((length(lateviews)/2)+1:length(lateviews));
%     ifv_late=mfrate_ra.if(s_vsa(lateviews),:);
%     ifr_late_sub1 = mean(ifv_late(lvsub1,:),1);
%     ifr_late_sub2 = mean(ifv_late(lvsub2,:),1);
% 
%     plot([0:10:460],ifr_late_sub1(1,1:47),'-b');hold all;
%     plot([0:10:460],ifr_late_sub2(1,1:47),'-r');
%     plot([0:10:460],ifr_late(1,1:47),'*-k'); ylim([4 10]);axis square
% end
%%
s_vsa = find(vsa<0.05);

fv=mfrate_ra5_50.f(s_vsa,:);ifv=mfrate_ra5_50.if(s_vsa,:);nv=mfrate_ra5_50.n(s_vsa,:);
nnum=11;
% for nnum=1:size(fv,1)
    figure;set(gcf,'Color','w');
    plot([0:5:465],fv(nnum,1:94),'b'); hold all;
    plot([0:5:465],nv(nnum,1:94),'r');
    plot([0:5:465],ifv(nnum,1:94),'g');
%     legend('Familiar','Novel','IF');      
    xlabel('Time (ms)');ylabel('Firing rate');%title(sprintf('PSTH, Neuron %i, step=5 ms, bin=50 ms',nnum));
%     export_fig fnif_nbyn.pdf -nocrop -append 
%     close all;
% end
%%
s_vsa = find(vsa<0.05);
fv=mfrate_ra5_50.f(s_vsa,:);ifv=mfrate_ra5_50.if(s_vsa,:);nv=mfrate_ra5_50.n(s_vsa,:);

for nnum=1:size(fv,1)
    frate_nfif(nnum,1)=nanmean(fv(nnum,:));
    frate_nfif(nnum,2)=nanmean(nv(nnum,:));
    frate_nfif(nnum,3)=nanmean(ifv(nnum,:));
end
