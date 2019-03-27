% load matfile
ffile = 'C:\Repositories\pyAPES_Samuli\pyAPES_utilities\Hy2005APES.mat';
load(ffile);
dat = Hy05APES;
D = struct2cell(dat);

fn = fieldnames(dat);

%outfile
outf='Hy2005_';

%% massage struct

sv = {'year'; 'month'; 'day'; 'hour'; 'minute'};
mv = {};
A = [];
A(:,1:5) = D{1}; %year month day hour minute
j = 6; m = 1; n = 1;

for k=2:length(fn),
    [r, c] = size(D{k});
    if r > 1000 && c ==1, 
        A(:,5+n) = D{k,:};
        sv{j} = fn{k};
        j = j+1;
        n = n+1;
    elseif r > 0,
        mv{m} = fn{k};
        m = m+1;
    end
end

%% now loop for few variables in mv

% storage fluxes
f = fieldnames(dat.StorageFlux);
b = struct2cell(dat.StorageFlux);
[r,c] = size(b);
[ra, ca] = size(A);
for k = 1:r,
    A(:, ca+k) = b{k};
    sv{end+1} =strcat('Fst_', f{k});
end

%fluxflags
f = fieldnames(dat.Flags);
b = struct2cell(dat.Flags);
[r,c] = size(b);
[ra, ca] = size(A);
for k = 1:r,
    A(:, ca+k) = b{k};
    sv{end+1} =strcat('QC_', f{k});
end

% soil temperature
f = fieldnames(dat.Tsoil);
b = struct2cell(dat.Tsoil);
[r,c] = size(b);
[ra, ca] = size(A);
for k = 1:r,
    A(:, ca+k) = b{k};
    sv{end+1} =strcat('Ts_', f{k});
end

% soil moisture
f = fieldnames(dat.SWC);
b = struct2cell(dat.SWC);
[r,c] = size(b);
[ra, ca] = size(A);
for k = 1:r,
    A(:, ca+k) = b{k};
    sv{end+1} =strcat('W_', f{k});
end

% Runoff

%f = fieldnames(dat.Runoff);
b = dat.Runoff;
[r,c] = size(b);
[ra, ca] = size(A);
for k = 1:c,
    A(:, ca+k) = b(:,k);
    sv{end+1} =strcat('Roff_', num2str(k));
end

% Trfallraw
b = dat.Trfallraw;

[r,c] = size(b);
[ra, ca] = size(A);
for k = 1:c,
    A(:, ca+k) = b(:,k);
    sv{end+1} =strcat('Trfall_', num2str(k));
end

%% convert to table and save to csv

 tt = array2table(A, 'VariableNames', sv);
 writetable(tt, 'Hyde2005.csv', 'Delimiter', ';');

%% sub-canopy radiation

d = dat.SubRad;

rad = [d.time d.Par d.Rnet];
cc = {'year'; 'month'; 'day'; 'hour'; 'minute'; 'Par1'; 'Par2'; 'Par3'; 'Par4'; ...
     'Rnet1'; 'Rnet2'; 'Rnet3'; 'Rnet4'; 'Rnet5'};
 
 rad = array2table(rad, 'VariableNames', cc);
 writetable(rad, 'Hyde2005_SubcanopyRadi.csv', 'Delimiter', ';');

 %% moss water content
d = dat.MossWaterContent.Auto

dd = [d.time zeros(length(d.time),1) d.MWC];
dd = array2table(dd, 'VariableNames', {'year'; 'month'; 'day'; 'hour'; 'minute'; 'MWC'});
writetable(dd, 'Hyde2005_MossMoisture_Autoch.csv', 'Delimiter', ';');

d = dat.MossWaterContent.Manual;

dd = [d.time zeros(length(d.time),1) d.MWC];
dd = array2table(dd, 'VariableNames', {'year'; 'month'; 'day'; 'hour'; 'minute'; 'MWC_1'; 'MWC_2';...
    'MWC_3'; 'MWC_4'; 'MWC_5'; });
writetable(dd, 'Hyde2005_MossMoisture_Manualch.csv', 'Delimiter', ';');
 % mv = 
% 
%   Columns 1 through 5
% 
%     'PrecFMI'    'Trfallraw'    'Runoff'    'FluxFiles'    'Flags'
% 
%   Columns 6 through 10
% 
%     'Fluxheights'    'Gmeas'    'StorageFlux'    'Tsoil'    'SWC'
% 
%   Columns 11 through 14
% 
%     'SoilDepths'    'StoneContent'    'Profile'    'ShootCuvette'
% 
%   Columns 15 through 19
% 
%     'Snow'    'ForcingFile'    'SubRad'    'SoilRespi'    'SapFlow'
% 
%   Columns 20 through 23
% 
%     'MossWaterContent'    'FloorCh'    'Pits'    'MossFc'