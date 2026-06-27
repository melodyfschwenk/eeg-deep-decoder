%% ================================================================
%% Export labeled epochs WITHOUT EEGLAB.
%% Reads each .set as a plain MATLAB file and its .fdt as raw
%% float32, so it never calls eeglab/pop_loadset/eeglab_options
%% (the functions that were recursing and hanging on your machine).
%%
%% Method verified against the SCCN reference (Makoto's EEGLAB code)
%% and MNE-Python's EEGLAB loader: .fdt is float32, channels x
%% points x trials, read one trial at a time.
%%
%% You can run this in a plain MATLAB session. EEGLAB does NOT
%% need to be on the path.
%% ================================================================
clear; clc;

base   = 'C:\Users\melod\OneDrive\Desktop\eeg';
proc   = fullfile(base, 'processed_data');
outdir = fullfile(base, 'export_for_python');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% Label -> integer code maps (must match the Python side)
nav_keys  = {'egocentric','allocentric','control','none'};
nav_vals  = [0, 1, 2, 2];
diff_keys = {'easy','hard','control','none'};
diff_vals = [0, 1, 2, 2];

groups = {'deaf','hearing'};
n_done = 0;

for g = 1:numel(groups)
    folder = fullfile(proc, groups{g});
    files  = dir(fullfile(folder, '*_maxretention.set'));

    for f = 1:numel(files)
        fn  = files(f).name;
        pid = erase(fn, '_maxretention.set');

        try
            fprintf('Reading %s ...\n', fn);

            % --- load the .set as a plain MATLAB struct (no EEGLAB) ---
            S = load('-mat', fullfile(folder, fn));
            if isfield(S, 'EEG'); EEG = S.EEG; else; EEG = S; end

            nbchan = EEG.nbchan;
            pnts   = EEG.pnts;
            trials = EEG.trials;

            % --- get the time series ---
            if ischar(EEG.data)
                % data lives in the .fdt next to the .set
                fdt = fullfile(folder, strrep(fn, '.set', '.fdt'));
                fid = fopen(fdt, 'r', 'ieee-le');
                if fid < 0; error('cannot open %s', fdt); end
                data3 = zeros(nbchan, pnts, trials, 'single');
                for tr = 1:trials
                    data3(:,:,tr) = fread(fid, [nbchan pnts], 'float32');
                end
                fclose(fid);
            else
                data3 = EEG.data;   % already inline
            end

            % --- condition/difficulty labels (attached by your pipeline) ---
            if ~isfield(EEG, 'epoch') || ~isfield(EEG.epoch, 'navigation')
                warning('%s: no navigation field on epochs. Skipping.', pid);
                continue;
            end
            nav = -ones(trials, 1);
            dff = -ones(trials, 1);
            acc = nan(trials, 1);
            rt  = nan(trials, 1);
            for i = 1:trials
                nv = EEG.epoch(i).navigation;  if iscell(nv); nv = nv{1}; end
                dv = EEG.epoch(i).difficulty;  if iscell(dv); dv = dv{1}; end
                if ischar(nv)
                    k = find(strcmpi(nav_keys, nv), 1);  if ~isempty(k); nav(i) = nav_vals(k); end
                end
                if ischar(dv)
                    k = find(strcmpi(diff_keys, dv), 1); if ~isempty(k); dff(i) = diff_vals(k); end
                end
                if isfield(EEG.epoch, 'accuracy')
                    av = EEG.epoch(i).accuracy; if iscell(av); av = av{1}; end
                    if ~isempty(av) && isnumeric(av); acc(i) = double(av); end
                end
                if isfield(EEG.epoch, 'rt')
                    rv = EEG.epoch(i).rt; if iscell(rv); rv = rv{1}; end
                    if ~isempty(rv) && isnumeric(rv); rt(i) = double(rv); end
                end
            end

            data       = permute(data3, [3 1 2]);    % trial x channel x time
            chanlabels = {EEG.chanlocs.labels};
            srate      = EEG.srate;
            group      = groups{g};

            save(fullfile(outdir, [pid '.mat']), ...
                 'data', 'nav', 'dff', 'acc', 'rt', 'srate', 'chanlabels', 'group', 'pid', '-v7');

            n_done = n_done + 1;
            fprintf('%-6s exported: %d trials  (ego=%d allo=%d ctrl=%d)\n', ...
                pid, trials, sum(nav==0), sum(nav==1), sum(nav==2));

        catch ME
            warning('%s: failed (%s). Skipping.', pid, ME.message);
            continue;
        end
    end
end

fprintf('\nDone. %d participants written to %s\n', n_done, outdir);
