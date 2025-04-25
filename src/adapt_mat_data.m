% Define subjects and pollutants
subjects   = 1:20;
pollutants = {'CO2','P','PM1','PM10','PM25','RH','T','VOC'};

% Base directories
input_base  = '../data/raw_data/';
output_base = '../data/adapted_raw_data/';

for subj = subjects
    subj_str        = sprintf('S%02d', subj);    % S01, S02, …
    subj_str_no_zero= sprintf('S%d',   subj);    % S1, S2, …

    input_dir  = fullfile(input_base,  subj_str);
    output_dir = fullfile(output_base, subj_str);
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end

    for i = 1:numel(pollutants)
        pol       = pollutants{i};
        filename  = sprintf('%s%s.mat', subj_str_no_zero, pol);
        input_path= fullfile(input_dir, filename);

        try
            data   = load(input_path);
            fields = fieldnames(data);

            for f = 1:numel(fields)
                fn   = fields{f};
                val  = data.(fn);

                if istable(val)
                    % — write parquet instead of CSV —
                    pq_name = sprintf('%s_%s_%s.parquet', ...
                                      subj_str_no_zero, pol, fn);
                    pq_path = fullfile(output_dir, pq_name);
                    parquetwrite(pq_path, val);
                    fprintf('Parquet saved: %s\n', pq_path);

                    % remove table from data before saving .mat
                    data.(fn) = [];
                    continue;
                end

                % if you still need to convert datetime fields outside tables
                if isdatetime(val)
                    data.(fn) = datestr(val,'HH:MM:SS');
                end
            end

            % clean up any emptied fields
            fields = fieldnames(data);
            keep   = ~cellfun(@(f) isempty(data.(f)), fields);
            fields = fields(keep);

            % assign remaining to workspace and save .mat
            for f = 1:numel(fields)
                assignin('base', fields{f}, data.(fields{f}));
            end
            out_mat = fullfile(output_dir, sprintf('%s%s.mat',subj_str_no_zero,pol));
            save(out_mat, fields{:});
            fprintf('MAT saved: %s\n', out_mat);

        catch ME
            fprintf('Error with %s: %s\n', input_path, ME.message);
        end
    end
end
