% Define subjects and pollutants
subjects   = 1:20;
pollutants = {'CO2','P','PM1','PM10','PM25','RH','T','VOC'};

% Base directories
input_base  = '../data/raw_data/';
output_base = '../data/adapted_raw_data/';

for subj = subjects
    subj_str         = sprintf('S%02d', subj);    % e.g. 'S01'
    subj_str_no_zero = sprintf('S%d',   subj);    % e.g. 'S1'

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
                fn  = fields{f};
                val = data.(fn);

                if istable(val)
                    % write each table as Parquet
                    pq_name = sprintf('%s_%s_%s.parquet', ...
                                      subj_str_no_zero, pol, fn);
                    pq_path = fullfile(output_dir, pq_name);
                    parquetwrite(pq_path, val);
                    fprintf('Parquet saved: %s\n', pq_path);
                end
            end

        catch ME
            fprintf('Error with %s: %s\n', input_path, ME.message);
        end
    end
end
