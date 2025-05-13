% Define subjects and pollutants
subjects   = 1:20;
pollutants = {'CO2','P','PM1','PM10','PM25','RH','T','VOC'};

% Base directories
input_base  = 'raw_data/';
output_base = 'adapted_raw_data/';

for subj = subjects
    subj_str         = sprintf('S%02d', subj);    % S01, S02, …
    subj_str_no_zero = sprintf('S%d',   subj);    % S1, S2, …

    input_dir  = fullfile(input_base,  subj_str);
    output_dir = fullfile(output_base, subj_str);
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end

    for i = 1:numel(pollutants)
        pol        = pollutants{i};
        filename   = sprintf('%s%s.mat', subj_str_no_zero, pol);
        input_path = fullfile(input_dir, filename);

        try
            data   = load(input_path);
            fields = fieldnames(data);

            % Try to find TableResampled and S
            has_table_resampled = any(strcmp(fields, 'TableResampled'));
            has_S = any(strcmp(fields, 'S'));

            if has_table_resampled
                T = data.TableResampled;

                % If S exists and TableResampled has only 2 columns (time + 1 var)
                if has_S && width(T) == 2
                    S = data.S;
                    if size(S,2) >= 2
                        signal_values = S(:,2);
                        T.signal = signal_values;
                    else
                        warning('S does not have a second column (signal) for %s%s.mat', subj_str_no_zero, pol);
                    end
                end

                % Save the (possibly modified) TableResampled as parquet
                pq_name = sprintf('%s_%s_TableResampled.parquet', subj_str_no_zero, pol);
                pq_path = fullfile(output_dir, pq_name);
                parquetwrite(pq_path, T);
                fprintf('Parquet saved: %s\n', pq_path);
            end

            % For the other tables (if you want to save them), you can handle similarly
            % This loop saves all remaining fields, after handling TableResampled separately
            fields = setdiff(fields, {'TableResampled'});

            % Clean and save remaining .mat
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
