% Define subjects and pollutants
subjects = 1:20;
pollutants = {'CO2', 'P', 'PM1', 'PM10', 'PM25', 'RH', 'T', 'VOC'};

% Base directories
input_base = '../data/raw_data/';
output_base = '../data/modified/';

for subj = subjects
    % Ensure the subject folder name includes leading zero
    subj_str = sprintf('S%02d', subj);  % Folder: S01, S02, etc.
    subj_str_no_zero = sprintf('S%d', subj);  % Filename: S1, S2, etc.
    input_dir = fullfile(input_base, subj_str);
    output_dir = fullfile(output_base, subj_str);

    % Create the output folder if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    for i = 1:numel(pollutants)
        pol = pollutants{i};
        % Ensure the filename without the leading zero for .mat
        filename = sprintf('%s%s.mat', subj_str_no_zero, pol);
        input_path = fullfile(input_dir, filename);

        try
            % Load the .mat file
            data = load(input_path);
            fields = fieldnames(data);

            for f = 1:numel(fields)
                fieldName = fields{f};
                val = data.(fieldName);

                % If it's a table, save it as CSV
                if istable(val)
                    % Convert datetime columns to strings
                    for col = 1:width(val)
                        if isdatetime(val.(col))
                            val.(col) = cellstr(datestr(val.(col), 'HH:MM:SS'));
                        end
                    end

                    % Save to CSV
                    csv_filename = sprintf('%s_%s_%s.csv', subj_str_no_zero, pol, fieldName);
                    csv_path = fullfile(output_dir, csv_filename);
                    writetable(val, csv_path);
                    fprintf('CSV saved: %s\n', csv_path);

                    % Remove this field from the struct before saving .mat
                    data.(fieldName) = [];
                    continue;
                end

                % If datetime, convert it to string
                if isdatetime(val)
                    strVal = datestr(val, 'HH:MM:SS');
                    data.(fieldName) = strVal;
                end
            end

            % Clean up empty fields
            fields = fieldnames(data);
            fields = fields(~cellfun(@(f) isempty(data.(f)), fields));

            % Assign variables to base workspace
            for f = 1:numel(fields)
                assignin('base', fields{f}, data.(fields{f}));
            end

            % Save the remaining data as .mat with subject name without leading zero
            output_path = fullfile(output_dir, sprintf('%s%s.mat', subj_str_no_zero, pol));
            save(output_path, fields{:});
            fprintf('MAT saved: %s\n', output_path);

        catch ME
            fprintf('Error with file %s: %s\n', input_path, ME.message);
        end
    end
end
